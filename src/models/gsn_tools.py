import gc
from typing import Any, List, Union

import numpy as np
import torch
from datasets.features.features import Optional
from PIL import Image

from ..utils import RankedLogger
from .gsn_config import DistribConfig, GsngConfig, IterefConfig
from .gsn_criterion import AttendAndExciteGSN, BoxDiffGSN, RetentionLoss
from .gsn_criterion.utils_attention import AttentionStore
from .utils_distrib import (
    compute_with_var,
    generate_bayer_matrix,
    generate_random_colored_image,
    get_sigma_init,
    numpy_to_image_bayer_matrix,
    return_sigma,
)

logger = RankedLogger(__name__, rank_zero_only=True)

from concurrent.futures import ThreadPoolExecutor

from diffusers.utils.torch_utils import randn_tensor

from .gsn_criterion.initno import InitNOGSN

AVAILABLE_MODEL = ["sd1.4", "sd3", "flux"]


def update_latent(
    latents: torch.Tensor, loss: torch.Tensor, step_size: float
) -> torch.Tensor:
    """Update the latent according to the computed loss."""
    grad_cond = torch.autograd.grad(
        loss.requires_grad_(True), [latents], retain_graph=True
    )[0]
    latents = latents - step_size * grad_cond
    return latents


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class ToolsClassMixin:
    """Mixin that implements GSN-related guidance utilities for diffusion pipelines.

    This mixin centralizes helper methods used during inference to:
    - Configure and validate GSN criteria (ITEREF, GSNG, SAGA/distribution).
    - Prepare and transform latent tensors across supported models.
    - Run optimization loops that update latents from attention-based losses.
    - Manage attention stores, intermediate artifacts, and memory cleanup.

    The class is designed to be composed with a diffusion pipeline-like object that
    already exposes model components such as scheduler, VAE, UNet/Transformer, and
    pipeline utility methods (for example `progress_bar`, `maybe_free_model_hooks`,
    and model-specific latent pack/unpack helpers).

    Attributes:
        scheduler (Any): Active scheduler used during denoising and timestep logic.
        vae (Any): VAE module used to encode image inputs into latent space.
        image_processor (Any): Image preprocessor used before VAE encoding.
        _execution_device (Any): Device used for runtime execution.
        unet (Any): UNet model used by SD1.4 pipelines.
        transformer (Any): Transformer model used by SD3/Flux pipelines.
        attention_store (Optional[AttentionStore]): Runtime storage for attention maps.
        joint_attention_kwargs (Any): Additional attention kwargs forwarded to models.
        intermediate_values (Any): Dictionary-like container for debug/intermediate outputs.
        save_executor (Optional[ThreadPoolExecutor]): Optional worker pool for async saves.
        store_attention_path (Optional[str]): Output path where attention maps are stored.
    """

    scheduler: Any
    vae: Any
    image_processor: Any
    _execution_device: Any
    unet: Any
    transformer: Any
    attention_store: Optional[AttentionStore] = None
    joint_attention_kwargs: Any
    intermediate_values: Any
    save_executor: Optional[ThreadPoolExecutor] = None
    store_attention_path: Optional[str] = None

    def release_memory(self):
        if self.attention_store is not None:
            self.attention_store.reset()
        torch.cuda.empty_cache()
        gc.collect()

    def wait_for_pending_saves(self):
        if self.save_executor is not None:
            self.save_executor.shutdown(wait=True)
            self.save_executor = None
        self.store_attention_path = None

    def setup_save_executor(
        self,
        store_attention_path: str,
        max_save_workers: int = 4,
    ):
        if store_attention_path is None:
            return
        logger.info(f"Storing attention maps to {store_attention_path}")
        self.save_executor = ThreadPoolExecutor(max_workers=max_save_workers)
        self.store_attention_path = store_attention_path

    def init_intermediate_values(self):
        self.intermediate_values = {}

    def define_momentum_distrib(self, momentum_saga):
        self.momentum_saga = momentum_saga

    def setup_params(self, batch_size, num_images_per_prompt):
        if self.criterion_iteref is not None:
            self.steps_iteref = self.update_steps_gsn(self.steps_iteref)
            self.step_size_iteref = self.get_step_size(
                self.scale_range_iteref,
                self.scale_factor_iteref,
            )
        if self.criterion_gsng is not None:
            self.steps_gsng = self.update_steps_gsn(self.steps_gsng)
            self.step_size_gsng = self.get_step_size(
                self.scale_range_gsng,
                self.scale_factor_gsng,
            )

        self.check_inputs_gsn(
            self.extra_params_iteref_default,
            self.extra_params_gsng_default,
            self.extra_params_distrib_default,
            batch_size,
        )

        (
            self.extra_params_iteref_updated,
            self.extra_params_gsng_updated,
            self.extra_params_distrib_updated,
        ) = self.update_extra_parameters(
            extra_params_iteref=self.extra_params_iteref_default,
            extra_params_gsng=self.extra_params_gsng_default,
            extra_params_distrib=self.extra_params_distrib_default,
            num_images_per_prompt=(
                1
                if (
                    self.criterion_distrib is not None
                    and not self.one_image_per_distrib
                )
                else num_images_per_prompt
            ),
            batch_size=batch_size,
        )

        for criterion in [
            self.criterion_iteref,
            self.criterion_gsng,
            self.criterion_distrib,
        ]:
            if criterion is not None:
                criterion.set_version(self.model)

    def check_one_image_per_distrib(
        self, list_generators_distrib, list_generator_latents
    ):
        if self.one_image_per_distrib:
            if not isinstance(list_generators_distrib, list):
                raise ValueError(
                    f"You have set one_image_per_distrib to True, but list_generators_distrib is not a list, but {type(list_generators_distrib)}"
                )
            if not isinstance(list_generator_latents, list):
                raise ValueError(
                    f"You have set one_image_per_distrib to True, but list_generator_latents is not a list, but {type(list_generator_latents)}"
                )

            if len(list_generators_distrib) != len(list_generator_latents):
                raise ValueError(
                    "You have set one_image_per_distrib to True, but the length of list_generators_distrib and list_generator_latents does not match."
                )

    def set_criterions(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        model: Optional[str] = None,
        return_intermediate_features: Optional[bool] = None,
        iteref_config: Optional[IterefConfig] = None,
        gsng_config: Optional[GsngConfig] = None,
        distrib_config: Optional[DistribConfig] = None,
    ):
        # Iterative refinement
        if iteref_config:
            self.criterion_iteref = iteref_config.criterion
            self.extra_params_iteref_default = iteref_config.extra_params
            self.steps_iteref = iteref_config.steps
            self.scale_range_iteref = iteref_config.scale_range
            self.scale_factor_iteref = iteref_config.scale_factor
            self.max_opti_iteref = iteref_config.max_opti
            self.optimizer_class_iteref = iteref_config.optimizer_class
            self.thresholds_iteref = iteref_config.thresholds
        else:
            self.criterion_iteref = None
            self.extra_params_iteref_default = None
            self.steps_iteref = None
            self.scale_range_iteref = None
            self.scale_factor_iteref = None
            self.max_opti_iteref = None
            self.optimizer_class_iteref = None
            self.thresholds_iteref = None

        self.extra_params_iteref_updated = None

        # GSN Guidance
        if gsng_config:
            self.criterion_gsng = gsng_config.criterion
            self.extra_params_gsng_default = gsng_config.extra_params
            self.steps_gsng = gsng_config.steps
            self.scale_range_gsng = gsng_config.scale_range
            self.scale_factor_gsng = gsng_config.scale_factor
        else:
            self.criterion_gsng = None
            self.extra_params_gsng_default = None
            self.steps_gsng = None
            self.scale_range_gsng = None
            self.scale_factor_gsng = None
        self.extra_params_gsng_updated = None

        # SAGA
        if distrib_config:
            self.criterion_distrib = distrib_config.criterion
            self.extra_params_distrib_default = distrib_config.extra_params
            self.optimizer_class = distrib_config.optimizer_class
            self.step_distrib = distrib_config.step
            self.step_size_distrib = distrib_config.step_size
            self.max_opti_distrib = distrib_config.max_opti
            self.block = distrib_config.block
            self.log_var = distrib_config.log_var
            self.per_channel = distrib_config.per_channel
            self.init_mu = distrib_config.init_mu
            self.batch_size_noise = distrib_config.batch_size_noise
            self.rescale = distrib_config.rescale
            self.one_image_per_distrib = distrib_config.one_image_per_distrib
            self.define_momentum_distrib(distrib_config.momentum_saga)
        else:
            self.criterion_distrib = None
            self.extra_params_distrib_default = None
            self.optimizer_class = None
            self.step_distrib = None
            self.step_size_distrib = None
            self.max_opti_distrib = None
            self.block = None
            self.log_var = None
            self.per_channel = None
            self.init_mu = None
            self.batch_size_noise = None
            self.rescale = None
            self.one_image_per_distrib = False
            self.define_momentum_distrib(0.0)

        self.extra_params_distrib_updated = None

        # generation info
        self.model = model  #! uptade for flux
        if model is not None and model not in AVAILABLE_MODEL:
            raise ValueError(f"model should be in {AVAILABLE_MODEL}, not {model}")

        self.height = height
        self.width = width
        self.return_intermediate_features = return_intermediate_features
        self.check_optimizer(self.optimizer_class_iteref if iteref_config else None)
        self.check_optimizer(self.optimizer_class if distrib_config else None)
        self.mu_reference_rescale: Optional[torch.Tensor] = None
        self.collect_mu_reference_rescale = None

    def should_desactivate_distrib(self, num_inference_steps):
        if self.step_distrib is None or (
            self.step_distrib < 0 and self.step_distrib >= num_inference_steps
        ):
            self.criterion_distrib = None

    def check_optimizer(self, optimizer):
        if optimizer not in ["Adam", "SGD"] and optimizer is not None:
            raise ValueError(
                f"optimizer_class should be Adam or SGD or None, not {optimizer}"
            )

    def get_log_var2(self, latent_example_dim, alphas, t):
        if self.model in ["sd3", "flux"]:
            # alphas[t] is the std so we need to square it but with log is equivalent to multiply by 2
            return 2 * torch.log(
                torch.ones_like(latent_example_dim) * (alphas[self.convert_t_sd3(t)])
            )
        else:  # (1 - alphas[t]) already the var
            return torch.log(torch.ones_like(latent_example_dim) * (1 - alphas[t]))

    def get_shape_latents_flux(self, latents):
        batch_size, _, _ = latents.shape
        num_channels_latents = self.transformer.config.in_channels // 4
        height = 2 * (int(self.height) // self.vae_scale_factor)
        width = 2 * (int(self.width) // self.vae_scale_factor)
        return batch_size, num_channels_latents, height, width

        # params du pack_latent sont les dimensions du latents
        # unpack latent prends
        # _unpack_latents(latents, self.height, self.width, self.vae_scale_factor)

    def get_sample(self, mu, var, noise):
        if self.model in ["flux"]:
            mu = self._unpack_latents(
                mu, self.height, self.width, self.vae_scale_factor
            )
        samples = compute_with_var(
            mu=mu,
            var=var,
            noise=noise,
            log_var=self.log_var,
            per_channel=self.per_channel,
            block=self.block,
        )
        if self.model in ["flux"]:
            batch_size, channels, height, width = samples.shape
            return self._pack_latents(
                samples,
                batch_size,
                channels,
                height,
                width,
            )
        return samples

    def convert_t_sd3(self, t):
        if self.model in ["sd3", "flux"]:
            if self.scheduler.step_index is None:
                self.scheduler._init_step_index(t)
            return self.scheduler.step_index
        else:
            return t

    def get_var(self, latents, alphas, t, per_channel, log_var, block):
        if self.model in ["flux"]:
            batch_size, channel, height, width = self.get_shape_latents_flux(latents)
        else:
            batch_size, channel, height, width = latents.shape

        t = self.convert_t_sd3(t)
        return get_sigma_init(
            per_channel=per_channel,
            log_var=log_var,
            std=(
                alphas[self.convert_t_sd3(t)].cpu().numpy()
                if self.model in ["sd3", "flux"]
                else np.sqrt(1 - alphas[t])
            ),
            batch_size=batch_size,
            block=block,
            channel=channel,
            height=height,
            width=width,
        )

    def scale_x0(self, x0, alphas, t):
        if self.model in ["sd3", "flux"]:
            t = self.convert_t_sd3(t)
            return (1.0 - alphas[t]) * x0
        else:
            return (alphas[t] ** 0.5) * x0

    def unscale_x0(self, x0, alphas, t):
        t = self.convert_t_sd3(t)
        return (
            x0 / (1.0 - alphas[t].to(x0.device))
            if self.model in ["sd3", "flux"]
            else x0 / alphas[t] ** 0.5
        )

    def get_return_var(self, var, latents):
        if self.model in ["flux"]:
            latents = self._unpack_latents(
                latents, self.height, self.width, self.vae_scale_factor
            )

        return (
            return_sigma(
                var=var,
                latents=latents,
                per_channel=self.per_channel,
                log_var=self.log_var,
                block=self.block,
            )
            .clone()
            .detach()
            .cpu()
        )

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        if self.model in ["flux", "sd3"]:
            image_latents = (
                image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        elif self.model == "sd1.4":
            image_latents = self.vae.config.scaling_factor * image_latents
        return image_latents

    def prepare_latents_(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if self.model == "flux":
            height = 2 * (int(height) // self.vae_scale_factor)
            width = 2 * (int(width) // self.vae_scale_factor)
            #! Check but think we dont need latents_image_ids cause we dotn change the dim of image
            # latent_image_ids = self._prepare_latent_image_ids(
            #     batch_size, height, width, device, dtype
            # )
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            if (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat(
                    [image_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] != 0
            ):
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, height, width
            )

            return image_latents

        elif self.model in ["sd3", "sd1.4"]:
            image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1
            )
            return image_latents

    @torch.no_grad()
    def encode_image_sd(
        self, image: Union[List[Image.Image], Image.Image], generator=None, dtype=None
    ):
        if isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = 1
        image = self.image_processor.preprocess(image)
        current_device = self.vae.device
        device = self._execution_device
        self.vae.to(device)
        if dtype is not None:
            image = image.type(dtype).to(device)
        else:
            image = image.to(device)

        latents = self.prepare_latents_(
            image=image,
            batch_size=batch_size,
            height=self.height,
            width=self.width,
            num_channels_latents=(
                self.vae.config.latent_channels
                if self.model != "flux"
                else self.transformer.config.in_channels // 4
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        self.vae.to(current_device)
        return latents

    def get_step_size(
        self,
        scale_range,
        scale_factor,
    ):
        scale_range = np.linspace(
            scale_range[0], scale_range[1], len(self.scheduler.timesteps)
        )
        step_size = scale_factor * np.sqrt(scale_range)
        return step_size

    def update_extra_parameters(
        self,
        extra_params_iteref,
        extra_params_gsng,
        extra_params_distrib,
        num_images_per_prompt,
        batch_size,
    ):
        if self.criterion_iteref is not None:
            extra_params_iteref = self.criterion_iteref.update_extra_parameters(
                num_images_per_prompt=num_images_per_prompt,
                batch_size=batch_size,
                **extra_params_iteref,
            )

        if self.criterion_gsng is not None:
            extra_params_gsng = self.criterion_gsng.update_extra_parameters(
                num_images_per_prompt=num_images_per_prompt,
                batch_size=batch_size,
                **extra_params_gsng,
            )
        if self.criterion_distrib is not None:
            extra_params_distrib = self.criterion_distrib.update_extra_parameters(
                num_images_per_prompt=num_images_per_prompt,
                batch_size=batch_size,
                **extra_params_distrib,
            )
        return extra_params_iteref, extra_params_gsng, extra_params_distrib

    def update_steps_gsn(self, steps):
        if isinstance(steps, int):
            steps = [steps]
        return steps

    def check_inputs_gsn(
        self, extra_params_iteref, extra_params_gsng, extra_params_distrib, batch_size
    ):
        if self.criterion_iteref is not None:
            self.criterion_iteref.check_inputs(
                batch_size=batch_size,
                **extra_params_iteref,
            )
        if self.criterion_gsng is not None:
            self.criterion_gsng.check_inputs(
                batch_size=batch_size,
                **extra_params_gsng,
            )
        if self.criterion_distrib is not None:
            self.criterion_distrib.check_inputs(
                batch_size=batch_size,
                **extra_params_distrib,
            )

    def update_iteref_thresholds(self, i):
        if self.criterion_iteref is not None and self.thresholds_iteref is not None:
            if i in self.thresholds_iteref:
                if isinstance(self.criterion_iteref, AttendAndExciteGSN) or isinstance(
                    self.criterion_iteref, BoxDiffGSN
                ):
                    self.criterion_iteref.threshold = self.thresholds_iteref[i]

    def display_loss(self, loss, loss_dict):
        info_string = ", ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
        info_string += f", total_loss: {loss:.4f}"
        logger.info(info_string)

    def is_masks_needed(self, criterion):
        if isinstance(criterion, RetentionLoss) and not criterion.masks_available:
            return True
        return False

    def iteref_fc(
        self,
        text_embeddings,
        latents,
        i,
        t,
        cross_attention_kwargs=None,
        pooled_embeddings=None,
        text_ids=None,
        latent_image_ids=None,
    ):
        """Run Iterative Refinement (Iteref) on latent samples for one timestep.

        Iteref performs multiple optimization updates on each latent at selected
        diffusion steps. For each latent, the method repeatedly:
        1. Runs a model forward pass to collect attention maps.
        2. Computes the criterion loss.
        3. Updates the latent using either a manual gradient step or an optimizer.
        4. Stops early when the criterion reports optimization success.

        This corresponds to the iterative variant of GSN guidance where updates are
        applied up to a maximum number of iterations or until a threshold is met.

        Args:
            text_embeddings: Text conditioning embeddings for each latent in the batch.
            latents: Latent tensor batch to refine at the current diffusion step.
            i: Current denoising step index (used for schedule/threshold lookup).
            t: Current diffusion timestep value.
            cross_attention_kwargs: Optional cross-attention kwargs passed to the model
                forward pass (mainly used by SD1.4 UNet path).
            pooled_embeddings: Optional pooled text embeddings (used by SD3/Flux).
                If None, a list of None is used.
            text_ids: Optional token ids used by Flux transformer.
            latent_image_ids: Optional latent-image ids used by Flux transformer.

        Returns:
            torch.Tensor: Refined latent tensor batch after Iteref optimization.

        Note:
            - If the criterion is `InitNOGSN`, attention caches are updated and
              propagated across iterations.
            - When `return_intermediate_features` is enabled, intermediate refined
              latents are stored in `self.intermediate_values`.
        """
        self.update_iteref_thresholds(i)
        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=True,
            gsn_guidance_step=False,
        )
        if pooled_embeddings is None:
            pooled_embeddings = [None] * len(text_embeddings)

        # todo: experimental features never tested and not finished
        rescale_iteref = False

        with torch.enable_grad():
            updated_latents = []
            if isinstance(self.criterion_iteref, InitNOGSN):
                attention_maps_cache_clip_stack = []
                attention_maps_cache_t5_stack = []
            else:
                attention_maps_cache_clip_stack = None
                attention_maps_cache_t5_stack = None

            with self.progress_bar(total=len(latents)) as latents_processing_bar:
                latents_processing_bar.set_description("ITEREF Processing")
                for (
                    batch_image_id,
                    latent,
                    text_embedding,
                    pooled_embedding,
                    extra_params,
                ) in zip(
                    range(len(latents)),
                    latents,
                    text_embeddings,
                    pooled_embeddings,
                    self.extra_params_iteref_updated,
                ):
                    # logger.info("-------------------")
                    should_break = False
                    torch.cuda.empty_cache()

                    if rescale_iteref:
                        std_rescale = latent.std()

                    latent = latent.clone().detach().requires_grad_(True)

                    if self.optimizer_class_iteref is None:
                        optimizer = None
                    elif self.optimizer_class_iteref == "Adam":
                        optimizer = torch.optim.Adam(
                            [latent],
                            lr=self.step_size_iteref[i],
                            weight_decay=0,
                            eps=1e-3,
                        )
                    else:
                        optimizer = torch.optim.SGD(
                            [latent],
                            lr=self.step_size_iteref[i],
                            weight_decay=0,
                            momentum=0.0,
                        )

                    latent = latent.unsqueeze(0)
                    text_embedding = text_embedding.unsqueeze(0)
                    if pooled_embedding is not None:
                        pooled_embedding = pooled_embedding.unsqueeze(0)

                    with self.progress_bar(
                        total=self.max_opti_iteref
                    ) as optimization_bar:
                        optimization_bar.set_description(
                            f"Optimization (Latent {batch_image_id})",
                        )
                        for r in range(self.max_opti_iteref):
                            if rescale_iteref and std_rescale < latent.std():
                                latent = latent * (std_rescale / latent.std())
                            self.process_latents(
                                latents=latent,
                                text_embedding=text_embedding,
                                pooled_embedding=pooled_embedding,
                                timestep=t,
                                cross_attention_kwargs=cross_attention_kwargs,
                                latent_image_ids=latent_image_ids,
                                text_ids=text_ids,
                            )
                            if optimizer is not None:
                                optimizer.zero_grad()
                            loss, optimization_success, loss_dict = (
                                self.criterion_iteref.compute_loss(
                                    attention_store=self.attention_store,
                                    attention_file_name=f"ITEREF_img_{batch_image_id}_timestep_{t}_iteration_{r}",
                                    **extra_params,
                                )
                            )
                            if isinstance(self.criterion_iteref, InitNOGSN):
                                (
                                    attention_maps_cache_clip,
                                    attention_maps_cache_t5,
                                ) = self.criterion_iteref.get_attention_cache()
                                # update for the next iteration
                                extra_params["attention_maps_cache_clip"] = (
                                    attention_maps_cache_clip
                                )
                                extra_params["attention_maps_cache_t5"] = (
                                    attention_maps_cache_t5
                                )

                            postfix_dict = {
                                f"{k}": f"{v:.4f}" for k, v in loss_dict.items()
                            }
                            postfix_dict["total_loss"] = f"{loss.item():.4f}"
                            optimization_bar.set_postfix(postfix_dict)
                            optimization_bar.update(1)
                            if optimization_success:
                                logger.info(f"break {r=}")
                                should_break = True
                            if should_break:
                                del loss
                                break
                            if loss != 0:
                                if optimizer is None:
                                    latent = update_latent(
                                        latents=latent,
                                        loss=loss,
                                        step_size=self.step_size_iteref[i],
                                    )
                                else:
                                    loss.requires_grad_(True)
                                    loss.backward(retain_graph=True)
                                    optimizer.step()
                            del loss
                            self.release_memory()

                        if rescale_iteref and std_rescale < latent.std():
                            latent = latent * (std_rescale / latent.std())
                    if isinstance(self.criterion_iteref, InitNOGSN):
                        # get the last attention maps cache
                        (
                            attention_maps_cache_clip,
                            attention_maps_cache_t5,
                        ) = self.criterion_iteref.get_attention_cache()
                        attention_maps_cache_clip_stack.append(
                            attention_maps_cache_clip
                        )
                        attention_maps_cache_t5_stack.append(attention_maps_cache_t5)
                    if rescale_iteref:
                        latent = latent * (std_rescale / latent.std())

                    updated_latents.append(latent)
                    latents_processing_bar.update(1)
        latents = torch.cat(updated_latents, dim=0)

        if self.return_intermediate_features:
            self.intermediate_values[f"latents_ITEREF_{int(t)}"] = (
                latents.clone().detach().cpu()
            )
        # get the mask if needed
        if self.is_masks_needed(self.criterion_distrib) and self.is_masks_needed(
            self.criterion_gsng
        ):
            masks = []
            for latent, text_embedding, pooled_embedding, extra_params in zip(
                latents,
                text_embeddings,
                pooled_embeddings,
                self.extra_params_iteref_updated,
            ):
                latent = latent.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0)
                if pooled_embedding is not None:
                    pooled_embedding = pooled_embedding.unsqueeze(0)

                self.process_latents(
                    latents=latent,
                    text_embedding=text_embedding,
                    pooled_embedding=pooled_embedding,
                    timestep=t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    latent_image_ids=latent_image_ids,
                    text_ids=text_ids,
                )

                masks.append(
                    RetentionLoss.get_masks(
                        self.attention_store,
                        **extra_params,
                    )
                )
                self.release_memory()

            if self.is_masks_needed(self.criterion_distrib):
                self.extra_params_distrib_default["masks"] = masks
                self.criterion_distrib.masks_available = True
            if self.is_masks_needed(self.criterion_gsng):
                if self.criterion_distrib is None:
                    # update direct the extra params, because extra params are only updated after a distrib step
                    for item, masks_ in zip(self.extra_params_gsng_updated, masks):
                        item["masks"] = masks_
                else:
                    self.extra_params_gsng_default["masks"] = masks
                self.criterion_gsng.masks_available = True
            self.save_masks_if_necessary(masks, t)

        self.update_cache_initNO(
            attention_maps_cache_clip_stack, attention_maps_cache_t5_stack
        )

        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=False,
            gsn_guidance_step=False,
        )
        return latents

    @torch.enable_grad()
    def gsng_fc(
        self,
        latents,
        text_embeddings,
        i,
        t,
        cross_attention_kwargs=None,
        pooled_embeddings=None,
        text_ids=None,
        latent_image_ids=None,
    ):
        """Apply one-step GSN guidance update to each latent at timestep ``t``.

        GSN guidance (GSNG) performs a single latent shift per selected diffusion
        step using the gradient of an attention-based loss:
        ``x_t <- x_t - alpha_t * grad_{x_t}(L)``.
        Compared with Iteref, this method executes one loss-driven update per latent
        for the current timestep instead of multiple inner optimization iterations.

        Args:
            latents: Latent tensor batch to update.
            text_embeddings: Text conditioning embeddings for each latent.
            i: Current denoising step index (used to select guidance step size).
            t: Current diffusion timestep value.
            cross_attention_kwargs: Optional cross-attention kwargs for model forward.
            pooled_embeddings: Optional pooled text embeddings (used by SD3/Flux).
                If None, a list of None is used.
            text_ids: Optional token ids used by Flux transformer.
            latent_image_ids: Optional latent-image ids used by Flux transformer.

        Returns:
            torch.Tensor: Latent tensor batch after one GSNG update.

        Note:
            - If the criterion is `InitNOGSN`, attention caches are collected and
              pushed back into criterion extra parameters.
            - Intermediate outputs are stored when
              `self.return_intermediate_features` is True.
        """
        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=False,
            gsn_guidance_step=True,
        )
        if pooled_embeddings is None:
            pooled_embeddings = [None] * len(text_embeddings)
        latents = latents.clone().detach().requires_grad_(True)
        updated_latents = []

        if isinstance(self.criterion_gsng, InitNOGSN):
            attention_maps_cache_clip_stack = []
            attention_maps_cache_t5_stack = []
        else:
            attention_maps_cache_clip_stack = None
            attention_maps_cache_t5_stack = None

        with self.progress_bar(total=len(latents)) as latents_processing_bar:
            latents_processing_bar.set_description("GSN Processing")
            for (
                batch_image_id,
                latent,
                text_embedding,
                pooled_embedding,
                extra_params,
            ) in zip(
                range(len(latents)),
                latents,
                text_embeddings,
                pooled_embeddings,
                self.extra_params_gsng_updated,
            ):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                torch.cuda.empty_cache()
                if pooled_embedding is not None:
                    pooled_embedding = pooled_embedding.unsqueeze(0)

                text_embedding = text_embedding.unsqueeze(0)

                self.process_latents(
                    latents=latent,
                    text_embedding=text_embedding,
                    pooled_embedding=pooled_embedding,
                    timestep=t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    latent_image_ids=latent_image_ids,
                    text_ids=text_ids,
                )

                loss, _, loss_dict = self.criterion_gsng.compute_loss(
                    attention_store=self.attention_store,
                    attention_file_name=f"GSNG_img_{batch_image_id}_timestep_{t}",
                    **extra_params,
                )
                self.display_loss(loss, loss_dict)
                if loss != 0:
                    latent = update_latent(
                        latents=latent,
                        loss=loss,
                        step_size=self.step_size_gsng[i],
                    )
                if isinstance(self.criterion_gsng, InitNOGSN):
                    # update the cached cross attention maps
                    (
                        attention_maps_cache_clip,
                        attention_maps_cache_t5,
                    ) = self.criterion_gsng.get_attention_cache()
                    attention_maps_cache_clip_stack.append(attention_maps_cache_clip)
                    attention_maps_cache_t5_stack.append(attention_maps_cache_t5)

                del loss
                self.release_memory()
                updated_latents.append(latent)
                latents_processing_bar.update(1)

        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=False,
            gsn_guidance_step=False,
        )
        self.update_cache_initNO(
            attention_maps_cache_clip_stack, attention_maps_cache_t5_stack
        )
        updated_latents = torch.cat(updated_latents, dim=0)
        if self.return_intermediate_features:
            self.intermediate_values[f"latents_GSNG_{int(t)}"] = (
                updated_latents.clone().detach().cpu()
            )
        return updated_latents

    def update_cache_initNO(
        self, attention_maps_cache_clip_stack, attention_maps_cache_t5_stack
    ):
        if isinstance(self.criterion_gsng, InitNOGSN):
            for item, cache_clip, cache_t5 in zip(
                self.extra_params_gsng_updated,
                attention_maps_cache_clip_stack,
                attention_maps_cache_t5_stack,
            ):
                item["attention_maps_cache_clip"] = cache_clip
                item["attention_maps_cache_t5"] = cache_t5
        if isinstance(self.criterion_iteref, InitNOGSN):
            for item, cache_clip, cache_t5 in zip(
                self.extra_params_iteref_updated,
                attention_maps_cache_clip_stack,
                attention_maps_cache_t5_stack,
            ):
                item["attention_maps_cache_clip"] = cache_clip
                item["attention_maps_cache_t5"] = cache_t5

    def get_initial_mu(self, latents, alphas, t):
        if self.model in ["flux", "sd3"]:
            return torch.zeros_like(latents)
        else:
            if self.model in ["flux"]:
                latents = self._unpack_latents(
                    latents, self.height, self.width, self.vae_scale_factor
                )
            latents = (latents - (1 - alphas[t]) ** 0.5 * latents) / (alphas[t] ** 0.5)
            if self.model in ["flux"]:
                batch_size, channels, height, width = latents.shape
                return self._pack_latents(
                    latents,
                    batch_size,
                    channels,
                    height,
                    width,
                )
            return latents

    def get_mu(self, mu, mode="x0", block_size=(8, 8)):
        # x0, x0_mean, null, random, bayer
        resolution = (self.height, self.width)  # Specify the resolution of the matrix
        # Specify the block size for each color (4 pixels per color block)
        if mode == "x0":
            return mu
        elif mode == "x0_mean":
            if self.model in ["flux"]:
                mu = self._unpack_latents(
                    mu, self.height, self.width, self.vae_scale_factor
                )
            mu = torch.zeros_like(mu) + mu.mean(dim=(-1, -2), keepdim=True)
            if self.model in ["flux"]:
                batch_size, channels, height, width = mu.shape
                return self._pack_latents(
                    mu,
                    batch_size,
                    channels,
                    height,
                    width,
                )
            return torch.zeros_like(mu) + mu.mean(dim=(-1, -2), keepdim=True)
        elif mode == "null" or mode is None:
            return torch.zeros_like(mu)
        elif mode == "random":
            colored_img = generate_random_colored_image(resolution, block_size)
            return self.encode_image_sd([colored_img], dtype=mu.dtype).repeat(
                mu.shape[0], 1, 1, 1
            )
        elif mode == "shuffle":
            if self.model in ["flux"]:
                mu = self._unpack_latents(
                    mu, self.height, self.width, self.vae_scale_factor
                )
            batch_size, channels, height, width = mu.shape
            flat_latents = mu.view(batch_size, channels, -1)
            perm = torch.randperm(height * width, device=mu.device)
            mu_shuffled = flat_latents[:, :, perm].view(
                batch_size, channels, height, width
            )
            if self.model in ["flux"]:
                return self._pack_latents(
                    mu_shuffled,
                    batch_size,
                    channels,
                    height,
                    width,
                )
            return mu_shuffled

        elif mode == "bayer":
            bayer_matrix = numpy_to_image_bayer_matrix(
                generate_bayer_matrix(resolution, block_size)
            )
            return self.encode_image_sd([bayer_matrix], dtype=mu.dtype).repeat(
                mu.shape[0], 1, 1, 1
            )

        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    def masks_to_cpu(self, masks):
        return [[m.clone().detach().cpu() for m in mask] for mask in masks]

    def save_masks_if_necessary(self, masks, t):
        if self.return_intermediate_features:
            self.intermediate_values[f"masks_{int(t)}"] = self.masks_to_cpu(masks)

    def rescale_mu(self, mu_learned, mu_init):
        if self.model in ["flux"]:
            mu_learned = self._unpack_latents(
                mu_learned, self.height, self.width, self.vae_scale_factor
            )
            mu_init = self._unpack_latents(
                mu_init, self.height, self.width, self.vae_scale_factor
            )

        mu_init_std = mu_init.std()
        mu_learned_std = mu_learned.std()
        if self.init_mu == "null" or self.init_mu is None:
            mu_learned_std = mu_learned_std + 1e-8
        scale_factor = mu_init_std / mu_learned_std

        if scale_factor < 1:
            mu_learned = mu_learned * scale_factor

        if self.model in ["flux"]:
            batch_size, channels, height, width = mu_learned.shape
            return self._pack_latents(
                mu_learned,
                batch_size,
                channels,
                height,
                width,
            )
        return mu_learned

    def process_latents(
        self,
        latents,
        timestep,
        text_embedding,
        pooled_embedding,
        cross_attention_kwargs,
        text_ids,
        latent_image_ids,
    ):
        if self.model == "sd3":
            self.transformer(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0]),
                encoder_hidden_states=text_embedding,
                pooled_projections=pooled_embedding,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            self.transformer.zero_grad()
        elif self.model == "sd1.4":
            self.unet(
                latents,
                timestep,
                encoder_hidden_states=text_embedding,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            self.unet.zero_grad()
        elif self.model == "flux":
            if self.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [1],
                    self._guidance_scale,
                    device=latents.device,
                    dtype=torch.float32,
                )
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            self.transformer(
                hidden_states=latents,
                timestep=timestep.expand(latents.shape[0]) / 1000,
                guidance=guidance,
                pooled_projections=pooled_embedding,
                encoder_hidden_states=text_embedding,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            self.transformer.zero_grad()

    def saga_fct(
        self,
        latents,
        noises_sampling,
        text_embeddings,
        i,
        t,
        noise_generators,
        num_images_per_prompt,
        cross_attention_kwargs=None,
        pooled_embeddings=None,
        mu=None,
        text_ids=None,
        latent_image_ids=None,
    ):
        """Run SAGA distribution optimization and resample guided latents.

        SAGA learns a signal-aligned latent distribution at a chosen denoising step,
        then samples updated latents from the learned distribution. For each input
        latent, the method optimizes distribution parameters (mean ``mu`` and
        optionally variance ``var``) by minimizing an attention-based criterion over
        multiple stochastic samples.

        The method supports two usage modes:
        - One distribution per prompt (sample multiple latents from it).
        - One distribution per image (`one_image_per_distrib=True`).

        Args:
            latents: Input latent batch used to estimate/initialize distribution
                parameters.
            noises_sampling: Noise tensor(s) used for final latent reconstruction from
                the optimized distribution.
            text_embeddings: Text conditioning embeddings.
            i: Current denoising step index.
            t: Current diffusion timestep value.
            noise_generators: Random generators used during optimization sampling.
            num_images_per_prompt: Number of images generated per prompt.
            cross_attention_kwargs: Optional cross-attention kwargs for model forward.
            pooled_embeddings: Optional pooled text embeddings (used by SD3/Flux).
                If None, a list of None is used.
            mu: Optional initial mean tensor. If None, a default is computed from
                the current model/timestep.
            text_ids: Optional token ids used by Flux transformer.
            latent_image_ids: Optional latent-image ids used by Flux transformer.

        Returns:
            torch.Tensor: Reconstructed latent batch sampled from the optimized
            distribution.

        Note:
            - If needed by downstream criteria, masks are recomputed from attention
              maps and stored for later guidance steps.
            - When `return_intermediate_features` is enabled, optimized ``mu`` and
              ``var`` snapshots are stored in `self.intermediate_values`.
        """

        if pooled_embeddings is None:
            pooled_embeddings = [None] * len(text_embeddings)
        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=False,
            gsn_guidance_step=False,
            create_signal=True,
            batch_size=self.batch_size_noise,
        )

        alphas_timesteps = (
            self.scheduler.sigmas
            if self.model in ["flux", "sd3"]
            else self.scheduler.alphas_cumprod
        )

        var = (
            self.get_var(
                latents=latents,
                alphas=alphas_timesteps,
                t=t,
                per_channel=self.per_channel,
                log_var=self.log_var,
                block=self.block,
            )
            .to(latents.device)
            .type(latents.dtype)
        )

        if mu is None:
            mu = self.get_initial_mu(latents, alphas_timesteps, t)

        if self.return_intermediate_features:
            for idx, var_ in enumerate(var):
                self.intermediate_values[f"var_{t}_{idx}"] = self.get_return_var(
                    var=var_,
                    latents=mu,
                )

        if self.mu_reference_rescale is not None:
            mu_for_loss = self.scale_x0(
                alphas=alphas_timesteps, t=t, x0=self.mu_reference_rescale
            )
        else:
            mu_for_loss = self.scale_x0(alphas=alphas_timesteps, t=t, x0=mu)

        # else:
        # mu_collected_for_loss = [None] * mu.shape[0]

        mu = self.get_mu(
            mode=self.init_mu,
            mu=mu,
        )
        if self.return_intermediate_features:
            self.intermediate_values[f"mu_init_{int(t)}"] = mu.clone().detach().cpu()

        mu = self.scale_x0(alphas=alphas_timesteps, t=t, x0=mu)

        if self.model in ["flux"]:
            _, channels, height, width = self.get_shape_latents_flux(mu)
        else:
            _, channels, height, width = mu.shape
        shape_noise = (
            self.batch_size_noise,
            channels,
            height,
            width,
        )

        with torch.enable_grad():
            updated_mu = []
            updated_var = []

            with self.progress_bar(total=len(latents)) as latents_processing_bar:
                latents_processing_bar.set_description("SAGA Processing")
                for (
                    batch_image_id,
                    mu_,
                    var_,
                    mu_loss,
                    text_embedding,
                    pooled_embedding,
                    gen,
                    extra_params,
                ) in zip(
                    range(len(mu)),
                    mu,
                    var,
                    mu_for_loss,
                    text_embeddings,
                    pooled_embeddings,
                    noise_generators,
                    self.extra_params_distrib_updated,
                ):
                    # logger.info("-------------------")
                    should_break = False

                    torch.cuda.empty_cache()
                    mu_ = mu_.clone().detach().requires_grad_(True)
                    params = [mu_]
                    if self.block > 0:
                        var_ = var_.clone().detach().requires_grad_(True)
                        params.append(var_)

                    if self.optimizer_class == "Adam":
                        opti = torch.optim.Adam(
                            params,
                            lr=self.step_size_distrib,
                            weight_decay=0,
                            eps=1e-3,
                        )
                    else:
                        opti = torch.optim.SGD(
                            params,
                            lr=self.step_size_distrib,
                            weight_decay=0,
                            momentum=self.momentum_saga,
                        )

                    mu_ = mu_.unsqueeze(0)
                    mu_loss = mu_loss.unsqueeze(0)
                    text_embedding = text_embedding.unsqueeze(0).repeat(
                        self.batch_size_noise, 1, 1
                    )

                    if pooled_embedding is not None:
                        pooled_embedding = pooled_embedding.unsqueeze(0).repeat(
                            self.batch_size_noise, 1
                        )

                    with self.progress_bar(
                        total=self.max_opti_distrib
                    ) as optimization_bar:
                        optimization_bar.set_description(
                            f"Optimization (Latent {batch_image_id})",
                        )
                        for r in range(self.max_opti_distrib):
                            torch.cuda.empty_cache()

                            x = self.get_sample(
                                mu=self.rescale_mu(
                                    mu_learned=(mu_),
                                    mu_init=mu_loss,
                                ),
                                var=var_,
                                noise=randn_tensor(
                                    device=mu.device,
                                    dtype=mu_.dtype,
                                    shape=shape_noise,
                                    generator=gen,
                                ),
                            )

                            self.process_latents(
                                latents=x,
                                text_embedding=text_embedding,
                                pooled_embedding=pooled_embedding,
                                timestep=t,
                                cross_attention_kwargs=cross_attention_kwargs,
                                latent_image_ids=latent_image_ids,
                                text_ids=text_ids,
                            )
                            if opti is not None:
                                opti.zero_grad()
                            loss, optimization_success, loss_dict = (
                                self.criterion_distrib.compute_loss(
                                    attention_store=self.attention_store,
                                    attention_file_name=[
                                        f"SAGA_img_{batch_image_id}_timestep_{t}_iteration_{r}_batch_{b}"
                                        for b in range(self.batch_size_noise)
                                    ],
                                    **extra_params,
                                )
                            )

                            if optimization_success:
                                logger.info(f"break {r=}")
                                should_break = True
                            if should_break:
                                del loss
                                break

                            if loss != 0:
                                loss.requires_grad_(True)
                                loss.backward(retain_graph=True)
                                opti.step()

                            postfix_dict = {
                                f"{k}": f"{v:.4f}" for k, v in loss_dict.items()
                            }
                            postfix_dict["total_loss"] = f"{loss.item():.4f}"
                            optimization_bar.set_postfix(postfix_dict)
                            optimization_bar.update(1)
                            del loss
                            self.release_memory()

                    if self.rescale:
                        # compute std on mu_loss, and rescale mu_ to have the same std
                        mu_ = self.rescale_mu(
                            mu_init=mu_loss,
                            mu_learned=(mu_),
                        )

                    updated_mu.append(mu_)
                    updated_var.append(var_)
                    latents_processing_bar.update(1)

        mu = torch.cat(updated_mu, dim=0)
        var = torch.stack(updated_var, dim=0)

        if self.is_masks_needed(self.criterion_gsng) or self.is_masks_needed(
            self.criterion_iteref
        ):
            masks = []
            # if you use a criterion that do not use cross attention maps in the attention store
            # you should change update the framework to add a change of the attention store
            for (
                mu_,
                var_,
                text_embedding,
                pooled_embedding,
                gen,
                extra_params,
            ) in zip(
                mu,
                var,
                text_embeddings,
                pooled_embeddings,
                noise_generators,
                self.extra_params_distrib_updated,
            ):
                mu_ = mu_.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0).repeat(
                    self.batch_size_noise, 1, 1
                )
                if pooled_embedding is not None:
                    pooled_embedding = pooled_embedding.unsqueeze(0).repeat(
                        self.batch_size_noise, 1
                    )
                x = self.get_sample(
                    mu=(mu_),
                    var=var_,
                    noise=randn_tensor(
                        device=mu.device,
                        dtype=mu_.dtype,
                        shape=shape_noise,
                        generator=gen,
                    ),
                )

                self.process_latents(
                    latents=x,
                    text_embedding=text_embedding,
                    pooled_embedding=pooled_embedding,
                    timestep=t,
                    cross_attention_kwargs=cross_attention_kwargs,
                    latent_image_ids=latent_image_ids,
                    text_ids=text_ids,
                )
                masks.append(
                    RetentionLoss.get_masks(
                        self.attention_store,
                        **extra_params,
                    )
                )
                self.release_memory()

            if self.is_masks_needed(self.criterion_gsng):
                self.extra_params_gsng_default["masks"] = masks
                self.criterion_gsng.masks_available = True
            if self.is_masks_needed(self.criterion_iteref):
                self.extra_params_iteref_default["masks"] = masks
                self.criterion_iteref.masks_available = True
            self.save_masks_if_necessary(masks, t)

        if self.return_intermediate_features:
            mu_to_return = mu.clone().detach().cpu()
            is_sd3_first_step = self.model in ["flux", "sd3"] and i == 0

            if not is_sd3_first_step:
                mu_to_return = self.unscale_x0(
                    x0=mu_to_return, alphas=alphas_timesteps, t=t
                )

            self.intermediate_values[f"mu_optimized_{int(t)}"] = mu_to_return
            for idx, var_ in enumerate(var):
                self.intermediate_values[f"var_optimized_{t}_{idx}"] = (
                    self.get_return_var(
                        var=var_,
                        latents=mu,
                    )
                )
        latents = []
        # reconstruct latent
        # split the noise sampling in group of size num_images_per_prompt
        if self.model in ["flux"]:
            noises_sampling = self._unpack_latents(
                noises_sampling, self.height, self.width, self.vae_scale_factor
            )

        if self.one_image_per_distrib:
            noises_sampling = noises_sampling.split(1, dim=0)
        else:
            noises_sampling = noises_sampling.split(num_images_per_prompt, dim=0)

        for mu_, var_, noises_sampling_ in zip(mu, var, noises_sampling):
            mu_ = mu_.unsqueeze(0)
            latents.append(
                self.get_sample(
                    mu=(mu_),
                    var=var_,
                    noise=noises_sampling_,
                )
            )

        self.update_processor(
            height=self.height,
            width=self.width,
            iteref_step=False,
            gsn_guidance_step=False,
            create_signal=False,
            batch_size=1,
        )
        return torch.cat(latents, dim=0)

    def can_do_something(self):
        if (
            self.criterion_iteref is not None
            or self.criterion_gsng is not None
            or self.criterion_distrib is not None
        ):
            return True
        else:
            return False

    def inference_loop(
        self,
        i,
        t,
        latents,
        latents_distrib,
        text_embeddings,
        text_embeddings_distrib,
        mu,
        batch_size,
        num_images_per_prompt,
        generator_distrib,
        pooled_embeddings=None,
        pooled_embeddings_distrib=None,
        cross_attention_kwargs=None,
        text_ids=None,
        latent_image_ids=None,
        text_ids_distrib=None,
        latent_image_ids_distrib=None,
    ):
        if (
            self.criterion_distrib is not None
            and i == self.collect_mu_reference_rescale
            and mu is not None
        ):
            self.mu_reference_rescale = mu
        if self.criterion_distrib is not None and i == self.step_distrib:
            latents = self.saga_fct(
                latents=latents_distrib,
                noises_sampling=latents,
                text_embeddings=text_embeddings_distrib,
                i=i,
                t=t,
                noise_generators=generator_distrib,
                pooled_embeddings=pooled_embeddings_distrib,
                mu=mu,
                num_images_per_prompt=num_images_per_prompt,
                cross_attention_kwargs=cross_attention_kwargs,
                text_ids=text_ids_distrib,
                latent_image_ids=latent_image_ids_distrib,
            )
            # to know that the distrib is done, can be do only one time
            self.criterion_distrib = None
            (
                self.extra_params_iteref_updated,
                self.extra_params_gsng_updated,
                _,
            ) = self.update_extra_parameters(
                extra_params_iteref=self.extra_params_iteref_default,
                extra_params_gsng=self.extra_params_gsng_default,
                extra_params_distrib=None,
                num_images_per_prompt=num_images_per_prompt,
                batch_size=batch_size,
            )

            latents_distrib = None  # don't need it anymore

        ######## ITEREF ########
        if (self.criterion_iteref is not None) and (i in self.steps_iteref):
            output = self.iteref_fc(
                latents=(
                    latents if self.criterion_distrib is None else latents_distrib
                ),
                text_embeddings=(
                    text_embeddings
                    if self.criterion_distrib is None
                    else text_embeddings_distrib
                ),
                cross_attention_kwargs=cross_attention_kwargs,
                i=i,
                t=t,
                pooled_embeddings=(
                    pooled_embeddings
                    if self.criterion_distrib is None
                    else pooled_embeddings_distrib
                ),
                text_ids=(
                    text_ids if self.criterion_distrib is None else text_ids_distrib
                ),
                latent_image_ids=(
                    latent_image_ids
                    if self.criterion_distrib is None
                    else latent_image_ids_distrib
                ),
            )
            if self.criterion_distrib is None:
                latents = output
            else:
                latents_distrib = output

        #### GSN GUIDANCE ####
        # can be combined with iteref
        # It is often seen in other implementation one step of GSN guidance after the iteref
        if (self.criterion_gsng is not None) and (i in self.steps_gsng):
            output = self.gsng_fc(
                latents=(
                    latents if self.criterion_distrib is None else latents_distrib
                ),
                text_embeddings=(
                    text_embeddings
                    if self.criterion_distrib is None
                    else text_embeddings_distrib
                ),
                i=i,
                t=t,
                cross_attention_kwargs=cross_attention_kwargs,
                pooled_embeddings=(
                    pooled_embeddings
                    if self.criterion_distrib is None
                    else pooled_embeddings_distrib
                ),
                text_ids=(
                    text_ids if self.criterion_distrib is None else text_ids_distrib
                ),
                latent_image_ids=(
                    latent_image_ids
                    if self.criterion_distrib is None
                    else latent_image_ids_distrib
                ),
            )
            if self.criterion_distrib is None:
                latents = output
            else:
                latents_distrib = output

        return latents, latents_distrib

    def end_of_pipeline(self):
        self.intermediate_values = {}
        for criterion in [
            self.criterion_iteref,
            self.criterion_gsng,
            self.criterion_distrib,
        ]:
            if isinstance(criterion, RetentionLoss):
                # desactivate the mask in case the object is reused
                criterion.masks_available = False
        self.maybe_free_model_hooks()
        self.set_criterions()
        self.wait_for_pending_saves()

    # todo: add the initNO boosting function
    # def initno_boosting()
