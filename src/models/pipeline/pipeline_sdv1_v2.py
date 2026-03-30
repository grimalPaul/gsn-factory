# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import (
    replace_example_docstring,
)

from ...utils import RankedLogger
from ..gsn_config import DistribConfig, GsngConfig, IterefConfig
from ..gsn_criterion.utils import AbstractGSN
from ..gsn_tools import ToolsClassMixin
from ..utils import get_indices_from_tokens

log = RankedLogger(__name__, rank_zero_only=True)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import
        ```
"""


class StableDiffusionGSN(StableDiffusionPipeline, ToolsClassMixin):
    def get_indice_last_token(self, prompt: str) -> int:
        return len(self.tokenizer(prompt)["input_ids"]) - 1

    def get_attention_idx_from_token(
        self,
        prompt: str,
        word_list: List[str],
    ) -> List[List[int]]:
        token_indices = self.get_indices(prompt)
        token_indices = {
            idx: word.replace("▁", "").replace("</w>", "")
            for idx, word in token_indices.items()
        }
        idx = get_indices_from_tokens(
            token_by_word=token_indices,
            word_list=word_list,
        )
        return idx

    def get_indices(self, prompt: str):
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                self.tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    def register_attention_control(self, processor, processor_blocks=None):
        attn_procs = {}
        cross_att_count = 0

        for name in self.unet.attn_processors.keys():
            inside = processor_blocks is not None and any(
                n in name for n in processor_blocks
            )
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if inside:
                attn_procs[name] = processor(
                    attnstore=self.attention_store, place_in_unet=place_in_unet
                )
                cross_att_count += 1
            else:
                attn_procs[name] = AttnProcessor2_0()

        self.unet.set_attn_processor(attn_procs)
        if self.attention_store is not None:
            self.attention_store.num_att_layers = cross_att_count

    def update_processor(
        self,
        height,
        width,
        iteref_step=False,
        gsn_guidance_step=False,
        create_signal=False,
        batch_size=1,
    ):
        res_height = int(np.ceil(height / 32))
        res_width = int(np.ceil(width / 32))

        def setup_attention(criterion: AbstractGSN):
            attention_store_cls = criterion.get_attention_store()
            params_attn_store, processor_blocks = criterion.get_params_attn_store(
                res_height=res_height, res_width=res_width, batch_size=batch_size
            )
            self.attention_store = attention_store_cls(
                executor=self.save_executor,
                store_attention_path=self.store_attention_path,
                **params_attn_store,
            )
            processor_cls = criterion.get_processor()
            self.register_attention_control(processor_cls, processor_blocks)

        if iteref_step and self.criterion_iteref is not None:
            setup_attention(self.criterion_iteref)

        elif gsn_guidance_step and self.criterion_gsng is not None:
            setup_attention(self.criterion_gsng)
        elif create_signal and self.criterion_distrib is not None:
            setup_attention(self.criterion_distrib)
        else:
            self.register_attention_control(AttnProcessor2_0)
            if self.attention_store is not None:
                self.attention_store.reset()
                self.attention_store = None

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # GSN
        iteref_config: Optional[IterefConfig] = None,
        gsng_config: Optional[GsngConfig] = None,
        distrib_config: Optional[DistribConfig] = None,
        # Distrib
        prompt_embeds_distrib=None,
        negative_prompt_embeds_distrib=None,
        generator_distrib: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        # Return or save intermediate features
        return_intermediate_features=False,
        decode_all=False,
        store_attention_path: Optional[str] = None,
    ):
        r"""
        The call function t o the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            max_iter_to_alter (`int`, *optional*, defaults to `25`):
                Number of denoising steps to apply attend-and-excite. The `max_iter_to_alter` denoising steps are when
                attend-and-excite is applied. For example, if `max_iter_to_alter` is `25` and there are a total of `30`
                denoising steps, the first `25` denoising steps applies attend-and-excite and the last `5` will not.
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor to control the step size of each attend-and-excite update.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        log.info(f"{prompt}")
        self.setup_save_executor(store_attention_path=store_attention_path)
        self.set_criterions(
            iteref_config=iteref_config,
            gsng_config=gsng_config,
            distrib_config=distrib_config,
            model="sd1.4",
            height=height,
            width=width,
            return_intermediate_features=return_intermediate_features,
        )

        self.should_desactivate_distrib(num_inference_steps)
        if distrib_config:
            self.check_one_image_per_distrib(
                list_generators_distrib=generator_distrib,
                list_generator_latents=generator,
            )
        self.init_intermediate_values()

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if self.criterion_distrib is not None:
            prompt_embeds_distrib, negative_prompt_embeds_distrib = self.encode_prompt(
                prompt,
                device,
                (
                    1
                    if not distrib_config.one_image_per_distrib
                    else num_images_per_prompt
                ),
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds_distrib,
                negative_prompt_embeds=negative_prompt_embeds_distrib,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if return_intermediate_features:
            self.intermediate_values["latents_initial"] = latents.clone().detach().cpu()

        if self.criterion_distrib is not None:
            latents_distrib = self.prepare_latents(
                (
                    batch_size * num_images_per_prompt
                    if distrib_config.one_image_per_distrib
                    else batch_size
                ),
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator_distrib,
                latents=None,
            )
            if return_intermediate_features:
                self.intermediate_values["latents_distrib_init"] = (
                    latents_distrib.clone().detach().cpu()
                )

        else:
            latents_distrib = None

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        #### SETUP GSN ####

        self.setup_params(
            batch_size=batch_size, num_images_per_prompt=num_images_per_prompt
        )
        if self.criterion_gsng is not None or self.criterion_iteref is not None:
            text_embeddings = prompt_embeds

        if self.criterion_distrib is not None:
            text_embeddings_distrib = prompt_embeds_distrib
            mu = None
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if self.criterion_distrib is not None:
                prompt_embeds_distrib = torch.cat(
                    [negative_prompt_embeds_distrib, prompt_embeds_distrib]
                )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.can_do_something():
                    latents, latents_distrib = self.inference_loop(
                        i=i,
                        t=t,
                        latents=latents,
                        latents_distrib=latents_distrib,
                        text_embeddings=(
                            None
                            if (
                                self.criterion_gsng is None
                                and self.criterion_iteref is None
                            )
                            else text_embeddings
                        ),
                        text_embeddings_distrib=(
                            None
                            if self.criterion_distrib is None
                            else text_embeddings_distrib
                        ),
                        batch_size=batch_size,
                        cross_attention_kwargs=cross_attention_kwargs,
                        generator_distrib=generator_distrib,
                        mu=None if self.criterion_distrib is None else mu,
                        num_images_per_prompt=num_images_per_prompt,
                    )

                # expand the latents if we are doing classifier free guidance
                if self.criterion_distrib is not None:
                    latent_model_input = (
                        torch.cat([latents_distrib] * 2)
                        if do_classifier_free_guidance
                        else latents_distrib
                    )
                else:
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=(
                        prompt_embeds
                        if self.criterion_distrib is None
                        else prompt_embeds_distrib
                    ),
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                step_latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents_distrib if self.criterion_distrib else latents,
                    **extra_step_kwargs,
                )
                if self.criterion_distrib is not None:
                    mu = step_latents.pred_original_sample
                    latents_distrib = step_latents.prev_sample
                    if return_intermediate_features:
                        self.intermediate_values[f"mu_{t}"] = mu.clone().detach().cpu()
                else:
                    latents = step_latents.prev_sample
                    if return_intermediate_features:
                        self.intermediate_values[f"x0_hat_{t}"] = (
                            step_latents.pred_original_sample.clone().detach().cpu()
                        )
                        # detect if last one
                        if i == len(timesteps) - 1:
                            self.intermediate_values[f"latents_{0}"] = (
                                latents.clone().detach().cpu()
                            )
                        else:
                            self.intermediate_values[
                                f"latents_{int(self.scheduler.timesteps[i + 1])}"
                            ] = (latents.clone().detach().cpu())

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(
                            step_idx,
                            t,
                            (
                                latents_distrib
                                if self.criterion_distrib is not None
                                else latents_distrib
                            ),
                        )

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if not return_dict:
            self.end_of_pipeline()
            return (image, has_nsfw_concept)

        output = StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
        if not return_intermediate_features:
            self.end_of_pipeline()
            return output
        if decode_all and return_intermediate_features:
            dict_latents = {}
            for key, values in self.intermediate_values.items():
                if ("var" in key) or ("mask" in key):
                    dict_latents[key] = values
                else:
                    dict_latents[key] = self.image_processor.postprocess(
                        self.vae.decode(
                            values.to(self.device) / self.vae.config.scaling_factor,
                            return_dict=False,
                        )[0],
                        output_type=output_type,
                        do_denormalize=[True] * values.shape[0],
                    )
        elif return_intermediate_features:
            dict_latents = self.intermediate_values
        self.end_of_pipeline()
        return output, dict_latents
