from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.pipelines.flux.pipeline_flux import (
    EXAMPLE_DOC_STRING,
    FluxPipelineOutput,
    calculate_shift,
    is_torch_xla_available,
    replace_example_docstring,
    retrieve_timesteps,
)

from ..gsn_config import DistribConfig, GsngConfig, IterefConfig
from ..gsn_criterion.utils import AbstractGSN
from ..gsn_tools import ToolsClassMixin
from ..utils import get_indices_from_tokens

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class FluxPipelineGSN(FluxPipeline, ToolsClassMixin):
    def get_indice_last_token(self, prompt: str):
        last_t5_idx = len(self.tokenizer_2(prompt).input_ids) - 1
        return last_t5_idx

    def get_attention_idx_from_token(
        self,
        prompt: str,
        word_list: List[str],
    ) -> List[List[int]]:
        # Preprocess token_by_word to remove special characters

        token_indices = self.get_indices(prompt)
        t5_idx_token = {
            idx: word.replace("▁", "").replace("</w>", "")
            for idx, word in token_indices.items()
        }
        t5_idx = get_indices_from_tokens(t5_idx_token, word_list)
        return t5_idx

    def _get_t5_token_indices(
        self,
        prompt: str = None,
    ):
        ids = self.tokenizer_2(
            prompt,
        ).input_ids
        return {
            i: tok
            for tok, i in zip(
                self.tokenizer_2.convert_ids_to_tokens(ids), range(len(ids))
            )
        }

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alter"""
        t5_ids = self._get_t5_token_indices(prompt=prompt)
        return t5_ids

    def register_attention_control(
        self,
        processor,
        processor_blocks=None,
        attention_store=False,
    ):
        attn_procs = {}
        if attention_store:
            att_count = 0
            for name in self.transformer.attn_processors.keys():
                inside = False
                id_block = int(name.split(".")[1])
                if processor_blocks is not None:
                    if id_block in processor_blocks:
                        inside = True

                if inside:
                    # print(f"use {id_block}", end=" ")
                    attn_procs[name] = processor(
                        attnstore=self.attention_store,
                        block_id=id_block,
                    )
                    att_count += 1
                else:
                    # print(f"do not use {id_block}", end=" ")
                    attn_procs[name] = self.transformer.attn_processors[name]

            self.attention_store.num_att_layers = att_count
        else:
            for name in self.transformer.attn_processors.keys():
                attn_procs[name] = FluxAttnProcessor2_0()
        self.transformer.set_attn_processor(attn_procs)

    def update_processor(
        self,
        height,
        width,
        iteref_step=False,
        gsn_guidance_step=False,
        create_signal=False,
        batch_size=1,
    ):
        res_height = (int(height) // self.vae_scale_factor) // 2  # patchification
        res_width = (int(width) // self.vae_scale_factor) // 2

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
            self.register_attention_control(
                processor_cls,
                processor_blocks,
                attention_store=True,  # todo: maybe remove ? see how to save attention maps
            )

        if iteref_step and self.criterion_iteref is not None:
            setup_attention(self.criterion_iteref)
        elif gsn_guidance_step and self.criterion_gsng is not None:
            setup_attention(self.criterion_gsng)
        elif create_signal and self.criterion_distrib is not None:
            setup_attention(self.criterion_distrib)
        else:
            self.register_attention_control(FluxAttnProcessor2_0, attention_store=False)
            if self.attention_store is not None:
                self.attention_store.reset()
                self.attention_store = None

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # GSN
        iteref_config: Optional[IterefConfig] = None,
        gsng_config: Optional[GsngConfig] = None,
        distrib_config: Optional[DistribConfig] = None,
        # Distrib
        prompt_embeds_distrib=None,
        pooled_prompt_embeds_distrib=None,
        generator_distrib: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        # Return or save intermediate features
        return_intermediate_features=False,
        decode_all=False,
        store_attention_path: Optional[str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.setup_save_executor(store_attention_path=store_attention_path)
        self.set_criterions(
            iteref_config=iteref_config,
            gsng_config=gsng_config,
            distrib_config=distrib_config,
            model="flux",
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
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.criterion_distrib is not None:
            (prompt_embeds_distrib, pooled_prompt_embeds_distrib, text_ids_distrib) = (
                self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_embeds=prompt_embeds_distrib,
                    pooled_prompt_embeds=pooled_prompt_embeds_distrib,
                    device=device,
                    num_images_per_prompt=(
                        1
                        if not distrib_config.one_image_per_distrib
                        else num_images_per_prompt
                    ),
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
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
            latents_distrib, latent_image_ids_distrib = self.prepare_latents(
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

        #### SETUP GSN ####
        self.setup_params(
            batch_size=batch_size, num_images_per_prompt=num_images_per_prompt
        )
        if self.criterion_gsng is not None or self.criterion_iteref is not None:
            text_embeddings = prompt_embeds

        if self.criterion_distrib is not None:
            prompt_embeds_distrib = prompt_embeds_distrib
            pooled_embeddings_distrib = pooled_prompt_embeds_distrib
            mu_distrib = None

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.can_do_something():
                    latents, latents_distrib = self.inference_loop(
                        i=i,
                        t=t,
                        latents=latents,
                        latents_distrib=latents_distrib,
                        text_embeddings=(
                            None
                            if self.criterion_gsng is None
                            and self.criterion_iteref is None
                            else prompt_embeds
                        ),
                        text_embeddings_distrib=(
                            None
                            if self.criterion_distrib is None
                            else prompt_embeds_distrib
                        ),
                        batch_size=batch_size,
                        cross_attention_kwargs=None,
                        generator_distrib=generator_distrib,
                        mu=None if self.criterion_distrib is None else mu_distrib,
                        num_images_per_prompt=num_images_per_prompt,
                        pooled_embeddings=pooled_prompt_embeds,
                        pooled_embeddings_distrib=(
                            None
                            if self.criterion_distrib is None
                            else pooled_embeddings_distrib
                        ),
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        text_ids_distrib=(
                            None if self.criterion_distrib is None else text_ids_distrib
                        ),
                        latent_image_ids_distrib=(
                            None
                            if self.criterion_distrib is None
                            else latent_image_ids_distrib
                        ),
                    )
                if self.criterion_distrib is not None:
                    latents_input = latents_distrib
                else:
                    latents_input = latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=(
                        pooled_prompt_embeds
                        if self.criterion_distrib is None
                        else pooled_embeddings_distrib
                    ),
                    encoder_hidden_states=(
                        prompt_embeds
                        if self.criterion_distrib is None
                        else prompt_embeds_distrib
                    ),
                    txt_ids=(
                        text_ids if self.criterion_distrib is None else text_ids_distrib
                    ),
                    img_ids=(
                        latent_image_ids
                        if self.criterion_distrib is None
                        else latent_image_ids_distrib
                    ),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                if self.criterion_distrib is not None:
                    if self.scheduler.step_index is None:
                        self.scheduler._init_step_index(t)
                    sample = latents_distrib.to(torch.float32)
                    sigma = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_next = self.scheduler.sigmas[-1]
                    prev_sample = sample + (sigma_next - sigma) * noise_pred
                    # Cast sample back to model compatible dtype
                    mu_distrib = prev_sample.to(noise_pred.dtype)
                    latents_distrib = self.scheduler.step(
                        noise_pred, t, latents_distrib, return_dict=False
                    )[0]
                    if return_intermediate_features:
                        self.intermediate_values[f"mu_{int(t)}"] = (
                            mu_distrib.clone().detach().cpu()
                        )
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]
                    if return_intermediate_features:
                        if i == len(timesteps) - 1:
                            self.intermediate_values[f"latents_{0}"] = (
                                latents.clone().detach().cpu()
                            )
                        else:
                            self.intermediate_values[
                                f"latents_{int(self.scheduler.timesteps[i + 1])}"
                            ] = (latents.clone().detach().cpu())

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            self.end_of_pipeline()
            return (image,)

        output = FluxPipelineOutput(images=image)
        if not return_intermediate_features:
            self.end_of_pipeline()
            return output

        if decode_all and return_intermediate_features:
            dict_latents = {}
            for key, values in self.intermediate_values.items():
                if "var" in key or "mask" in key:
                    dict_latents[key] = values
                else:
                    values = self._unpack_latents(
                        values, height, width, self.vae_scale_factor
                    )
                    values = (
                        values / self.vae.config.scaling_factor
                    ) + self.vae.config.shift_factor
                    image = self.vae.decode(
                        values.to(self._execution_device), return_dict=False
                    )[0]
                    image = self.image_processor.postprocess(
                        image, output_type=output_type
                    )
                    dict_latents[key] = image
        elif return_intermediate_features:
            dict_latents = self.intermediate_values
        self.end_of_pipeline()
        return output, dict_latents
