from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import (
    JointAttnProcessor2_0,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils import is_torch_xla_available, replace_example_docstring

from ...utils import RankedLogger
from ..gsn_config import DistribConfig, GsngConfig, IterefConfig
from ..gsn_criterion.utils import AbstractGSN
from ..gsn_tools import ToolsClassMixin
from ..utils import get_indices_from_tokens

logger = RankedLogger(__name__, rank_zero_only=True)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


class StableDiffusion3PipelineGSN(StableDiffusion3Pipeline, ToolsClassMixin):
    def get_indice_last_token(self, prompt: str, clip_model_index: int = 0) -> Tuple:
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        tokenizer = clip_tokenizers[clip_model_index]
        last_clip_idx = len(tokenizer(prompt).input_ids) - 1
        if self.text_encoder_3 is None:
            return None
        else:
            last_t5_idx = len(self.tokenizer_3(prompt).input_ids) - 1
        return last_clip_idx, last_t5_idx

    def get_attention_idx_from_token(
        self,
        prompt: str,
        word_list: List[str],
    ) -> List[List[int]]:
        # Preprocess token_by_word to remove special characters

        token_indices = self.get_indices(prompt)
        clip_idx_token = {
            idx: word.replace("▁", "").replace("</w>", "")
            for idx, word in token_indices[0].items()
        }

        clip_idx = get_indices_from_tokens(clip_idx_token, word_list)
        t5_idx_token = (
            None
            if token_indices[1] is None
            else {
                idx: word.replace("▁", "").replace("</w>", "")
                for idx, word in token_indices[1].items()
            }
        )
        if t5_idx_token is not None:
            t5_idx = get_indices_from_tokens(t5_idx_token, word_list)
        return clip_idx, t5_idx

    def _get_t5_token_indices(
        self,
        prompt: str = None,
    ):
        if self.text_encoder_3 is None:
            return None
        else:
            ids = self.tokenizer_3(
                prompt,
            ).input_ids

            return {
                i: tok
                for tok, i in zip(
                    self.tokenizer_3.convert_ids_to_tokens(ids), range(len(ids))
                )
            }

    def _get_clip_token_indices(
        self,
        prompt: str,
        clip_model_index: int = 0,
    ):
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        tokenizer = clip_tokenizers[clip_model_index]
        ids = tokenizer(
            prompt,
        ).input_ids

        return {
            i: tok
            for tok, i in zip(tokenizer.convert_ids_to_tokens(ids), range(len(ids)))
        }

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""

        clip_ids_0 = self._get_clip_token_indices(prompt=prompt, clip_model_index=0)
        # Dont need, token between clip are aligned
        # clip_ids_1 = self._get_clip_token_indices(prompt=prompt, clip_model_index=0)
        t5_ids = (
            None
            if self.text_encoder_3 is None
            else self._get_t5_token_indices(prompt=prompt)
        )
        return clip_ids_0, t5_ids

    def register_attention_control(
        self,
        processor,
        processor_blocks=None,
    ):
        """
        Registers a new attention processor to the transformer's attention blocks.
        """
        attn_procs = {}
        att_count = 0

        # Use the existing processors as the default fallback
        default_processors = self.transformer.attn_processors

        for name, current_processor in default_processors.items():
            try:
                id_block = int(name.split(".")[1])
            except (ValueError, IndexError):
                # If the name doesn't fit the expected format,
                # just use the default processor and skip.
                attn_procs[name] = current_processor
                continue

            # Check if this block_id is in the list of blocks we want to control
            is_target_block = (
                processor_blocks is not None and id_block in processor_blocks
            )

            if is_target_block:
                attn_procs[name] = processor(
                    attnstore=self.attention_store,
                    block_id=id_block,
                )
                att_count += 1
            else:
                # Keep the default processor
                attn_procs[name] = current_processor

        # Set the processor map on the transformer
        self.transformer.set_attn_processor(attn_procs)

        # Update the attention store with the count, if it exists
        if self.attention_store is not None:
            self.attention_store.num_att_layers = att_count

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

        def setup_attention(criterion: AbstractGSN, average_t5_clip: bool):
            attention_store_cls = criterion.get_attention_store()
            params_attn_store, processor_blocks = criterion.get_params_attn_store(
                res_height=res_height, res_width=res_width, batch_size=batch_size
            )
            params_attn_store["average_t5_clip"] = average_t5_clip
            self.attention_store = attention_store_cls(
                executor=self.save_executor,
                store_attention_path=self.store_attention_path,
                **params_attn_store,
            )
            processor_cls = criterion.get_processor()
            self.register_attention_control(
                processor_cls,
                processor_blocks,
            )

        if iteref_step and self.criterion_iteref is not None:
            setup_attention(self.criterion_iteref, self.average_t5_clip)
        elif gsn_guidance_step and self.criterion_gsng is not None:
            setup_attention(self.criterion_gsng, self.average_t5_clip)
        elif create_signal and self.criterion_distrib is not None:
            setup_attention(self.criterion_distrib, self.average_t5_clip)
        else:
            self.register_attention_control(
                JointAttnProcessor2_0,
            )
            if self.attention_store is not None:
                self.attention_store.reset()
                self.attention_store = None

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        # GSN
        iteref_config: Optional[IterefConfig] = None,
        gsng_config: Optional[GsngConfig] = None,
        distrib_config: Optional[DistribConfig] = None,
        # Distrib
        prompt_embeds_distrib=None,
        negative_prompt_embeds_distrib=None,
        pooled_prompt_embeds_distrib=None,
        negative_pooled_prompt_embeds_distrib=None,
        generator_distrib: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        # Return or save intermediate features
        return_intermediate_features=False,
        average_t5_clip=True,
        decode_all=False,
        store_attention_path: Optional[str] = None,
        batch_size_vae: int = 16,
        **kwargs,
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
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
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
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
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
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
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
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.setup_save_executor(store_attention_path=store_attention_path)
        self.set_criterions(
            iteref_config=iteref_config,
            gsng_config=gsng_config,
            distrib_config=distrib_config,
            model="sd3",
            height=height,
            width=width,
            return_intermediate_features=return_intermediate_features,
        )
        self.average_t5_clip = average_t5_clip

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
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
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
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.criterion_distrib is not None:
            (
                prompt_embeds_distrib,
                negative_prompt_embeds_distrib,
                pooled_prompt_embeds_distrib,
                negative_pooled_prompt_embeds_distrib,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds_distrib,
                negative_prompt_embeds=negative_prompt_embeds_distrib,
                pooled_prompt_embeds=pooled_prompt_embeds_distrib,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_distrib,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=(
                    1
                    if not distrib_config.one_image_per_distrib
                    else num_images_per_prompt
                ),
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
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
        #### SETUP GSN ####
        self.setup_params(
            batch_size=batch_size, num_images_per_prompt=num_images_per_prompt
        )
        if self.criterion_gsng is not None or self.criterion_iteref is not None:
            text_embeddings = prompt_embeds

        if self.criterion_distrib is not None:
            text_embeddings_distrib = prompt_embeds_distrib
            pooled_embeddings_distrib = pooled_prompt_embeds_distrib
            mu = None

        if self.criterion_gsng is not None or self.criterion_iteref is not None:
            text_embeddings = prompt_embeds
            pooled_embeddings = pooled_prompt_embeds
        else:
            text_embeddings = None
            pooled_embeddings = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            if self.criterion_distrib is not None:
                prompt_embeds_distrib = torch.cat(
                    [negative_prompt_embeds_distrib, prompt_embeds_distrib], dim=0
                )
                pooled_prompt_embeds_distrib = torch.cat(
                    [
                        negative_pooled_prompt_embeds_distrib,
                        pooled_prompt_embeds_distrib,
                    ],
                    dim=0,
                )

        # 6. Denoising loop
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
                            if self.criterion_gsng is None
                            and self.criterion_iteref is None
                            else text_embeddings
                        ),
                        text_embeddings_distrib=(
                            None
                            if self.criterion_distrib is None
                            else text_embeddings_distrib
                        ),
                        batch_size=batch_size,
                        generator_distrib=generator_distrib,
                        mu=None if self.criterion_distrib is None else mu,
                        num_images_per_prompt=num_images_per_prompt,
                        pooled_embeddings=pooled_embeddings,
                        pooled_embeddings_distrib=(
                            None
                            if self.criterion_distrib is None
                            else pooled_embeddings_distrib
                        ),
                    )
                if self.criterion_distrib is not None:
                    latent_model_input = (
                        torch.cat([latents_distrib] * 2)
                        if self.do_classifier_free_guidance
                        else latents_distrib
                    )
                    # print(prompt_embeds_distrib.shape)
                    # print(pooled_prompt_embeds_distrib.shape)
                    # print(latent_model_input.shape)

                else:
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=(
                        prompt_embeds
                        if self.criterion_distrib is None
                        else prompt_embeds_distrib
                    ),
                    pooled_projections=(
                        pooled_prompt_embeds
                        if self.criterion_distrib is None
                        else pooled_prompt_embeds_distrib
                    ),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

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
                    mu = prev_sample.to(noise_pred.dtype)

                    latents_distrib = self.scheduler.step(
                        noise_pred, t, latents_distrib, return_dict=False
                    )[0]
                    if return_intermediate_features:
                        self.intermediate_values[f"mu_{int(t)}"] = (
                            mu.clone().detach().cpu()
                        )

                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]
                if return_intermediate_features:
                    # detect if last one
                    if i == len(timesteps) - 1:
                        self.intermediate_values[f"latents_{0}"] = (
                            latents.clone().detach().cpu()
                        )
                    else:
                        self.intermediate_values[
                            f"latents_{int(self.scheduler.timesteps[i + 1])}"
                        ] = (latents.clone().detach().cpu())

                # return value

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                if (
                    latents_distrib is not None
                    and latents_distrib.dtype != latents_dtype
                ):
                    if torch.backends.mps.is_available():
                        latents_distrib = latents_distrib.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

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
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            # image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

            # image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

            # latents_batch_1 = latents[:16]
            # latents_batch_2 = latents[16:]
            image = []
            for b in create_batch(latents, batch_size=batch_size_vae):
                image_batch = self.vae.decode(b, return_dict=False)[0]
                image_batch = self.image_processor.postprocess(
                    image_batch, output_type=output_type
                )
                image.extend(image_batch)
            if output_type == "pt":
                image = torch.stack(image)
            # image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            self.end_of_pipeline()
            return (image,)

        output = StableDiffusion3PipelineOutput(images=image)

        # decode all the intermediate values
        if not return_intermediate_features:
            self.end_of_pipeline()
            return output
        if decode_all and return_intermediate_features:
            dict_latents = {}
            for key, values in self.intermediate_values.items():
                if "var" in key or "mask" in key:
                    dict_latents[key] = values
                else:
                    images_temp = []
                    for b in create_batch(values, batch_size=batch_size_vae):
                        b = (
                            b / self.vae.config.scaling_factor
                        ) + self.vae.config.shift_factor

                        image_batch = self.vae.decode(
                            b.to(self._execution_device), return_dict=False
                        )[0]
                        image_batch = self.image_processor.postprocess(
                            image_batch, output_type=output_type
                        )
                        images_temp.extend(image_batch)
                    dict_latents[key] = images_temp
        elif return_intermediate_features:
            dict_latents = self.intermediate_values
        self.end_of_pipeline()
        return output, dict_latents


def create_batch(latents, batch_size=16):
    for i in range(0, len(latents), batch_size):
        yield latents[i : i + batch_size]
