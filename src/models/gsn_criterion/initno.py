from typing import List, Optional

from .utils import (
    AbstractGSN,
    average_dict,
    get_item_tensor,
    is_optimization_success,
)
from .utils_attention import AttentionStore
from .utils_processing_attention_loss import (
    cross_attention_initno_loss,
    initno_alignment_loss,
    initno_loss_self_attention,
)


class InitNOGSN(AbstractGSN):
    """Reimplementation of the criterion proposed in the paper
    'Boosting Text-to-Image Diffusion Models via Initial Noise Optimization'
    https://github.com/xiefan-guo/initno

    Attributes:
        tau_cross_attn (float): Target threshold for cross attention loss to be reached during the optimization.
        tau_self_attn (float): Target threshold for self attention loss to be reached during the optimization.
        desactivate_iou (float): Threshold for the cross attention loss under which the self attention loss is desactivated (multiplied by 0).
        k (int): Number of top attention maps to consider for the loss computation.
        smooth_attention (bool): Whether to apply smoothing to the attention maps before computing the loss.
        gsn_guidance (bool): Whether the criterion is used for GSN guidance
        alpha_alignment (float): Alpha parameter for the exponential moving average of the attention maps used for the alignment loss when gsn_guidance is True.
        lambda_cross (float): Weight of the cross attention loss in the joint loss.
        lambda_self (float): Weight of the self attention loss in the joint loss.
    """

    def __init__(
        self,
        tau_cross_attn: float = 0.2,
        tau_self_attn: float = 0.3,
        desactivate_iou=0.5,
        k=1,
        smooth_attentions=True,
        gsn_guidance=False,
        alpha_alignment=0.0,
        lambda_cross=1.0,
        lambda_self=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau_cross_attn = tau_cross_attn
        self.tau_self_attn = tau_self_attn
        self.k = k

        self.smooth_attention = smooth_attentions
        self.gsn_guidance = gsn_guidance
        self.desactivate_iou = desactivate_iou
        self.alpha_alignment = alpha_alignment
        self.attention_maps_cache_clip = None
        self.attention_maps_cache_t5 = None
        self.lambda_cross = lambda_cross
        self.lambda_self = lambda_self

    def update_extra_parameters(self, **kwargs):
        return super().update_extra_parameters(**kwargs)

    def intepolate_attention_maps(
        self, attention_maps, cache_attention_maps, token_indices
    ):
        if attention_maps is None:
            return None
        if cache_attention_maps is None:
            cache_attention_maps = []
            for i in token_indices:
                cache_attention_maps.append(attention_maps[i].detach().clone())
        else:
            for i in token_indices:
                cache_attention_maps[i] = cache_attention_maps[
                    i
                ] * self.alpha_alignment + attention_maps[i].detach().clone() * (
                    1 - self.alpha_alignment
                )

        return cache_attention_maps

    def get_attention_cache(self):
        return self.attention_maps_cache_clip, self.attention_maps_cache_t5

    def _compute_loss(
        self,
        attention_store: AttentionStore,
        attention_file_name: str,
        token_indices_clip: Optional[List[int]] = None,
        token_indices_t5: Optional[List[int]] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
        attention_maps_cache_clip=None,
        attention_maps_cache_t5=None,
    ):  # aggregate maps
        joint_loss = 0
        self.initno_score = 0
        dict_loss = {"loss_excite": 0, "loss_iou": 0}
        cross_attention_clip, cross_attention_t5 = attention_store.aggregate_attention(
            is_cross=True, attention_file_name=attention_file_name
        )
        self_attention_clip, self_attention_t5 = attention_store.aggregate_attention(
            is_cross=False, attention_file_name=attention_file_name
        )

        batch_size = self.get_batch_size(cross_attention_clip, cross_attention_t5)

        token_indices_clip, cross_attention_clip = self.initialize_token_attention(
            token_indices=token_indices_clip,
            start_idx=start_idx_clip,
            batch_size=batch_size,
            attention=cross_attention_clip,
        )
        if cross_attention_clip[0] is None:
            self_attention_clip = [None] * batch_size
        token_indices_t5, cross_attention_t5 = self.initialize_token_attention(
            token_indices=token_indices_t5,
            start_idx=start_idx_t5,
            batch_size=batch_size,
            attention=cross_attention_t5,
        )
        if cross_attention_t5[0] is None:
            self_attention_t5 = [None] * batch_size

        for (
            cross_attention_clip_,
            cross_attention_t5_,
            self_attention_clip_,
            self_attention_t5_,
        ) in zip(
            cross_attention_clip,
            cross_attention_t5,
            self_attention_clip,
            self_attention_t5,
        ):
            attention_clip_ = attention_store.attention_maps_processing(
                cross_attention_clip_, start_idx_clip, last_idx_clip
            )
            attention_t5_ = attention_store.attention_maps_processing(
                cross_attention_t5_, start_idx_t5, last_idx_t5
            )
            attention_clip_position = attention_store.attention_maps_per_position(
                attention_clip_, token_indices_clip
            )
            attention_t5_position = attention_store.attention_maps_per_position(
                attention_t5_, token_indices_t5
            )

            clip_token_positions_range, t5_token_positions_range = (
                self.get_indices_per_position(
                    attention_clip_position, attention_t5_position
                )
            )

            if self.smooth_attention:
                attention_clip_position = attention_store.attention_maps_smoothing(
                    attention_clip_position, clip_token_positions_range
                )
                attention_t5_position = attention_store.attention_maps_smoothing(
                    attention_t5_position, t5_token_positions_range
                )

            (
                (
                    cross_attn_loss,
                    clean_cross_attention_loss,
                ),
                topk_coord_list_list,
            ) = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {
                        "token_indices": clip_token_positions_range,
                        "k": self.k,
                    },
                    "params_t5": {
                        "token_indices": t5_token_positions_range,
                        "k": self.k,
                    },
                },
                function_loss=cross_attention_initno_loss,
            )
            if self.gsn_guidance:
                self.attention_maps_cache_clip = self.intepolate_attention_maps(
                    attention_maps=attention_clip_position,
                    cache_attention_maps=attention_maps_cache_clip,
                    token_indices=clip_token_positions_range,
                )
                self.attention_maps_cache_t5 = self.intepolate_attention_maps(
                    attention_maps=attention_t5_position,
                    cache_attention_maps=attention_maps_cache_t5,
                    token_indices=t5_token_positions_range,
                )
                (
                    cross_attn_alignment_loss,
                    _,
                ) = attention_store.compute_cross_attention_loss(
                    attention_clip=attention_clip_position,
                    attention_t5=attention_t5_position,
                    params_function={
                        "params_clip": {
                            "token_indices": clip_token_positions_range,
                            "attention_maps_cache": self.attention_maps_cache_clip,
                            "smooth_attention": self.smooth_attention,
                        },
                        "params_t5": {
                            "token_indices": t5_token_positions_range,
                            "smooth_attention": self.smooth_attention,
                            "attention_maps_cache": self.attention_maps_cache_t5,
                        },
                    },
                    function_loss=initno_alignment_loss,
                )

            if isinstance(topk_coord_list_list, tuple):
                topk_coord_list_list_clip = topk_coord_list_list[0]
                topk_coord_list_list_t5 = topk_coord_list_list[1]
            else:
                topk_coord_list_list_clip = topk_coord_list_list
                topk_coord_list_list_t5 = None

            self_attn_loss, _ = attention_store.compute_self_attention_loss(
                attention_clip=self_attention_clip_,
                attention_t5=self_attention_t5_,
                params_function={
                    "params_clip": {
                        "topk_coord_list_list": topk_coord_list_list_clip,
                        "res_height": attention_store.res_height,
                        "res_width": attention_store.res_width,
                        "smooth_attention": self.smooth_attention,
                    },
                    "params_t5": {
                        "topk_coord_list_list": topk_coord_list_list_t5,
                        "res_height": attention_store.res_height,
                        "res_width": attention_store.res_width,
                        "smooth_attention": self.smooth_attention,
                    },
                },
                function_loss=initno_loss_self_attention,
            )

            if cross_attn_loss > self.desactivate_iou:
                self_attn_loss = self_attn_loss * 0

            if self.gsn_guidance:
                joint_loss += (
                    cross_attn_loss * self.lambda_cross
                    + clean_cross_attention_loss * 0.1
                    + cross_attn_alignment_loss * 0.1
                    + self_attn_loss * self.lambda_self
                )
            else:
                joint_loss += (
                    cross_attn_loss * self.lambda_cross
                    + self_attn_loss * self.lambda_self
                    + clean_cross_attention_loss * 1.0
                )
            self.initno_score += get_item_tensor(cross_attn_loss + self_attn_loss)
            if cross_attn_loss < self.tau_cross_attn:
                dict_loss["loss_excite"] += get_item_tensor(cross_attn_loss) * 0
            else:
                dict_loss["loss_excite"] += get_item_tensor(cross_attn_loss)
            if self_attn_loss < self.tau_self_attn:
                dict_loss["loss_iou"] += get_item_tensor(self_attn_loss) * 0
            else:
                dict_loss["loss_iou"] += get_item_tensor(self_attn_loss)

        joint_loss /= batch_size
        self.initno_score /= batch_size
        dict_loss = average_dict(dict_loss, batch_size)
        return joint_loss, is_optimization_success(dict_loss), dict_loss

    def get_params_attn_store(
        self,
        res_height,
        res_width,
        processor_blocks=None,
        cross_attn=True,
        self_attn=True,
        batch_size=1,
    ):
        return super().get_params_attn_store(
            res_height, res_width, processor_blocks, cross_attn, self_attn, batch_size
        )

    def get_initno_score(self):
        return self.initno_score
