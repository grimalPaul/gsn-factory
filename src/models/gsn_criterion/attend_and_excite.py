from typing import List, Optional

from .utils import (
    AbstractGSN,
    average_dict,
    get_item_tensor,
    is_optimization_success,
)
from .utils_attention import AttentionStore
from .utils_processing_attention_loss import attend_and_excite_loss


class AttendAndExciteGSN(AbstractGSN):
    """Attend and Excite GSN criterion.
    Reimplementation of the criterion proposed in the paper "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models"
    https://github.com/yuval-alaluf/Attend-and-Excite
    """

    def __init__(self, smooth_attention: bool = True, threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.smooth_attention = smooth_attention
        self.threshold = threshold

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
    ):
        loss = 0.0
        dict_loss = {"loss_excite": 0.0}
        attention_clip, attention_t5 = attention_store.aggregate_attention(
            is_cross=True, attention_file_name=attention_file_name
        )
        batch_size = self.get_batch_size(attention_clip, attention_t5)

        token_indices_clip, attention_clip = self.initialize_token_attention(
            token_indices=token_indices_clip,
            start_idx=start_idx_clip,
            batch_size=batch_size,
            attention=attention_clip,
        )

        token_indices_t5, attention_t5 = self.initialize_token_attention(
            token_indices=token_indices_t5,
            start_idx=start_idx_t5,
            batch_size=batch_size,
            attention=attention_t5,
        )

        for attention_clip_, attention_t5_ in zip(attention_clip, attention_t5):
            attention_clip_ = attention_store.attention_maps_processing(
                attention_clip_, start_idx_clip, last_idx_clip
            )
            attention_t5_ = attention_store.attention_maps_processing(
                attention_t5_, start_idx_t5, last_idx_t5
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

            # MAX LOSS
            loss_excite, _ = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {"token_indices": clip_token_positions_range},
                    "params_t5": {"token_indices": t5_token_positions_range},
                },
                function_loss=attend_and_excite_loss,
            )

            if loss_excite <= 1 - self.threshold:
                # reach the threshold, then we need to break
                dict_loss["loss_excite"] = 0.0
            else:
                dict_loss["loss_excite"] += get_item_tensor(loss_excite)

            loss += loss_excite
        loss /= batch_size
        dict_loss = average_dict(dict_loss, batch_size)
        return loss, is_optimization_success(dict_loss), dict_loss
