from typing import List, Optional

import torch

from .utils import (
    AbstractGSN,
    average_dict,
    get_item_tensor,
    is_optimization_success,
)
from .utils_attention import AttentionStore
from .utils_processing_attention_loss import (
    attend_and_excite_loss,
    batch_attend_and_excite_loss,
    batch_iou_loss,
    iou_loss,
)


class IOUGSN(AbstractGSN):
    """GSN criterion combining the max loss from Attend and Excite and an IOU loss.
    The IOU loss is only activated if the max loss is above a certain threshold, and can be desactivated if the max loss is below a certain threshold.

    Attributes:
        smooth_attention (bool): Whether to apply smoothing to the attention maps.
        lambda_max (float): Weight for the max loss.
        lambda_iou (float): Weight for the IOU loss.
        threshold_max (float): Target threshold for the max loss to be reached during optimization.
        threshold_iou (float): Target threshold for the IOU loss to be reached during optimization.
        desactivate_iou (float): Threshold for the max loss to desactivate the IOU loss.
        mean (bool): Whether to average the max and IOU losses or to sum them.

    """

    def __init__(
        self,
        smooth_attention=True,
        lambda_max=1,
        lambda_iou=1,
        threshold_max=0.00,
        threshold_iou=0.00,
        desactivate_iou=1.0,
        mean=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.smooth_attention = smooth_attention
        # give the signal that you reach the threshold
        self.threshold_max = threshold_max
        self.threshold_iou = threshold_iou
        # amplify/reduce impact of loss
        self.lambda_max = lambda_max
        self.lambda_iou = lambda_iou
        # desactivate iou if the max threshold is not met
        self.desactivate_iou = desactivate_iou
        # k is 1 in initno loss
        self.mean = mean

    def compute_loss(
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
        return self._compute_loss(
            attention_store,
            attention_file_name,
            token_indices_clip,
            token_indices_t5,
            start_idx_clip,
            start_idx_t5,
            last_idx_clip,
            last_idx_t5,
        )

        # return self.__compute_loss_batched(
        #     attention_store,
        #     attention_file_name,
        #     token_indices_clip,
        #     token_indices_t5,
        #     start_idx_clip,
        #     start_idx_t5,
        #     last_idx_clip,
        #     last_idx_t5,
        # )

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
        dict_loss = {"loss_excite": 0.0, "loss_iou": 0.0}

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
            loss_max, _ = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {"token_indices": clip_token_positions_range},
                    "params_t5": {"token_indices": t5_token_positions_range},
                },
                function_loss=attend_and_excite_loss,
            )

            if loss_max < self.threshold_max:
                dict_loss["loss_excite"] += 0.0
            else:
                dict_loss["loss_excite"] += get_item_tensor(loss_max)

            loss_max *= self.lambda_max
            # IOU LOSS
            if self.desactivate_iou < loss_max:
                loss_iou = 0.0
                dict_loss["loss_iou"] += 1
            else:
                loss_iou, _ = attention_store.compute_cross_attention_loss(
                    attention_clip=attention_clip_position,
                    attention_t5=attention_t5_position,
                    params_function={
                        "params_clip": {
                            "token_indices": clip_token_positions_range,
                        },
                        "params_t5": {
                            "token_indices": t5_token_positions_range,
                        },
                    },
                    function_loss=iou_loss,
                )

                if loss_iou < self.threshold_iou:
                    dict_loss["loss_iou"] += 0
                else:
                    dict_loss["loss_iou"] += get_item_tensor(loss_iou)
                loss_iou *= self.lambda_iou
            if self.mean:
                loss += (loss_iou + loss_max) / 2
            else:
                loss += loss_iou + loss_max
        loss /= batch_size
        dict_loss = average_dict(dict_loss, batch_size)
        return loss, is_optimization_success(dict_loss), dict_loss

    def _compute_loss_batched(
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
        dict_loss = {"loss_excite": 0.0, "loss_iou": 0.0}

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
        # Process attention maps for the whole batch

        attention_clip = attention_store.batched_attention_maps_processing(
            attention_clip, start_idx_clip, last_idx_clip
        )
        attention_t5 = attention_store.batched_attention_maps_processing(
            attention_t5, start_idx_t5, last_idx_t5
        )

        attention_clip = attention_store.batched_attention_maps_per_position(
            attention_clip, token_indices_clip
        )
        attention_t5 = attention_store.batched_attention_maps_per_position(
            attention_t5, token_indices_t5
        )

        # We only have attention maps for the selected token indices
        # So we need to adjust the token indices accordingly
        token_indices_clip, token_indices_t5 = (
            (
                list(range(attention_clip.shape[-1]))
                if token_indices_clip is not None
                else None
            ),
            (
                list(range(attention_t5.shape[-1]))
                if token_indices_t5 is not None
                else None
            ),
        )
        if self.smooth_attention:
            attention_clip = attention_store.batch_attention_maps_smoothing(
                attention_clip,
            )
            if attention_t5 is not None:
                attention_t5 = attention_store.batch_attention_maps_smoothing(
                    attention_t5,
                )

        # MAX LOSS in batch
        loss_max_batch, _ = attention_store.compute_cross_attention_loss(
            attention_clip=attention_clip,
            attention_t5=attention_t5,
            function_loss=batch_attend_and_excite_loss,
            params_function={
                "params_clip": {"token_indices": token_indices_clip},
                "params_t5": {"token_indices": token_indices_t5},
            },
        )

        # Update dict_loss based on thresholds
        dict_loss["loss_excite"] = get_item_tensor(
            loss_max_batch[loss_max_batch >= self.threshold_max].sum()
        )

        loss_max_batch *= self.lambda_max

        # IOU LOSS in batch
        loss_iou_batch, _ = attention_store.compute_cross_attention_loss(
            attention_clip=attention_clip,
            attention_t5=attention_t5,
            function_loss=batch_iou_loss,
            params_function={
                "params_clip": {"token_indices": token_indices_clip},
                "params_t5": {"token_indices": token_indices_t5},
            },
        )

        # Deactivate IOU based on max loss
        iou_active_mask = self.desactivate_iou >= loss_max_batch
        loss_iou_batch_final = loss_iou_batch * iou_active_mask.float()

        dict_loss["loss_iou"] = (
            get_item_tensor(
                loss_iou_batch_final[loss_iou_batch_final >= self.threshold_iou].sum()
            )
            + torch.logical_not(iou_active_mask).sum().item()
        )

        loss_iou_batch_final *= self.lambda_iou

        if self.mean:
            total_loss_batch = (loss_iou_batch_final + loss_max_batch) / 2.0
        else:
            total_loss_batch = loss_iou_batch_final + loss_max_batch

        dict_loss = average_dict(dict_loss, batch_size)
        return total_loss_batch.mean(), is_optimization_success(dict_loss), dict_loss
