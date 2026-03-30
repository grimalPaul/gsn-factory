from typing import List, Optional, Union

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
    get_clean_otsu_mask,
    iou_mask,
)


class RetentionLoss(AbstractGSN):
    """
    GSN criterion that enforces attention retention inside target regions.

    This criterion computes a mask-based retention loss (and optional IoU-based term)
    between CLIP and T5 cross-attention maps using provided binary masks or normalized bounding boxes.

    Attributes:
        smooth_attention (bool): Whether to apply smoothing to attention maps.
        threshold_mask (float): Minimum mask loss value to consider when aggregating.
        threshold_iou (float): Minimum IoU value threshold (if IoU term is used).
        lambda_mask (float): Weight applied to the mask-based loss.
        lambda_iou (float): Weight applied to the IoU-based loss.
        mean (bool): If True, average IoU and attend-excite losses.
    """

    def __init__(
        self,
        smooth_attention=True,
        threshold_mask=0.0,
        threshold_iou=0.0,
        lambda_mask=1.0,
        lambda_iou=1.0,
        mean=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smooth_attention = smooth_attention
        self.threshold_mask = threshold_mask
        self.threshold_iou = threshold_iou
        self.lambda_mask = lambda_mask
        self.lambda_iou = lambda_iou
        self.masks_available = False
        self.mean = mean

    def update_extra_parameters(
        self,
        num_images_per_prompt: int,
        batch_size: int,  # number of prompts
        token_indices_clip: Optional[List] = None,
        masks: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None,
        bboxes: Optional[Union[List[List[float]], List[float]]] = None,
        token_indices_t5: Optional[List] = None,
        start_idx_clip: Optional[Union[int, List[int]]] = None,
        start_idx_t5: Optional[Union[int, List[int]]] = None,
        last_idx_clip: Optional[Union[int, List[int]]] = None,
        last_idx_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        extra_params = super().update_extra_parameters(
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            token_indices_clip=token_indices_clip,
            token_indices_t5=token_indices_t5,
            start_idx_clip=start_idx_clip,
            start_idx_t5=start_idx_t5,
            last_idx_clip=last_idx_clip,
            last_idx_t5=last_idx_t5,
        )
        if masks is None and bboxes is None:
            self.masks_available = False
        else:
            if masks is not None:
                if isinstance(masks[0], torch.Tensor):
                    masks = [masks]
                masks = self.mul_indices_per_num_images_per_prompt(
                    masks, num_images_per_prompt
                )
                for item, masks_ in zip(extra_params, masks):
                    item["masks"] = masks_
                return extra_params
            else:
                if not isinstance(bboxes[0], list):
                    bboxes = [bboxes]
                bboxes = self.mul_indices_per_num_images_per_prompt(
                    bboxes, num_images_per_prompt
                )
                for item, bboxes_ in zip(extra_params, bboxes):
                    item["bboxes"] = bboxes_
                return extra_params

    def check_inputs(
        self,
        batch_size: int,
        token_indices_clip: Optional[List] = None,
        masks: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None,
        bboxes: Optional[Union[List[List[float]], List[float]]] = None,
        start_indices_clip: Optional[Union[int, List[int]]] = None,
        last_indices_clip: Optional[Union[int, List[int]]] = None,
        token_indices_t5: Optional[List] = None,
        start_indices_t5: Optional[Union[int, List[int]]] = None,
        last_indices_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        # print(
        #     f"{token_indices_clip=}, {batch_size=}, {start_indices_clip=}, {last_indices_clip=}, {token_indices_t5=}, {start_indices_t5=}, {last_indices_t5=}{kwargs=}"
        # )
        super().check_inputs(
            token_indices_clip=token_indices_clip,
            batch_size=batch_size,
            start_indices_clip=start_indices_clip,
            last_indices_clip=last_indices_clip,
            token_indices_t5=token_indices_t5,
            start_indices_t5=start_indices_t5,
            last_indices_t5=last_indices_t5,
        )
        # check for masks
        # we need same number of masks as token_indices and same numbe rof batch size
        if masks is None:
            self.masks_available = False
        else:
            self.masks_available = True
            if isinstance(masks[0], torch.Tensor):
                masks = [masks]
            if len(masks) != batch_size:
                raise ValueError(
                    f"Number of masks should be equal to batch size, got {len(masks)} masks and {batch_size} batch size"
                )
        if bboxes is not None:
            self.masks_available = True
            if len(bboxes) != batch_size:
                raise ValueError(
                    f"Number of bboxes should be equal to batch size, got {len(bboxes)} bboxes and {batch_size} batch size"
                )

    def _compute_loss(
        self,
        attention_store: AttentionStore,
        attention_file_name: str,
        token_indices_clip: Optional[List[int]] = None,
        masks: Optional[List[torch.Tensor]] = None,
        bboxes: Optional[List] = None,
        token_indices_t5: Optional[List[int]] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
    ):
        if not self.masks_available:
            raise ValueError(
                "Masks are not available. Either pass it as params or you use distrib/iteref before to generate masks"
            )
        if masks is None and bboxes is None:
            raise ValueError("Either masks or bboxes should be passed")

        loss = 0.0
        dict_loss = {"mask_loss": 0.0}
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
        device = (
            attention_clip.device
            if attention_clip[0] is not None
            else attention_t5.device
        )
        dtype = (
            attention_clip.dtype
            if attention_clip[0] is not None
            else attention_t5.dtype
        )

        if masks is None and bboxes is not None:
            masks = self.bbox_to_masks(
                bboxes=bboxes,
                height=attention_store.res_height,
                width=attention_store.res_width,
                device=device,
                dtype=dtype,
            )

        if masks[0].device != device:
            masks = [mask.to(device) for mask in masks]

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

            loss_mask, _ = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {
                        "token_indices": clip_token_positions_range,
                        "masks": masks,
                    },
                    "params_t5": {
                        "token_indices": t5_token_positions_range,
                        "masks": masks,
                    },
                },
                function_loss=iou_mask,
            )
            loss_ae, _ = attention_store.compute_cross_attention_loss(
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
                function_loss=attend_and_excite_loss,
            )
            if self.mean:
                loss_mask = (loss_mask + loss_ae) / 2
            else:
                loss_mask += loss_ae

            if loss_mask < self.threshold_mask:
                dict_loss["mask_loss"] += 0.0
            else:
                dict_loss["mask_loss"] += get_item_tensor(loss_mask)
            loss += loss_mask * self.lambda_mask

        loss /= batch_size
        dict_loss = average_dict(dict_loss, batch_size)
        return loss, is_optimization_success(dict_loss), dict_loss

    @staticmethod
    def get_masks(
        attention_store: AttentionStore,
        # common args for loss
        token_indices_clip: Optional[List[int]] = None,
        token_indices_t5: Optional[List[int]] = None,
        # case of syngen loss
        entities_token_t5: Optional[List] = None,
        entities_token_clip: Optional[List] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
        smooth_attention=True,
        **kwargs,
    ):
        """
        Generates masks based on attention maps.

        """

        attention_clip, attention_t5 = attention_store.aggregate_attention(
            is_cross=True, attention_file_name=None
        )
        batch_size = AbstractGSN.get_batch_size(attention_clip, attention_t5)
        if token_indices_clip is None:
            token_indices_clip = entities_token_clip
        if token_indices_t5 is None:
            token_indices_t5 = entities_token_t5

        token_indices_clip, attention_clip = AbstractGSN.initialize_token_attention(
            token_indices=token_indices_clip,
            start_idx=start_idx_clip,
            batch_size=batch_size,
            attention=attention_clip,
        )

        token_indices_t5, attention_t5 = AbstractGSN.initialize_token_attention(
            token_indices=token_indices_t5,
            start_idx=start_idx_t5,
            batch_size=batch_size,
            attention=attention_t5,
        )

        masks = [
            [] for _ in range(len(token_indices_clip))
        ]  # for each group of tokens one mask
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
                AbstractGSN.get_indices_per_position(
                    attention_clip_position, attention_t5_position
                )
            )

            if smooth_attention:
                attention_clip_position = attention_store.attention_maps_smoothing(
                    attention_clip_position, clip_token_positions_range
                )
                attention_t5_position = attention_store.attention_maps_smoothing(
                    attention_t5_position, t5_token_positions_range
                )

            masks_ = attention_store.create_masks(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                token_indices_position_clip=clip_token_positions_range,
                token_indices_position_t5=t5_token_positions_range,
            )
            for i in clip_token_positions_range:  # mask for each entity
                masks[i].append(masks_[i])

        # processing maps in case of distribution
        for i in range(len(masks)):
            masks[i] = get_clean_otsu_mask(
                torch.mean(torch.stack(masks[i], dim=0), dim=0)
            )
        return masks

    def bbox_to_masks(self, bboxes, height, width, dtype, device):
        """Convert bounding boxes to binary masks.

        Args:
            bboxes: List of bounding boxes with normalized coordinates (x1, y1, x2, y2)
            height: Height of the output masks
            width: Width of the output masks
            dtype: Tensor dtype
            device: Target device
        """
        masks = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            mask = torch.zeros(height, width, dtype=dtype, device=device)
            scale_w = width - 1
            scale_h = height - 1
            x1 = int(x1 * scale_w)
            y1 = int(y1 * scale_h)
            x2 = int(x2 * scale_w) + 1
            y2 = int(y2 * scale_h) + 1
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

        return masks
