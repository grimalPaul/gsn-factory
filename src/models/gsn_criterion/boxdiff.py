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
    compute_boxdiff_iteref,
    compute_loss_boxdiff_gsng,
)


class BoxDiffGSN(AbstractGSN):
    """
    Reimplementation of BoxDiff criterion proposed in the paper
    'BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion'
    https://github.com/showlab/BoxDiff

    """

    def __init__(
        self,
        smooth_attention=True,
        p: float = 0.2,
        l: int = 1,
        threshold=1.0,
        lambda_boxdiff=1.0,
        gsn_guidance=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gsn_guidance = gsn_guidance
        self.smooth_attention = smooth_attention
        # boxdiff
        self.p = p
        self.l = l
        self.threshold = threshold
        # lambda
        self.lambda_boxdiff = lambda_boxdiff

    def update_extra_parameters(
        self,
        num_images_per_prompt: int,
        batch_size: int,  # number of prompts
        bboxes: Union[List[List[float]], List[float]],
        token_indices_clip: Optional[List] = None,
        token_indices_t5: Optional[List] = None,
        start_idx_clip: Optional[Union[int, List[int]]] = None,
        start_idx_t5: Optional[Union[int, List[int]]] = None,
        last_idx_clip: Optional[Union[int, List[int]]] = None,
        last_idx_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        extra_params = super().update_extra_parameters(
            num_images_per_prompt,
            batch_size,
            token_indices_clip,
            token_indices_t5,
            start_idx_clip,
            start_idx_t5,
            last_idx_clip,
            last_idx_t5,
        )

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
        bboxes: List,
        token_indices_clip: Optional[List] = None,
        start_indices_clip: Optional[Union[int, List[int]]] = None,
        last_indices_clip: Optional[Union[int, List[int]]] = None,
        token_indices_t5: Optional[List] = None,
        start_indices_t5: Optional[Union[int, List[int]]] = None,
        last_indices_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        super().check_inputs(
            token_indices_clip,
            batch_size,
            start_indices_clip,
            last_indices_clip,
            token_indices_t5,
            start_indices_t5,
            last_indices_t5,
        )
        # check for masks
        # we need same number of masks as token_indices and same numbe rof batch size

        if len(bboxes) != batch_size:
            raise ValueError(
                f"Number of bboxes should be equal to batch size, got {len(bboxes)} bboxes and {batch_size} batch size"
            )

    def _compute_loss(
        self,
        attention_store: AttentionStore,
        attention_file_name: str,
        bboxes,
        token_indices_clip: Optional[List[int]] = None,
        token_indices_t5: Optional[List[int]] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
    ):
        loss = 0.0
        dict_loss = {"loss_boxdiff": 0.0}
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

        masks = self.bbox_to_masks(
            bboxes=bboxes,
            height=attention_store.res_height,
            width=attention_store.res_width,
            device=device,
            dtype=dtype,
        )
        if masks[0].device != device:
            masks = [mask.to(device) for mask in masks]

        scaled_bboxes = self.scale_all_bboxes(
            bboxes, attention_store.res_height, attention_store.res_width
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

            loss_, _ = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {
                        "token_indices": clip_token_positions_range,
                        "masks": masks,
                        "bboxes": scaled_bboxes,
                        "p": self.p,
                        "l": self.l,
                    },
                    "params_t5": {
                        "token_indices": t5_token_positions_range,
                        "masks": masks,
                        "bboxes": scaled_bboxes,
                        "p": self.p,
                        "l": self.l,
                    },
                },
                function_loss=(
                    compute_loss_boxdiff_gsng
                    if self.gsn_guidance
                    else compute_boxdiff_iteref
                ),
            )

            if loss_ <= 1 - self.threshold:
                dict_loss["loss_boxdiff"] += 0.0
            else:
                dict_loss["loss_boxdiff"] += get_item_tensor(loss_)
            loss += loss_ * self.lambda_boxdiff

        loss /= batch_size
        dict_loss = average_dict(dict_loss, batch_size)
        return loss, is_optimization_success(dict_loss), dict_loss

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
            x1, y1, x2, y2 = self.scale_bbox(bbox, height, width)
            mask = torch.zeros(height, width, dtype=dtype, device=device)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

        return masks

    def scale_all_bboxes(self, bboxes, height, width):
        return [self.scale_bbox(bbox, height, width) for bbox in bboxes]

    def scale_bbox(self, bbox, height, width):
        x1, y1, x2, y2 = bbox
        scale_w = width - 1
        scale_h = height - 1
        x1 = int(x1 * scale_w)
        y1 = int(y1 * scale_h)
        x2 = int(x2 * scale_w) + 1
        y2 = int(y2 * scale_h) + 1
        return x1, y1, x2, y2
