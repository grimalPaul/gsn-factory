import logging
from typing import List, Optional, Union

import torch

from .utils import (
    AbstractGSN,
    average_dict,
    check_inputs_token_adj_indices,
    fill_token_sequence_with_missing_indices,
    get_item_tensor,
    indices_to_position,
    is_optimization_success,
    merge_token_lists,
    position_for_subtrees,
)
from .utils_attention import AttentionStore
from .utils_processing_attention_loss import syngen_loss

logger = logging.getLogger(__name__)


class SynGen(AbstractGSN):
    """
    Reimplementation of SynGen criterion proposed in the paper
    'Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment'
    https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models
    """

    def __init__(
        self,
        threshold=-100.0,
        lambda_syngen=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.lambda_syngen = lambda_syngen

    def check_inputs(
        self,
        token_indices_clip: List,
        batch_size: int,
        last_idx_clip: Optional[Union[int, List[int]]] = None,
        adjs_token_clip: Optional[List] = None,
        start_indices_clip: Optional[Union[int, List[int]]] = None,
        token_indices_t5: Optional[List] = None,
        adjs_token_t5: Optional[List] = None,
        last_idx_t5: Optional[Union[int, List[int]]] = None,
        start_indices_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        """
        Sanity check for the input arguments.
        """
        super().check_inputs(
            token_indices_clip=token_indices_clip,
            batch_size=batch_size,
            last_indices_clip=last_idx_clip,
            last_indices_t5=last_idx_t5,
            token_indices_t5=token_indices_t5,
            start_indices_clip=start_indices_clip,
            start_indices_t5=start_indices_t5,
        )
        # check adjs token
        if adjs_token_clip is not None:
            check_inputs_token_adj_indices(adjs_token_clip, batch_size)

        if adjs_token_t5 is not None:
            check_inputs_token_adj_indices(adjs_token_t5, batch_size)

    def update_extra_parameters(
        self,
        num_images_per_prompt: int,
        batch_size: int,  # number of prompts
        token_indices_clip: Optional[List] = None,
        last_idx_clip: Optional[Union[int, List[int]]] = None,
        token_indices_t5: Optional[List] = None,
        adjs_token_clip: Optional[List] = None,
        adjs_token_t5: Optional[List] = None,
        last_idx_t5: Optional[Union[int, List[int]]] = None,
        start_idx_clip: Optional[Union[int, List[int]]] = None,
        start_idx_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        def init_indices(token_indices, adj_indices):
            if token_indices is None:
                token_indices = self.update_null_params_correct_dim(batch_size)
            if adj_indices is None:
                adj_indices = self.update_null_params_correct_dim(batch_size)
            return token_indices, adj_indices

        token_indices_clip, adjs_token_clip = init_indices(
            token_indices_clip, adjs_token_clip
        )
        token_indices_t5, adjs_token_t5 = init_indices(token_indices_t5, adjs_token_t5)

        token_indices_clip = self.update_token_indices(
            token_indices=token_indices_clip,
            num_images_per_prompt=num_images_per_prompt,
        )
        token_indices_t5 = self.update_token_indices(
            token_indices=token_indices_t5, num_images_per_prompt=num_images_per_prompt
        )

        adjs_token_clip = self.update_token_indices(
            token_indices=adjs_token_clip, num_images_per_prompt=num_images_per_prompt
        )
        adjs_token_t5 = self.update_token_indices(
            token_indices=adjs_token_t5, num_images_per_prompt=num_images_per_prompt
        )

        start_indices_clip = self.update_start_or_last_indices(
            idx=start_idx_clip,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=True,
        )
        start_indices_t5 = self.update_start_or_last_indices(
            idx=start_idx_t5,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=True,
        )

        last_idx_clip = self.update_start_or_last_indices(
            idx=last_idx_clip,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=False,
        )

        last_idx_t5 = self.update_start_or_last_indices(
            idx=last_idx_t5,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=False,
        )

        extra_params = [
            {
                "entities_token_clip": entities_token_clip_,
                "entities_token_t5": entities_token_t5_,
                "adjs_token_clip": adjs_token_clip_,
                "adjs_token_t5": adjs_token_t5_,
                "start_idx_clip": start_indices_clip_,
                "start_idx_t5": start_indices_t5_,
                "last_idx_clip": last_idx_clip_,
                "last_idx_t5": last_idx_t5_,
            }
            for entities_token_clip_, entities_token_t5_, adjs_token_clip_, adjs_token_t5_, start_indices_clip_, start_indices_t5_, last_idx_clip_, last_idx_t5_ in zip(
                token_indices_clip,
                token_indices_t5,
                adjs_token_clip,
                adjs_token_t5,
                start_indices_clip,
                start_indices_t5,
                last_idx_clip,
                last_idx_t5,
            )
        ]
        return extra_params

    def attention_maps_processing_syngen(
        self,
        attention_maps: torch.Tensor,
    ):
        if attention_maps is not None:
            return attention_maps * 100
        else:
            return attention_maps

    def _compute_loss(
        self,
        attention_store: AttentionStore,
        attention_file_name: str,
        entities_token_clip: Optional[List] = None,
        last_idx_clip: Optional[int] = None,
        start_idx_clip: Optional[int] = None,
        last_idx_t5: Optional[int] = None,
        entities_token_t5: Optional[List] = None,
        adjs_token_clip: Optional[List] = None,
        adjs_token_t5: Optional[List] = None,
        start_idx_t5: Optional[int] = None,
    ):
        loss_dict = {"loss_syngen": 0}
        loss = 0
        attention_clip, attention_t5 = attention_store.aggregate_attention(
            is_cross=True, attention_file_name=attention_file_name
        )
        if last_idx_clip is not None:
            last_idx_clip = last_idx_clip - start_idx_clip
        if last_idx_t5 is not None:
            last_idx_t5 = last_idx_t5 - start_idx_t5

        batch_size = self.get_batch_size(attention_clip, attention_t5)
        token_indices_clip = merge_token_lists(entities_token_clip, adjs_token_clip)
        token_indices_t5 = merge_token_lists(entities_token_t5, adjs_token_t5)

        # insert missing token
        filled_indices_clip = fill_token_sequence_with_missing_indices(
            start_idx_clip, last_idx_clip, token_indices_clip
        )
        filled_indices_t5 = fill_token_sequence_with_missing_indices(
            start_idx_t5, last_idx_t5, token_indices_t5
        )

        all_indices_clip, indices_to_pos_clip = indices_to_position(
            indices=filled_indices_clip
        )
        all_indices_t5, indices_to_pos_t5 = indices_to_position(
            indices=filled_indices_t5
        )

        subtrees_indices_clip_position = position_for_subtrees(
            entities_indices=entities_token_clip,
            adj_indices=adjs_token_clip,
            indices_to_position=indices_to_pos_clip,
        )
        subtrees_indices_t5_position = position_for_subtrees(
            indices_to_position=indices_to_pos_t5,
            adj_indices=adjs_token_t5,
            entities_indices=entities_token_t5,
        )
        for attention_clip_, attention_t5_ in zip(attention_clip, attention_t5):
            attention_clip_ = self.attention_maps_processing_syngen(attention_clip_)
            attention_t5_ = self.attention_maps_processing_syngen(attention_t5_)
            attention_clip_position = attention_store.attention_maps_per_position(
                attention_clip_, filled_indices_clip
            )
            attention_t5_position = attention_store.attention_maps_per_position(
                attention_t5_, filled_indices_t5
            )

            loss_linguistic, _ = attention_store.compute_cross_attention_loss(
                attention_clip=attention_clip_position,
                attention_t5=attention_t5_position,
                params_function={
                    "params_clip": {
                        "subtrees_indices": subtrees_indices_clip_position,
                        "all_indices": all_indices_clip,
                        "token_indices": filled_indices_clip,
                    },
                    "params_t5": {
                        "subtrees_indices": subtrees_indices_t5_position,
                        "all_indices": all_indices_t5,
                        "token_indices": filled_indices_t5,
                    },
                },
                function_loss=syngen_loss,
            )

            if loss_linguistic <= self.threshold:
                loss_dict["loss_syngen"] += get_item_tensor(loss_linguistic) * 0
            else:
                loss_dict["loss_syngen"] += get_item_tensor(loss_linguistic)

            loss += loss_linguistic * self.lambda_syngen

        loss /= batch_size
        loss_dict = average_dict(loss_dict, batch_size)
        return loss, is_optimization_success(loss_dict), loss_dict
