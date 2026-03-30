from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import (
    Attention,
    JointAttnProcessor2_0,
)

from .utils_processing_attention_loss import (
    batch_attention_maps_smoothing,
    smooth_attention_map_single,
)


class AttentionStore:
    def __init__(
        self,
        res_height: int,
        res_width: int,
        executor: ThreadPoolExecutor,
        cross_attn: bool = True,
        self_attn: bool = True,
        store_attention_path=None,
        batch_size=1,
    ):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.cross_attn = cross_attn
        self.self_attn = self_attn
        self.batch_size = batch_size
        self.res_height = res_height
        self.res_width = res_width
        self.reset()
        if store_attention_path is not None:
            self.path = Path(store_attention_path)
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            self.path = None

        self.save_executor = executor
        self.file_indices = {}

    def __repr__(self):
        # return all the information about the store
        string = f"AttentionStore: {self.__class__.__name__}\n"
        string += f"Number of attention layers: {self.num_att_layers}\n"
        string += f"Current attention layer: {self.cur_att_layer}\n"
        string += f"Cross attention: {self.cross_attn}\n"
        string += f"Self attention: {self.self_attn}\n"
        string += f"Batch size: {self.batch_size}\n"
        string += f"Resolution height: {self.res_height}\n"
        string += f"Resolution width: {self.res_width}\n"
        string += f"Save path: {self.path}\n"
        return string

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def _save_tensor_to_disk(self, tensor_slice: torch.Tensor, file_path: Path):
        """Helper function to be run in a separate thread."""
        try:
            torch.save(tensor_slice, file_path)
        except Exception as e:
            print(f"Error saving tensor to {file_path}: {e}")

    def generate_unique_attention_filename(self, base_name: str) -> str:
        """
        Generates a unique filename by caching the last index
        instead of scanning the disk (glob) on every call.
        """
        if base_name not in self.file_indices:
            existing_files = list(self.path.glob(f"{base_name}*.pt"))
            if len(existing_files) > 0:
                try:
                    next_index = (
                        max(
                            [
                                int(f.name.split("_")[-1].split(".")[0])
                                for f in existing_files
                            ]
                        )
                        + 1
                    )
                except ValueError:
                    next_index = 0
            else:
                next_index = 0
            self.file_indices[base_name] = next_index

        index_to_use = self.file_indices[base_name]
        self.file_indices[base_name] += 1

        return f"{base_name}_{index_to_use:03d}"

    def save_attention(
        self,
        attention_maps: torch.Tensor,
        type_attention: str,
        attention_file_name: Union[str, List[str]],
    ):
        if self.path is None:
            return
        attention_maps_cpu = attention_maps.detach().cpu()

        if isinstance(attention_file_name, list):
            if len(attention_file_name) != self.batch_size:
                raise ValueError(
                    "Length of attention_file_name list must be equal to batch_size"
                )

            for i in range(self.batch_size):
                save_name = self.generate_unique_attention_filename(
                    f"{type_attention}_{attention_file_name[i]}"
                )
                full_path = self.path / f"{save_name}.pt"

                self.save_executor.submit(
                    self._save_tensor_to_disk, attention_maps_cpu[i], full_path
                )
        else:
            save_name = self.generate_unique_attention_filename(
                f"{type_attention}_{attention_file_name}"
            )
            full_path = self.path / f"{save_name}.pt"

            self.save_executor.submit(
                self._save_tensor_to_disk, attention_maps_cpu[0], full_path
            )

    @abstractmethod
    def get_empty_store(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def aggregate_attention(
        self,
        is_cross: bool = True,
        attention_file_name: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_attention_maps(self, **kwargs):
        return self.aggregate_attention(**kwargs)

    @staticmethod
    def attention_maps_processing(attention_maps, start_idx, end_idx):
        if attention_maps is None:
            return None
        attention_maps = 100 * attention_maps[:, :, start_idx:end_idx]
        return torch.nn.functional.softmax(attention_maps, dim=-1)

    @staticmethod
    def batched_attention_maps_processing(attention_maps, start_idx, end_idx):
        """
        Processes a batch of attention maps with shape (B, H, W, N).
        Applies softmax over the token dimension (N) independently for each batch.
        """
        if attention_maps is None or (
            isinstance(attention_maps, list) and attention_maps[0] is None
        ):
            return None

        # 1. Select the maps for the tokens of interest. Shape: (B, H, W, num_selected_tokens)
        attention_maps = attention_maps[:, :, :, start_idx:end_idx]

        # 2. Scale and apply softmax over the token dimension (N).
        attention_maps = 100 * attention_maps
        attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)

        return attention_maps

    @abstractmethod
    def compute_self_attention_loss(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_cross_attention_loss(self, **kwargs):
        raise NotImplementedError

    def attention_maps_per_position(self, attention_maps, token_indices):
        if attention_maps is None or token_indices is None:
            return None
        attention_per_position = []
        for idx in token_indices:
            if isinstance(idx, int):
                attention_per_position.append(attention_maps[:, :, idx])
            else:
                attention_per_position.append(attention_maps[:, :, idx].mean(dim=-1))
        return attention_per_position

    @staticmethod
    def batched_attention_maps_per_position(
        attention_maps: torch.Tensor, token_indices: List[Union[int, List[int]]]
    ) -> torch.Tensor:
        """
        Creates a new attention map tensor by selecting and averaging maps based on token groups.

        :param attention_maps: The source attention maps of shape (B, H, W, N).
        :param token_indices: A list defining token groups.
                            Example: [[5, 6], [8]] will create a new map from the average of
                            tokens 5 and 6, and a second map from token 8.
        :return: A new attention map tensor of shape (B, H, W, M), where M is the number of groups.
        """
        if attention_maps is None or token_indices is None:
            return None

        processed_groups = []
        for group in token_indices:
            if isinstance(group, int):
                # If it's a single integer, treat it as a group of one.
                indices = [group]
            else:
                # It's a list of indices.
                indices = group

            if not indices:
                continue

            # Select the attention maps for the current group.
            # Shape: (B, H, W, num_indices_in_group)
            selected_maps = attention_maps[:, :, :, indices]

            # Average the maps along the token dimension.
            # The keepdim=True is important to maintain the dimension for stacking.
            # Shape: (B, H, W, 1)
            averaged_map = selected_maps.mean(dim=-1, keepdim=True)

            processed_groups.append(averaged_map)

        if not processed_groups:
            # Return an empty tensor with the correct dimensions if no groups were processed.
            B, H, W, _ = attention_maps.shape
            return torch.empty(
                B, H, W, 0, device=attention_maps.device, dtype=attention_maps.dtype
            )

        # Stack the processed groups along the last dimension to create the new tensor.
        # Shape: (B, H, W, M)
        return torch.cat(processed_groups, dim=-1)

    def attention_maps_smoothing(self, attention_maps, token_indices):
        if attention_maps is None:
            return None
        for token_index in token_indices:
            attention_maps[token_index] = smooth_attention_map_single(
                attention_maps[token_index]
            )
        return attention_maps

    def batch_attention_maps_smoothing(self, attention_maps):
        if attention_maps is None or (
            isinstance(attention_maps, list) and attention_maps[0] is None
        ):
            return None
        return batch_attention_maps_smoothing(attention_maps)

    @abstractmethod
    def create_masks(
        self,
        attention_clip,
        token_indices_position_clip,
        attention_t5=None,
        token_indices_position_t5=None,
    ):
        raise NotImplementedError


class AttentionStoreSD1(AttentionStore):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_empty_store(self):
        store = {}
        attn_type = []
        if self.cross_attn:
            attn_type.append("cross")
        if self.self_attn:
            attn_type.append("self")
        for k in attn_type:
            for loc in ["down", "mid", "up"]:
                store[f"{loc}_{k}"] = []
        return store

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross and self.cross_attn:
            if attn.shape[1] == np.prod(self.res_height * self.res_width):
                self.step_store[f"{place_in_unet}_cross"].append(attn)
        elif self.cur_att_layer >= 0 and not is_cross and self.self_attn:
            # self attention
            if attn.shape[1] == np.prod(self.res_height * self.res_width):
                self.step_store[f"{place_in_unet}_self"].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def aggregate_attention(
        self,
        from_where: List[str] = ["down", "mid", "up"],
        is_cross: bool = True,
        attention_file_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out_cross = None
        out_self = None
        attention_maps = self.get_average_attention()
        if is_cross:
            if self.cross_attn:
                out_cross = []
                for location in from_where:
                    for item in attention_maps[f"{location}_{'cross'}"]:
                        cross_maps = item.reshape(
                            self.batch_size,
                            -1,
                            self.res_height,
                            self.res_width,
                            item.shape[-1],
                        )
                        out_cross.append(cross_maps)
                out_cross = torch.cat(out_cross, dim=1)
                out_cross = torch.mean(out_cross, dim=1)
                self.save_attention(
                    out_cross,
                    type_attention="cross_attention",
                    attention_file_name=attention_file_name,
                )
                return out_cross, [None] * self.batch_size
        else:
            if self.self_attn:
                out_self = []
                for location in from_where:
                    for item in attention_maps[f"{location}_self"]:
                        self_maps = item.reshape(
                            self.batch_size,
                            -1,
                            self.res_height,
                            self.res_width,
                            item.shape[-1],
                        )
                        out_self.append(self_maps)
                out_self = torch.cat(out_self, dim=1)
                out_self = torch.mean(out_self, dim=1)
                self.save_attention(
                    out_self,
                    type_attention="self_attention",
                    attention_file_name=attention_file_name,
                )
                return out_self, [None] * self.batch_size

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def compute_self_attention_loss(
        self, attention_clip, function_loss, params_function, **kwargs
    ):
        return function_loss(attention_clip, **params_function["params_clip"])

    def compute_cross_attention_loss(
        self, attention_clip, function_loss, params_function, **kwargs
    ):
        return function_loss(attention_clip, **params_function["params_clip"])

    def create_masks(
        self,
        attention_clip,
        token_indices_position_clip,
        attention_t5=None,
        token_indices_position_t5=None,
    ):
        return [attention_clip[i] for i in token_indices_position_clip]


class AttentionStoreSD3(AttentionStore):
    def __init__(
        self,
        average_t5_clip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.average_t5_clip = average_t5_clip  # mean on the attention maps

    def get_empty_store(self):
        return []

    def __call__(self, attn, block_id: int):
        if self.cur_att_layer >= 0:
            _, height, width = attn.shape
            attention_maps = attn.reshape(self.batch_size, -1, height, width)
            # self.step_store[block_id] = attention_maps
            # # mean head => batch, mean_head, patch, token
            attention_maps = attention_maps.mean(dim=1, keepdim=True)
            dim_image = self.res_height * self.res_width
            attention_maps = attention_maps[:, :, :dim_image, dim_image:]
            _, _, _, n_tokens = attention_maps.shape
            attention_maps = attention_maps.reshape(
                self.batch_size,
                -1,
                self.res_height,
                self.res_width,
                n_tokens,
            )
            self.step_store.append(attention_maps)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def aggregate_attention(
        self,
        is_cross: bool = True,
        attention_file_name: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""

        if not is_cross:
            raise ValueError("With SD3, we only supports cross attention for now")

        attention_maps = self.get_average_attention()
        attn_maps = torch.cat(attention_maps, dim=1)
        attn_maps = torch.mean(attn_maps, dim=1)
        self.save_attention(
            attn_maps,
            type_attention="cross_attention",
            attention_file_name=attention_file_name,
        )
        # split for CLIP and T5
        return attn_maps[:, :, :, :77], attn_maps[:, :, :, 77:]

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = []

    def compute_self_attention_loss(
        self,
        attention_clip,
        attention_t5,
        function_loss,
        params_function,
        **kwargs,
    ):
        raise NotImplementedError
        # not implemented for now
        # for now we cannot mix the self attention of each embedding
        loss_clip = function_loss(attention_clip, **params_function["params_clip"])
        if attention_t5 is not None:
            loss_t5 = function_loss(attention_t5, **params_function["params_t5"])
        else:
            loss_t5 = 0.0
        return loss_clip + loss_t5

    def compute_cross_attention_loss(
        self,
        attention_clip: Union[List[torch.Tensor], torch.Tensor],
        attention_t5: Optional[Union[List[torch.Tensor], torch.Tensor]],
        function_loss,
        params_function: Dict,
    ):
        """
        Unified version of compute_cross_attention_loss.
        Handles both list-based and batched tensor-based attention maps.

        :param attention_clip: List of tensors or batched tensor for CLIP attention maps.
        :param attention_t5: List of tensors or batched tensor for T5 attention maps (optional).
        :param function_loss: Loss function to compute the loss.
        :param params_function: Dictionary containing parameters for the loss function.
        :return: Combined loss and additional data.
        """
        token_indices_t5 = (
            params_function["params_t5"]["token_indices"]
            if "params_t5" in params_function
            else None
        )

        if (
            self.average_t5_clip
            and attention_t5 is not None
            and token_indices_t5 is not None
        ):
            if isinstance(attention_clip, list):
                averaged_attention = [
                    (attention_clip[i] + attention_t5[i]) / 2
                    for i in range(len(attention_clip))
                ]
            else:
                averaged_attention = (attention_clip + attention_t5) / 2
            return function_loss(averaged_attention, **params_function["params_clip"])

        # Compute losses for CLIP and T5 separately
        loss_clip, data_clip = function_loss(
            attention_clip, **params_function["params_clip"]
        )
        if attention_t5 is None or token_indices_t5 is None:
            return loss_clip, data_clip

        loss_t5, data_t5 = function_loss(attention_t5, **params_function["params_t5"])
        return ((loss_clip + loss_t5) / 2.0), (data_clip, data_t5)

    # @staticmethod
    # def addition_loss(loss_clip, loss_t5, mean_loss=False):
    #     def combine_losses(loss1, loss2):
    #         return (loss1 + loss2) / 2.0 if mean_loss else loss1 + loss2

    #     if isinstance(loss_clip, tuple):
    #         return tuple(
    #             combine_losses(loss_clip[i], loss_t5[i]) for i in range(len(loss_clip))
    #         )
    #     else:
    #         return combine_losses(loss_clip, loss_t5)

    def create_masks(
        self,
        attention_clip,
        token_indices_position_clip,
        attention_t5=None,
        token_indices_position_t5=None,
    ):
        if (
            token_indices_position_t5 is not None
            and attention_t5 is not None
            and self.average_t5_clip
        ):
            attention_clip = [
                (attention_clip[i] + attention_t5[i]) / 2
                for i in token_indices_position_clip
            ]
            return [attention_clip[i] for i in token_indices_position_clip]
        else:  # return only the clip attention as mask
            return [attention_clip[i] for i in token_indices_position_clip]


class JointAttnProcessor2_0Store(JointAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attnstore, block_id, **kwargs):
        super().__init__(**kwargs)
        self.attnstore = attnstore
        self.block_id = block_id
        self.save_attention = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # print(f"{query.shape=}, {key.shape=}, {value.shape=}")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # `context` projections.

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            # print(
            #     f"{encoder_hidden_states_query_proj.shape=}, {encoder_hidden_states_key_proj.shape=}, {encoder_hidden_states_value_proj.shape=}"
            # )
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        if self.save_attention:
            query = query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            key = key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            value = value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # Only need to store attention maps during the refinement process
            self.attnstore(attention_probs, self.block_id)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class GSNAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.save_attention = True

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if self.save_attention:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class FluxAttnProcessor2_0Store:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attnstore, block_id, **kwargs):
        super().__init__(**kwargs)
        self.attnstore = attnstore
        self.block_id = block_id
        self.save_attention = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if self.save_attention:
            query = query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            key = key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            value = value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attnstore(attention_probs, self.block_id)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class AttentionStoreFlux(AttentionStore):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_empty_store(self):
        return []

    def __call__(self, attn, block_id: int):
        if self.cur_att_layer >= 0:
            _, height, width = attn.shape
            attention_maps = attn.reshape(self.batch_size, -1, height, width)
            attention_maps = attention_maps.mean(dim=1, keepdim=True)
            dim_image = self.res_height * self.res_width
            attention_maps = attention_maps[:, :, :dim_image, dim_image:]
            _, _, _, n_tokens = attention_maps.shape
            attention_maps = attention_maps.reshape(
                self.batch_size,
                -1,
                self.res_height,
                self.res_width,
                n_tokens,
            )
            self.step_store.append(attention_maps)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def aggregate_attention(
        self,
        is_cross: bool = True,
        attention_file_name: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""

        if not is_cross:
            raise ValueError("With Flux, we only supports cross attention for now")

        attention_maps = self.get_average_attention()
        attn_maps = torch.cat(attention_maps, dim=1)
        attn_maps = torch.mean(attn_maps, dim=1)
        self.save_attention(
            attn_maps,
            type_attention="cross_attention",
            attention_file_name=attention_file_name,
        )
        return None, attn_maps

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = []

    def compute_self_attention_loss(
        self,
        attention_clip,
        attention_t5,
        function_loss,
        params_function,
        **kwargs,
    ):
        raise NotImplementedError

    def compute_cross_attention_loss(
        self, attention_t5, function_loss, params_function, **kwargs
    ):
        return function_loss(attention_t5, **params_function["params_t5"])

    def create_masks(
        self,
        attention_t5,
        token_indices_position_t5,
        **kwargs,
    ):
        return [attention_t5[i] for i in token_indices_position_t5]
