from abc import abstractmethod
from typing import List, Optional, Union

import torch

from .utils_attention import (
    AttentionStore,
    AttentionStoreFlux,
    AttentionStoreSD1,
    AttentionStoreSD3,
    FluxAttnProcessor2_0Store,
    GSNAttnProcessor,
    JointAttnProcessor2_0Store,
)


def get_item_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def average_dict(dict_loss, batch_size):
    for key in dict_loss:
        dict_loss[key] /= batch_size
    return dict_loss


def is_optimization_success(dict_loss):
    sum_loss = sum(dict_loss.values())
    return sum_loss == 0


VERSION = ["flux", "sd1.4", "sd3"]


class AbstractGSN:
    """Abstract base class for Guided Spatial Attention (GSN) loss computation.

    This class provides a framework for computing loss based on attention mechanisms
    across different model versions (SD1.4, SD3, and Flux). It manages
    attention processor configuration, attention store initialization, and provides
    utility methods for handling token indices and batch processing.

    The class supports multiple attention mechanisms and allows customization of
    which model blocks store attention maps. Subclasses must implement the abstract
    methods `_compute_loss` and `_compute_loss_batched` to define specific loss
    computation logic.

    Attributes:
        version (str): The target model version for attention processing. Must be one
            of ["flux", "sd1.4", "sd3"]. Determines which attention processor and
            store to use.
        processor_blocks (Optional[List]): List of model blocks where attention
            processors will be registered. If None, default blocks for the version
            are used.
        path_to_save_attention (Optional[str]): File path for saving attention maps.
            If provided, attention stores will save their maps to this location.
        batched_processing_version (bool): Flag to enable batch processing version.
            Only relevant for specific loss implementations (e.g., IOU loss with SAGA).
            Defaults to False.
    """

    def __init__(
        self,
        version=None,
        processor_blocks: Optional[List] = None,
        path_to_save_attention=None,
        batched_processing_version=False,
        **kwargs,
    ):
        """Initialize the AbstractGSN instance.

        Args:
            version (Optional[str]): Model version for attention processing.
                Must be one of ["flux", "sd1.4", "sd3"].
            processor_blocks (Optional[List]): List of model blocks for attention
                processor registration. Defaults to None, which triggers selection
                of default blocks based on the version.
            path_to_save_attention (Optional[str]): Path to save attention maps.
                Defaults to None.
            batched_processing_version (bool): Enable batch processing variant.
                Currently used only for specific loss implementations. Defaults to False.
            **kwargs: Additional keyword arguments (ignored).
        """
        self.version = version
        self.processor_blocks = processor_blocks
        self.path_to_save_attention = path_to_save_attention
        # todo: a batch version only exist for IOU loss with SAGA
        self.batched_processing_version = batched_processing_version

    def __str__(self):
        """Return the class name as the string representation.

        Returns:
            str: The name of the class.
        """
        return f"{self.__class__.__name__}"

    def set_version(self, version):
        """Set and validate the model version.

        Args:
            version (str): Model version to set. Must be one of ["flux", "sd1.4", "sd3"].

        Raises:
            ValueError: If version is not in the supported VERSION list.
        """
        if version not in VERSION:
            raise ValueError(f"version must be in {VERSION}, get {version}")
        self.version = version

    def get_processor(self):
        """Get the appropriate attention processor class for the current version.

        Returns:
            type: The attention processor class corresponding to the model version.
                - GSNAttnProcessor for SD1.4
                - JointAttnProcessor2_0Store for SD3
                - FluxAttnProcessor2_0Store for Flux

        Raises:
            ValueError: If version is invalid or not supported.
        """
        if self.version in ["sd1.4"]:
            return GSNAttnProcessor
        elif self.version in ["sd3"]:
            return JointAttnProcessor2_0Store
        elif self.version in ["flux"]:
            return FluxAttnProcessor2_0Store
        else:
            raise ValueError(f"version must be in {VERSION}, get {self.version}")

    def get_attention_store(self):
        """Get the appropriate attention store class for the current version.

        The attention store is responsible for collecting and managing attention maps
        during the diffusion process.

        Returns:
            type: The attention store class corresponding to the model version.
                - AttentionStoreSD1 for SD1.4
                - AttentionStoreSD3 for SD3
                - AttentionStoreFlux for Flux

        Raises:
            ValueError: If version is invalid or not supported.
        """
        if self.version in ["sd1.4"]:
            return AttentionStoreSD1
        elif self.version in ["sd3"]:
            return AttentionStoreSD3
        elif self.version in ["flux"]:
            return AttentionStoreFlux
        else:
            raise ValueError(f"version must be in {VERSION}")

    def get_default_block(self, processor_blocks=None):
        """Get the default model blocks for attention processor registration.

        Returns the default blocks for the current model version if no blocks
        are explicitly provided, or returns the provided blocks if specified.

        Args:
            processor_blocks (Optional[List]): Explicit list of blocks to use.
                If None, returns default blocks for the version.

        Returns:
            List: Block identifiers for attention processor registration.
                - ["down_blocks.2", "up_blocks.1"] for SD1.4
                - list(range(24)) for SD3 and Flux
                - self.processor_blocks if set during initialization

        Raises:
            ValueError: If version is invalid or not supported.
        """
        if processor_blocks is None and self.processor_blocks is not None:
            return self.processor_blocks
        elif self.version in ["sd1.4"]:
            return ["down_blocks.2", "up_blocks.1"]
        elif self.version in ["sd3"]:
            return list(range(24))
        elif self.version in ["flux"]:
            return list(range(24))  # todo: study block to define usefull attention maps
        else:
            raise ValueError(f"version must be in {VERSION}")

    def get_params_attn_store(
        self,
        res_height: int,
        res_width: int,
        processor_blocks=None,
        cross_attn=True,
        self_attn=False,
        batch_size=1,
    ):
        """Build attention store parameters and default blocks.

        Prepares configuration parameters for attention store initialization along
        with the appropriate model blocks for processor registration.

        Args:
            res_height (int): Height of attention map resolution.
            res_width (int): Width of attention map resolution.
            processor_blocks (Optional[List]): Explicit blocks to use. Defaults to None.
            cross_attn (bool): Whether to store cross-attention maps. Defaults to True.
            self_attn (bool): Whether to store self-attention maps. Defaults to False.
            batch_size (int): Batch size for attention processing. Defaults to 1.

        Returns:
            Tuple[dict, List]: A tuple containing:
                - Dictionary of attention store parameters (res_height, res_width,
                  cross_attn, self_attn, batch_size)
                - List of processor blocks for attention registration
        """
        # todo: find a way to pass as hparams the default blocks
        processor_blocks = self.get_default_block(processor_blocks)
        return {
            "res_height": res_height,
            "res_width": res_width,
            "cross_attn": cross_attn,
            "self_attn": self_attn,
            "batch_size": batch_size,
        }, processor_blocks

    def update_start_or_last_indices(
        self, idx, num_images_per_prompt, batch_size, start=True
    ):
        """Expand start or end indices across batch and image replication.

        Converts indices (which may be scalar or per-prompt) into a list with one
        entry per image in the batch, accounting for multiple images per prompt.

        Args:
            idx (Optional[Union[int, List[int]]]): Start or end token index.
                Can be:
                - None: Uses default (1 for start, -1 for end)
                - int: Single index applied to all batches
                - List: One index per prompt in batch
            num_images_per_prompt (int): Number of images generated per prompt.
            batch_size (int): Number of prompts in the batch.
            start (bool): If True, uses default 1 for None input; else uses -1.
                Defaults to True.

        Returns:
            List[int]: Expanded indices with one value per image in full batch.
                Length = batch_size * num_images_per_prompt.
        """
        if idx is None or (isinstance(idx, list) and idx[0] is None):
            if start:
                idx = [1] * batch_size
            else:
                idx = [-1] * batch_size
        elif isinstance(idx, int):
            idx = [idx] * batch_size

        return self.mul_indices_per_num_images_per_prompt(
            idx=idx, num_images_per_prompt=num_images_per_prompt
        )

    def update_token_indices(
        self,
        token_indices,
        num_images_per_prompt,
    ):
        """Expand token indices across multiple images per prompt.

        Replicates per-prompt token indices to account for multiple images
        generated from each prompt.

        Args:
            token_indices (Union[List, List[List]]): Token indices specification.
                Can be:
                - [[idx1, idx2], [idx3]]: Per-prompt token index lists
                - [[[idx1, idx2]], [[idx3]]]: Already batched format
            num_images_per_prompt (int): Number of images per prompt.

        Returns:
            List: Token indices expanded to full batch size, with each prompt's
                indices repeated num_images_per_prompt times.
        """
        if token_indices[0] is not None and isinstance(token_indices[0][0], int):
            token_indices = [token_indices]
        return self.mul_indices_per_num_images_per_prompt(
            idx=token_indices, num_images_per_prompt=num_images_per_prompt
        )

    @staticmethod
    def mul_indices_per_num_images_per_prompt(idx, num_images_per_prompt):
        """Replicate per-prompt indices for multiple images per prompt.

        Args:
            idx (Union[List[int], List[List[int]]]): Per-prompt indices.
            num_images_per_prompt (int): Replication factor.

        Returns:
            List: Replicated indices where each prompt's index(es) appear
                num_images_per_prompt times consecutively.
        """
        indices = []
        for ind in idx:
            indices = indices + [ind] * num_images_per_prompt
        return indices

    @staticmethod
    def initialize_token_attention(token_indices, start_idx, batch_size, attention):
        """Initialize token attention with shifted indices or default None values.

        Args:
            token_indices (Optional[List]): Token indices for attention computation.
            start_idx (int): Starting index for token position shifting.
            batch_size (int): Batch size for creating default attention list.
            attention (List): Initial attention values.

        Returns:
            Tuple[Optional[List], List]: Shifted token indices and attention list.
                If token_indices is None, returns (None, [None] * batch_size).
        """
        if token_indices is not None:
            token_indices = shift_token_indices(token_indices, start_idx)
            return token_indices, attention
        else:
            attention = [None] * batch_size
            return token_indices, attention

    @staticmethod
    def get_indices_per_position(attention_clip_position, attention_t5_position):
        """
        Get the indices for clip and T5 token positions based on the provided attention positions.

        Args:
            attention_clip_position (Optional[List]): List of attention positions for clip tokens.
            attention_t5_position (Optional[List]): List of attention positions for T5 tokens.

        Returns:
            Tuple[Optional[List], Optional[List]]: Ranges of indices for clip and T5 token positions.
        """
        clip_token_positions_range = (
            None
            if attention_clip_position is None
            else list(range(len(attention_clip_position)))
        )
        t5_token_positions_range = (
            None
            if attention_t5_position is None
            else list(range(len(attention_t5_position)))
        )
        return clip_token_positions_range, t5_token_positions_range

    @staticmethod
    def get_batch_size(*args):
        """Extract batch size from the first tensor argument.

        Args:
            *args: Variable length argument list to search for tensors.

        Returns:
            int: Batch size (first dimension) of the first tensor found.

        Raises:
            ValueError: If no torch.Tensor is found in arguments.
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.shape[0]
        raise ValueError(f"batch size not found in the arguments: {args}")

    def compute_loss(
        self,
        **kwargs,
    ):
        """Public interface for loss computation.

        Delegates to the abstract method `_compute_loss` which must be implemented
        by subclasses. This method serves as the main entry point for loss calculation.

        Args:
            **kwargs: Keyword arguments passed to the subclass implementation of
                `_compute_loss`. Common arguments include:
                - attention_store: AttentionStore instance containing attention maps
                - attention_file_name: File path(s) for attention storage
                - token_indices_clip: Token indices for CLIP model
                - token_indices_t5: Token indices for T5 model
                - start_idx_clip: Starting index for CLIP tokens (default: 1)
                - start_idx_t5: Starting index for T5 tokens (default: 1)
                - last_idx_clip: Last index for CLIP tokens (default: -1)
                - last_idx_t5: Last index for T5 tokens (default: -1)

        Returns:
            The loss value computed by the subclass implementation.

        Raises:
            NotImplementedError: If called on AbstractGSN directly without a
                subclass implementation.
        """
        return self._compute_loss(
            **kwargs,
        )

    @abstractmethod
    def _compute_loss_batched(
        self,
        attention_store: AttentionStore,
        attention_file_name: Union[List[str], str],
        token_indices_clip: Optional[List[int]] = None,
        token_indices_t5: Optional[List[int]] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
        **kwargs,
    ):
        """Abstract method for batched loss computation.

        Subclasses must implement this method to support efficient batch processing
        of loss computation across multiple images or prompts. This variant is used
        when batched_processing_version is True.

        Args:
            attention_store (AttentionStore): Store containing collected attention maps
                from the diffusion model.
            attention_file_name (Union[List[str], str]): File path(s) for saving attention
                maps. Can be a single string or a list of strings for batch processing.
            token_indices_clip (Optional[List[int]]): Indices of CLIP tokens to focus on.
                Format: List of integers or nested list for batch. Defaults to None.
            token_indices_t5 (Optional[List[int]]): Indices of T5 tokens to focus on.
                Format: List of integers or nested list for batch. Defaults to None.
            start_idx_clip (int): Starting token index for CLIP (skip SOS token).
                Defaults to 1.
            start_idx_t5 (int): Starting token index for T5. Defaults to 1.
            last_idx_clip (int): Last token index for CLIP (EOT token). Defaults to -1.
            last_idx_t5 (int): Last token index for T5. Defaults to -1.
            **kwargs: Additional keyword arguments specific to subclass implementation.

        Returns:
            Loss value(s) computed on the batched input. Format depends on subclass.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_loss(
        self,
        attention_store: AttentionStore,
        attention_file_name: Union[List[str], str],
        token_indices_clip: Optional[List[int]] = None,
        token_indices_t5: Optional[List[int]] = None,
        start_idx_clip=1,
        start_idx_t5=1,
        last_idx_clip=-1,
        last_idx_t5=-1,
        **kwargs,
    ):
        """Abstract method for loss computation.

        Core abstract method that must be implemented by all subclasses to define
        the specific loss computation logic. This is the primary method invoked by
        the public `compute_loss` interface.

        Args:
            attention_store (AttentionStore): Store containing collected attention maps
                from the diffusion model across specified layers and tokens.
            attention_file_name (Union[List[str], str]): File path(s) for saving or
                loading attention maps. Can be a single string or list of strings.
            token_indices_clip (Optional[List[int]]): CLIP token indices of interest.
                Can be a flat list [[idx1, idx2], [idx3]] or nested for batch processing.
                Defaults to None.
            token_indices_t5 (Optional[List[int]]): T5 token indices of interest.
                Can be a flat list [[idx1, idx2], [idx3]] or nested for batch processing.
                Defaults to None.
            start_idx_clip (int): Start index for CLIP token sequence (typically 1 to
                skip SOS token). Defaults to 1.
            start_idx_t5 (int): Start index for T5 token sequence. Defaults to 1.
            last_idx_clip (int): Last index for CLIP token sequence (typically -1 for
                EOT token). Defaults to -1.
            last_idx_t5 (int): Last index for T5 token sequence. Defaults to -1.
            **kwargs: Additional implementation-specific keyword arguments.

        Returns:
            Loss value(s) computed from attention maps and token indices. The exact
            return type depends on the subclass implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Make the instance callable as a function.

        Provides a convenient interface for computing loss by allowing the instance
        to be called directly. Forwards all arguments to the `_compute_loss` method.

        Args:
            *args: Variable length positional arguments passed to `_compute_loss`.
            **kwargs: Arbitrary keyword arguments passed to `_compute_loss`.
                     (See `_compute_loss` for parameter details)

        Returns:
            Loss value computed by the `_compute_loss` implementation.

        Raises:
            NotImplementedError: If the subclass does not implement `_compute_loss`.

        Example:
            >>> gsn = ConcreteGSNSubclass(...)
            >>> loss = gsn(attention_store=store, token_indices_clip=indices)
        """
        return self._compute_loss(*args, **kwargs)

    def check_inputs(
        self,
        batch_size: int,
        token_indices_clip: Optional[Union[int, List[int]]] = None,
        start_indices_clip: Optional[Union[int, List[int]]] = None,
        last_indices_clip: Optional[Union[int, List[int]]] = None,
        token_indices_t5: Optional[Union[int, List[int]]] = None,
        start_indices_t5: Optional[Union[int, List[int]]] = None,
        last_indices_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        """Validate input arguments for loss computation.

        Performs sanity checks to ensure token indices and boundary indices match
        the expected batch size and are in the correct format.

        Args:
            batch_size (int): Expected batch size for validation.
            token_indices_clip (Optional[Union[int, List[int]]]): CLIP token indices.
                Must be valid if provided. Validated independently of T5.
            start_indices_clip (Optional[Union[int, List[int]]]): CLIP start indices.
                Can be a single int or list matching batch_size.
            last_indices_clip (Optional[Union[int, List[int]]]): CLIP end indices.
                Can be a single int or list matching batch_size.
            token_indices_t5 (Optional[Union[int, List[int]]]): T5 token indices.
                Must be valid if provided. Validated independently of CLIP.
            start_indices_t5 (Optional[Union[int, List[int]]]): T5 start indices.
                Can be a single int or list matching batch_size.
            last_indices_t5 (Optional[Union[int, List[int]]]): T5 end indices.
                Can be a single int or list matching batch_size.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            ValueError: If no token indices provided, batch sizes don't match,
                or indices are in an invalid format.

        Note:
            At least one of token_indices_clip or token_indices_t5 must be provided.
        """
        if token_indices_clip is None and token_indices_t5 is None:
            raise ValueError("You must provide token_indices")

        # clip
        if token_indices_clip is not None:
            check_inputs_token_indices(token_indices_clip, batch_size)
            if start_indices_clip is not None:
                check_inputs_start_last_indices(start_indices_clip, batch_size)
            if last_indices_clip is not None:
                check_inputs_start_last_indices(last_indices_clip, batch_size)

        # t5
        if token_indices_t5 is not None:
            check_inputs_token_indices(token_indices_t5, batch_size)
            if start_indices_t5 is not None:
                check_inputs_start_last_indices(start_indices_t5, batch_size)
            if last_indices_t5 is not None:
                check_inputs_start_last_indices(last_indices_t5, batch_size)

    def update_null_params_correct_dim(self, batch_size):
        """Create a list of None values matching batch size.

        Args:
            batch_size (int): Desired list length.

        Returns:
            List(None): List of None values with length equal to batch_size.
        """
        return [None] * batch_size

    def update_extra_parameters(
        self,
        num_images_per_prompt: int,
        batch_size: int,  # number of prompts
        token_indices_clip: Optional[List] = None,
        token_indices_t5: Optional[List] = None,
        start_idx_clip: Optional[Union[int, List[int]]] = None,
        start_idx_t5: Optional[Union[int, List[int]]] = None,
        last_idx_clip: Optional[Union[int, List[int]]] = None,
        last_idx_t5: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        """Aggregate and expand all token-related parameters for batch processing.

        Consolidates token indices and boundary indices handling, expanding them
        to the full batch size accounting for multiple images per prompt, and
        returns a list of parameter dictionaries, one per image in the batch.

        Args:
            num_images_per_prompt (int): Number of images per prompt.
            batch_size (int): Number of unique prompts in batch.
            token_indices_clip (Optional[List]): CLIP token indices.
            token_indices_t5 (Optional[List]): T5 token indices.
            start_idx_clip (Optional[Union[int, List[int]]]): CLIP start indices.
            start_idx_t5 (Optional[Union[int, List[int]]]): T5 start indices.
            last_idx_clip (Optional[Union[int, List[int]]]): CLIP end indices.
            last_idx_t5 (Optional[Union[int, List[int]]]): T5 end indices.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List[dict]: List of parameter dictionaries, one per image in batch.
                Each dict contains:
                - token_indices_clip: CLIP token indices for this image
                - token_indices_t5: T5 token indices for this image
                - start_idx_clip: CLIP start index for this image
                - start_idx_t5: T5 start index for this image
                - last_idx_clip: CLIP end index for this image
                - last_idx_t5: T5 end index for this image

        Note:
            Total returned list length = batch_size * num_images_per_prompt.
        """
        if token_indices_clip is None:
            token_indices_clip = self.update_null_params_correct_dim(batch_size)

        if token_indices_t5 is None:
            token_indices_t5 = self.update_null_params_correct_dim(batch_size)

        token_indices_clip = self.update_token_indices(
            token_indices=token_indices_clip,
            num_images_per_prompt=num_images_per_prompt,
        )
        token_indices_t5 = self.update_token_indices(
            token_indices=token_indices_t5,
            num_images_per_prompt=num_images_per_prompt,
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

        last_indices_clip = self.update_start_or_last_indices(
            idx=last_idx_clip,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=False,
        )

        last_indices_t5 = self.update_start_or_last_indices(
            idx=last_idx_t5,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            start=False,
        )

        extra_params = [
            {
                "token_indices_clip": token_indices_clip_,
                "token_indices_t5": token_indices_t5_,
                "start_idx_clip": start_indices_clip_,
                "start_idx_t5": start_indices_t5_,
                "last_idx_clip": indices_last_token_clip_,
                "last_idx_t5": indices_last_token_t5_,
            }
            for token_indices_clip_, token_indices_t5_, start_indices_clip_, start_indices_t5_, indices_last_token_clip_, indices_last_token_t5_ in zip(
                token_indices_clip,
                token_indices_t5,
                start_indices_clip,
                start_indices_t5,
                last_indices_clip,
                last_indices_t5,
            )
        ]
        # print(f"extra_params: {extra_params}")
        return extra_params


def check_inputs_start_last_indices(ind: Union[int, List[int]], batch_size: int):
    if isinstance(ind, int):
        ind = [ind]
    if len(ind) != batch_size:
        raise ValueError(
            f"start_indices_clip must be an int or a list of ints with the same length as the prompt batch size. get {ind}"
        )


def check_inputs_token_indices(
    token_indices: Union[List[List[int]], List[List[List[int]]]], batch_size: int
):
    indices_batched = (
        isinstance(token_indices, list)
        and isinstance(token_indices[0], list)
        and isinstance(token_indices[0][0], list)
        and isinstance(token_indices[0][0][0], int)
    )
    indices_not_batched = (
        isinstance(token_indices, list)
        and isinstance(token_indices[0], list)
        and isinstance(token_indices[0][0], int)
    )
    if not indices_batched and not indices_not_batched:
        raise ValueError(
            f"token_indices must be a list of list of ints or a list of a list of a list of ints, get {token_indices}"
        )

    if indices_not_batched:
        indices_batch_size = 1
    elif indices_batched:
        indices_batch_size = len(token_indices)

    if indices_batch_size != batch_size:
        raise ValueError(
            f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {batch_size}"
        )


def check_inputs_token_adj_indices(
    adj_indices: Union[List[List[List[int]]], List[List[List[List[int]]]]],
    batch_size: int,
):
    adj_indices_not_batched = (
        isinstance(adj_indices, list)
        and isinstance(adj_indices[0], list)
        and isinstance(adj_indices[0][0], list)
        and isinstance(adj_indices[0][0][0], int)
    )
    adj_indices_not_batched = (
        isinstance(adj_indices, list)
        and isinstance(adj_indices[0], list)
        and isinstance(adj_indices[0][0], list)
        and isinstance(adj_indices[0][0][0], list)
        and isinstance(adj_indices[0][0][0][0], int)
    )
    if not adj_indices_not_batched and not adj_indices_not_batched:
        raise ValueError(
            f"adj_indices must be a list of list of list of ints or a list of a list of a list of a list of ints, get {adj_indices}"
        )
    if adj_indices_not_batched:
        indices_batch_size = 1
    elif adj_indices_not_batched:
        indices_batch_size = len(adj_indices)

    if indices_batch_size != batch_size:
        raise ValueError(
            f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {batch_size}"
        )


def merge_token_lists(entity_indices, adj_indices):
    if adj_indices is None:
        adj_indices = []
    if entity_indices is None:
        entity_indices = []
    # pop empty lists inside the lists
    entity_indices = [x for x in entity_indices if x]
    adj_indices = [x for x in adj_indices if x]
    adj_indices = [item for sublist in adj_indices for item in sublist]

    # merge lists
    indices = entity_indices + adj_indices
    # sort lists based on first element
    indices = sorted(indices, key=lambda x: x[0])
    if len(indices) == 0:
        return None
    else:
        return indices


def fill_token_sequence_with_missing_indices(
    start_index: int, last_index: int, sparse_indices: List[Union[int, List[int]]]
) -> Optional[List[Union[int, List[int]]]]:
    """
    Fills the gaps between specific token groups (sublists) with individual token indices
    to create a continuous sequence from start_index to last_index.

    Args:
        start_index: The starting token index (usually 1 to skip SOS).
        last_index: The final token index (usually the index of EOT).
        sparse_indices: A list of token groups (e.g., [[3, 4], [7]]) representing entities or attributes.

    Returns:
        A list where gaps are filled with integers, and original groups are preserved.
    """
    if sparse_indices is None:
        return None
    filled_list = []
    current_index = start_index
    for sublist in sparse_indices:
        while current_index < sublist[0]:
            filled_list.append(current_index)
            current_index += 1
        filled_list.append(sublist)
        current_index = sublist[-1] + 1
    while current_index <= last_index:
        filled_list.append(current_index)
        current_index += 1
    return filled_list


def position_for_subtrees(entities_indices, adj_indices, indices_to_position):
    if entities_indices is None:
        return None

    if adj_indices is not None:
        if len(adj_indices) != len(entities_indices):
            raise ValueError(
                "adj_indices and entities_indices must have the same length"
            )
    else:
        adj_indices = [None] * len(entities_indices)
    updated_indices = []
    for entities, adj in zip(entities_indices, adj_indices):
        adj_list = []
        if adj is not None:
            for adj_indice in adj:
                adj_list.append(indices_to_position[to_tuple(adj_indice)])
        updated_indices.append(adj_list + [indices_to_position[to_tuple(entities)]])

    return updated_indices


def shift_token_indices(token_indices, start_idx):
    updated_indices = []
    for idx in token_indices:
        if isinstance(idx, int):
            idx = idx - start_idx
        else:
            idx = [i - start_idx for i in idx]
        updated_indices.append(idx)
    return updated_indices


def to_tuple(indices):
    if isinstance(indices, int):
        return (indices,)
    else:
        return tuple(indices)


def indices_to_position(indices):
    if indices is None:
        return None, None
    updated_indices = [i for i, _ in enumerate(indices)]
    indices_to_pos = {}
    for pos, idx in enumerate(indices):
        indices_to_pos[to_tuple(idx)] = pos

    return updated_indices, indices_to_pos
