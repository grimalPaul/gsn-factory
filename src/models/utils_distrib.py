import numpy as np
import torch
from PIL import Image


def info_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return f"Memory Allocated M: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB"


def get_sigma_init(
    batch_size,
    channel,
    height,
    width,
    std=1,
    log_var=False,
    block=0,
    per_channel=False,
):
    """
    Initializes and returns a tensor filled with a constant variance (or log-variance) value based on the given parameters.

    This function creates a tensor with a shape determined by the batch size, channel, height, and width provided. The constant value in the tensor is computed either as the squared standard deviation or, if log_var is True, as twice the logarithm of the standard deviation. The layout of the tensor can be modified based on the block and per_channel parameters.

    Parameters
    ----------
    batch_size : int
        The number of samples in the batch.
    channel : int
        The number of channels for each sample.
    height : int
        The height dimension of the data.
    width : int
        The width dimension of the data.
    std : float, optional
        The standard deviation used to compute the variance, by default 1.
    log_var : bool, optional
        If True, compute the variance as 2 * log(std) (i.e., log-variance) instead of std squared, by default False.
    block : int, optional
        If greater than 0, the height * width dimensions are assumed to be divided into blocks of size block x block.
        A ValueError is raised if block**2 does not evenly divide the product of height and width.
    per_channel : bool, optional
        If True, the tensor shape includes the channel dimension, applying the constant value per channel.
        Otherwise, the constant is applied per pixel, by default False.

    Returns
    -------
    torch.Tensor
        A tensor of shape:
          - (batch_size, channel, height * width // block**2) if per_channel is True and block > 0,
          - (batch_size, channel, height * width) if per_channel is True and block == 0,
          - (batch_size, height * width // block**2) if per_channel is False and block > 0,
          - (batch_size, height * width) if per_channel is False and block == 0.
        Each element of the tensor is set to the computed variance or log-variance.

    Raises
    ------
    ValueError
        If block > 0 and block**2 does not evenly divide height * width.
    """
    if log_var:
        var = np.log(std) * 2
    else:
        var = std**2
    if block > 0:
        if height * width % block**2 != 0:
            raise ValueError("block**2 should divide height * width")
    if per_channel:  # pixel_channel
        if block > 0:
            return torch.ones(batch_size, channel, height * width // block**2) * var
        else:
            return torch.ones(batch_size, channel, height * width) * var
    else:  # pixel
        if block > 0:
            return torch.ones(batch_size, height * width // block**2) * var
        else:
            return torch.ones(batch_size, height * width) * var


def construct_with_block(params, channel, height, width, block_dim):
    nb_blocks = height * width // block_dim**2
    dim_reshape = int(np.sqrt(nb_blocks))
    params = (
        params.reshape(channel, dim_reshape, dim_reshape)
        .repeat_interleave(block_dim, dim=-1)
        .repeat_interleave(block_dim, dim=-2)
        .reshape(channel, -1)
    )
    return params


def compute_with_var(mu, var, noise, per_channel, log_var=False, block=0):
    if log_var:
        std = torch.exp(var * 0.5)
    else:
        std = torch.sqrt(var)
    batch_size, channel, height, width = noise.shape
    if per_channel:  # pixel channel
        if block > 1:
            std = construct_with_block(std, channel, height, width, block)
        L = torch.cat(
            [construct_diag_matrix(std[i]).unsqueeze(0) for i in range(std.shape[0])],
            dim=0,
        ).transpose(1, 2)
        return mu + torch.einsum("bcd, cde -> bce", noise.view(batch_size, channel, -1), L).view(
            batch_size, channel, height, width
        )
    else:  # pixel
        if block > 0:
            std = construct_with_block(std.unsqueeze(0), 1, height, width, block).squeeze(0)
        L = construct_diag_matrix(std)
        return mu + torch.matmul(noise.view(batch_size, channel, -1), L.t()).view(batch_size, channel, height, width)


def construct_lower_triangular(params, size, ensure_semi_definite=False):
    L = torch.eye(size).to(params.device)
    tril_indices = torch.tril_indices(row=size, col=size, offset=0)
    L[tril_indices[0], tril_indices[1]] = params
    return L


def construct_diag_matrix(params, ensure_semi_definite=False):
    return torch.diag(params)


def return_sigma(var, latents, per_channel, block, log_var=False):
    _, channel, height, width = latents.shape
    if log_var:
        var = torch.exp(var * 0.5)
    if per_channel:  # pixel_channel
        if block > 1:
            var = construct_with_block(var, channel, height, width, block)
        var = torch.cat(
            [construct_diag_matrix(var[i]).unsqueeze(0) for i in range(var.shape[0])],
            dim=0,
        )
    else:
        if block > 1:
            var = construct_with_block(var.unsqueeze(0), 1, height, width, block).squeeze(0)
        var = construct_diag_matrix(var)
    return var


def generate_bayer_matrix(resolution, block_size=(2, 2)):
    """
    Generates an efficient Bayer matrix for a given resolution and color block size.

    Parameters:
    resolution (tuple): The resolution of the matrix (height, width).
    block_size (tuple): The size of each color block (height, width) in pixels.

    Returns:
    np.ndarray: The Bayer matrix with values 0 for Red, 1 for Green, and 2 for Blue.
    """
    height, width = resolution
    block_h, block_w = block_size

    # Create the base pattern for a single 2x2 Bayer block
    base_pattern = np.array([[0, 1], [1, 2]], dtype=np.int8)

    # Tile the base pattern to match the specified resolution
    tile_h, tile_w = height // (2 * block_h), width // (2 * block_w)
    bayer_matrix = np.tile(base_pattern, (tile_h, tile_w))

    # Resize to add block size (by repeating each pixel in base pattern by block size)
    bayer_matrix = np.repeat(np.repeat(bayer_matrix, block_h, axis=0), block_w, axis=1)

    return bayer_matrix[:height, :width]


def numpy_to_image_bayer_matrix(bayer_matrix):
    """
    Efficiently visualizes the Bayer matrix using PIL.

    Parameters:
    bayer_matrix (np.ndarray): The Bayer matrix with 0 for Red, 1 for Green, and 2 for Blue.

    Returns:
    PIL.Image: The visualized image.
    """
    # Define colors for Red, Green, and Blue
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    # Map the Bayer matrix values to RGB colors
    color_image = colors[bayer_matrix]

    # Convert to a PIL Image
    return Image.fromarray(color_image)


def generate_random_colored_image(resolution, block_size=(2, 2)):
    """
    Generates a random-colored image with specified resolution and block size.

    Parameters:
    resolution (tuple): The resolution of the matrix (height, width).
    block_size (tuple): The size of each color block (height, width) in pixels.

    Returns:
    PIL.Image: The random-colored image.
    """
    height, width = resolution
    block_h, block_w = block_size

    # Determine the number of blocks needed along each dimension
    num_blocks_y = height // block_h + (height % block_h > 0)
    num_blocks_x = width // block_w + (width % block_w > 0)

    # Generate random colors for each block
    random_colors = np.random.randint(0, 256, (num_blocks_y, num_blocks_x, 3), dtype=np.uint8)

    # Repeat each color block to match the specified block size
    color_image = np.repeat(np.repeat(random_colors, block_h, axis=0), block_w, axis=1)

    # Crop to the exact resolution in case of oversize due to block tiling
    color_image = color_image[:height, :width]

    # Convert to a PIL Image
    return Image.fromarray(color_image)


def update_optimizer_lr(optimizer, new_lr: float) -> None:
    """
    Update learning rate of an optimizer dynamically.

    Args:
        optimizer: PyTorch optimizer instance
        new_lr: New learning rate value (float)
    """
    if new_lr <= 0:
        raise ValueError("Learning rate must be positive")

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
