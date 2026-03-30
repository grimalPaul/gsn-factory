import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw

from src.models.gsn_criterion import SynGen
from src.models.pipeline import StableDiffusion3PipelineGSN, StableDiffusionGSN


def plot_images(images, titles=None, images_per_row=5, size_img=128):
    n = len(images)
    rows = n // images_per_row
    if n % images_per_row != 0:
        rows += 1
    factor_columns = 5
    fig, axs = plt.subplots(
        rows,
        images_per_row,
        figsize=(images_per_row * factor_columns, factor_columns * rows),
    )
    if size_img is None:
        new_size = None
    else:
        new_size = (size_img, size_img)
    for i, ax in enumerate(axs.flatten()):
        if i < n:
            if new_size is None:
                ax.imshow(images[i])
            else:
                ax.imshow(images[i].resize(new_size))
            if titles is not None:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(titles[i])
            else:
                ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout()


def dim_reshape(var):
    n_size = np.prod(var.shape)
    h = int(np.sqrt(n_size))
    return var.reshape(h, h)


def visu_variance(dict_values, idx=0, factor=1):
    key = []
    for k in list(dict_values.keys()):
        if "var" in k:
            key.append(k)

    if len(key) != 0:
        idx2var = {}
        idx2var_optimized = {}
        for v in key:
            splited = v.split("_")
            if len(splited) == 4:
                idx2var_optimized[int(splited[-1])] = v
            else:
                idx2var[int(splited[-1])] = v
        if idx not in idx2var and idx not in idx2var_optimized:
            print(f"{idx} not available {idx2var.keys()} available")
        if len(dict_values[idx2var[idx]].shape) == 3:
            channel = dict_values[idx2var[idx]].shape[0]
            fig, axs = plt.subplots(2, channel, figsize=(channel * 3, 2 * 3))
            for i, v in enumerate([idx2var[idx], idx2var_optimized[idx]]):
                for j in range(channel):
                    sns.heatmap(dim_reshape(np.diag(dict_values[v][j] * factor)), ax=axs[i][j])
        else:
            fig, axs = plt.subplots(2, 1, figsize=(3, 2 * 3))
            sns.heatmap(dim_reshape(np.diag(dict_values[idx2var[idx]] * factor)), ax=axs[0])
            sns.heatmap(
                dim_reshape(np.diag(dict_values[idx2var_optimized[idx]] * factor)),
                ax=axs[1],
            )


def visu_mu(dict_values, idx=0, size_img=128):
    key = []
    timestep = []
    for k in list(dict_values.keys()):
        if "mu" in k:
            key.append(k)
            timestep.append(int((k.split("_")[-1])))

    if len(key) != 0:
        key_to_plot = []
        timestep.sort(reverse=True)
        timestep = set(timestep)
        for t in timestep:
            if f"mu_{t}" in key:
                key_to_plot.append(f"mu_{t}")
            if f"mu_init_{t}" in key:
                key_to_plot.append(f"mu_init_{t}")
            if f"mu_optimized_{t}" in key:
                key_to_plot.append(f"mu_optimized_{t}")
        mu = []
        for key in key_to_plot:
            mu.append(dict_values[key][idx])
        if not isinstance(mu[0], Image.Image):
            print("mu's are not images")
            return None
        plot_images(mu, titles=key_to_plot, size_img=size_img)


def visu_mask(dict_values, idx=0, size_image=None):
    key = []
    timestep = []
    for k in list(dict_values.keys()):
        if "mask" in k:
            key.append(k)
            timestep.append(int(k.split("_")[-1]))

    if len(key) != 0:
        key_to_plot = []
        timestep.sort(reverse=True)
        timestep = set(timestep)
        key_to_plot = [f"masks_{t}" for t in timestep]
        n_entities = len(dict_values[key_to_plot[0]][idx])
        # size_image =  dict_values[key_to_plot[0]][idx][0].shape[0]
        images = []
        for key in key_to_plot:
            masks = dict_values[key][idx]
            print(masks[0].shape)
            images_ = [Image.fromarray(m.numpy().astype(np.float32) * 255) for m in masks]
            images.extend(images_)
        plot_images(images, size_img=size_image, images_per_row=n_entities)


def create_bbox_image(image_size=256, bboxes=None):
    """
    Create an image with bounding boxes
    Args:
        image_size (tuple): Size of image (width, height)
        bboxes (list): List of normalized bounding boxes [[x1,y1,x2,y2],...]
    Returns:
        PIL.Image: Image with drawn bounding boxes
    """
    # Create blank white image
    image_size = (image_size, image_size)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    if bboxes is None:
        return image

    # Draw each bbox
    for bbox in bboxes:
        # Convert normalized coordinates to pixel coordinates
        x1 = bbox[0] * image_size[0]
        y1 = bbox[1] * image_size[1]
        x2 = bbox[2] * image_size[0]
        y2 = bbox[3] * image_size[1]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return image


def get_token_indices(pipeline, prompt, entity_list, adj_list=None):
    if isinstance(pipeline, StableDiffusionGSN) or isinstance(pipeline, StableDiffusion3PipelineGSN):
        token_indices = pipeline.get_attention_idx_from_token(prompt, entity_list)
        if adj_list is not None:
            adj_indices = pipeline.get_attention_idx_from_token(prompt, adj_list)
        else:
            adj_indices = None
        last_idx = pipeline.get_indice_last_token(prompt)
    else:
        raise NotImplementedError("Pipeline not supported for token indices extraction.")
    print(f"token indices: {token_indices}")
    print(f"adj indices: {adj_indices}")
    print(f"last idx: {last_idx}")
    return token_indices, adj_indices, last_idx


def format_syngen_adj_indices(criterion_gsng, adj_indices):
    if isinstance(criterion_gsng, SynGen) and adj_indices is not None:
        print(
            "The number of entities and the number of group of attributes must be the same. Align the entity without attribute with an empty list."
        )
        adj_indices_gsng = [[i] for i in adj_indices]
    else:
        adj_indices_gsng = None
    return adj_indices_gsng
