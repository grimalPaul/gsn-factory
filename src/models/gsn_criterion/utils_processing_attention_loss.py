import itertools
from typing import List

import cv2
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
    GaussianSmoothing,
)


#### FROM INITNO ###
def get_clean_otsu_mask(attention_map, K=1):
    topk_coord_list, _ = fn_get_topk(attention_map, K=K)
    otsu_mask = fn_get_otsu_mask(attention_map)
    otsu_mask = fn_clean_mask(
        otsu_mask,
        topk_coord_list[0][0],
        topk_coord_list[0][1],
    )
    return otsu_mask


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.detach().view(H * W)
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []

    for index in topk_index:
        index = index.cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_coord_list, topk_value


def fn_get_otsu_mask(x):
    x_numpy = x
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opencv_threshold = opencv_threshold * 1.0 / 255.0

    otsu_mask = torch.where(
        x < opencv_threshold,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device),
    )

    return otsu_mask


def fn_clean_mask(otsu_mask, x, y):
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)

    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device),
    )

    return ret_otsu_mask


def get_topk_list(indices, cross_attention_maps, k=1):
    topk_coord_list_list = []
    for i in indices:
        cross_attention_map_cur_token = cross_attention_maps[i]
        topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=k)
        topk_coord_list_list.append(topk_coord_list)
    return topk_coord_list_list


def cross_attention_initno_loss(attention_maps, token_indices, k=1):
    clean_cross_attention_loss = 0.0

    topk_value_list, topk_coord_list_list = [], []
    for i in token_indices:
        cross_attention_map_cur_token = attention_maps[i]

        topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=k)

        topk_value = 0
        for coord_x, coord_y in topk_coord_list:
            topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
        topk_value = topk_value / k

        topk_value_list.append(topk_value)
        topk_coord_list_list.append(topk_coord_list)

        clean_cross_attention_map_cur_token = cross_attention_map_cur_token
        clean_cross_attention_map_cur_token_mask = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
        clean_cross_attention_map_cur_token_mask = fn_clean_mask(
            clean_cross_attention_map_cur_token_mask,
            topk_coord_list[0][0],
            topk_coord_list[0][1],
        )

        clean_cross_attention_map_cur_token_foreground = (
            clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask
            + (1 - clean_cross_attention_map_cur_token_mask)
        )
        clean_cross_attention_map_cur_token_background = clean_cross_attention_map_cur_token * (
            1 - clean_cross_attention_map_cur_token_mask
        )

        if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
            clean_cross_attention_loss = (
                clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max()
            )
        else:
            clean_cross_attention_loss = (
                clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max() * 0
            )

    cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
    return (max(cross_attn_loss_list), clean_cross_attention_loss), topk_coord_list_list


def initno_loss_self_attention(self_attention_maps, topk_coord_list_list, res_height, res_width, smooth_attention):
    self_attention_map_list = get_self_attention_maps_list(
        self_attention_maps,
        topk_coord_list_list,
        res_height,
        res_width,
        smooth_attention,
    )
    self_attn_loss, number_self_attn_loss_pair = 0, 0
    number_token = len(self_attention_map_list)
    for i in range(number_token):
        for j in range(i + 1, number_token):
            number_self_attn_loss_pair = number_self_attn_loss_pair + 1
            self_attention_map_1 = self_attention_map_list[i]
            self_attention_map_2 = self_attention_map_list[j]

            self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2)
            self_attention_map_sum = self_attention_map_1 + self_attention_map_2
            cur_self_attn_loss = self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6)
            self_attn_loss = self_attn_loss + cur_self_attn_loss

    if number_self_attn_loss_pair > 0:
        self_attn_loss = self_attn_loss / number_self_attn_loss_pair
    return self_attn_loss, None


def initno_alignment_loss(attention_maps, attention_maps_cache, token_indices, smooth_attention):
    cross_attn_alignment_loss = 0
    for i in token_indices:
        cross_attention_map_cur_token = attention_maps[i]
        cross_attention_map_cur_token_cache = attention_maps_cache[i]
        if smooth_attention:
            cross_attention_map_cur_token = smooth_attention_map_single(cross_attention_map_cur_token)

        cross_attn_alignment_loss = cross_attn_alignment_loss + torch.nn.L1Loss()(
            cross_attention_map_cur_token,
            cross_attention_map_cur_token_cache,
        )
    return cross_attn_alignment_loss, None


def get_self_attention_maps_list(self_attention_maps, topk_coord_list_list, res_height, res_width, smooth_attention):
    self_attention_map_list = []
    for topk_coord_list in topk_coord_list_list:
        self_attention_map_cur_token_list = []
        for coord_x, coord_y in topk_coord_list:
            self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_cur_token = self_attention_map_cur_token.view(res_height, res_width).contiguous()
            self_attention_map_cur_token_list.append(self_attention_map_cur_token)

        if len(self_attention_map_cur_token_list) > 0:
            self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(
                self_attention_map_cur_token_list
            )
            if smooth_attention:
                self_attention_map_cur_token = smooth_attention_map_single(self_attention_map_cur_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(res_height, res_width).contiguous()

        self_attention_map_list.append(self_attention_map_cur_token)
    return self_attention_map_list


### ATTEND AND EXCITE LOSS ###
def attend_and_excite_loss(attention_maps, token_indices):
    max_indices_list = []

    for i in token_indices:
        max_indices_list.append(attention_maps[i].max())
    losses = [max(0, 1.0 - curr_max) for curr_max in max_indices_list]
    return max(losses), None


def batch_attend_and_excite_loss(attention_maps: torch.Tensor, token_indices: List[int]):
    """
    Simplified batched version of attend_and_excite_loss.
    attention_maps: (B, H, W, N) where N is the number of tokens in token_indices.
    token_indices: List of token indices (not used here, but kept for compatibility).
    """
    # Flatten spatial dimensions and compute max attention per token
    max_attention_per_token = (
        attention_maps.view(attention_maps.shape[0], -1, attention_maps.shape[-1]).max(dim=1).values
    )

    # Find the minimum of the max attentions for each batch
    min_of_max_attentions = max_attention_per_token.min(dim=1).values

    # Loss is max(0, 1 - min_of_max_attentions)
    losses = torch.clamp(1.0 - min_of_max_attentions, min=0)

    return losses, None


### IOU loss ###
def iou(attention_map1, attentionmap2):
    """Compute the intersection over union between two attention maps.
    return value between 0 and 1
    Maximum overlap when value is 1
    Minimum overlap when value is 0
    """
    intersection = torch.min(attention_map1, attentionmap2).sum()
    union = (attention_map1 + attentionmap2).sum()
    return intersection / union


def iou_loss(attention_maps, token_indices):
    if len(token_indices) < 2:
        return 0, None

    ai_aj = list(itertools.combinations(token_indices, 2))
    loss_iou = []
    for ai, aj in ai_aj:
        loss_temp = iou(
            attention_maps[ai],
            attention_maps[aj],
        )
        loss_iou.append(loss_temp)
    return sum(loss_iou) / len(loss_iou), None


def batch_iou_loss(attention_maps: torch.Tensor, token_indices: List[int]):
    """
    Simplified batched version of iou_loss.
    attention_maps: (B, H, W, N) where N is the number of tokens in token_indices.
    token_indices: List of token indices (not used here, but kept for compatibility).
    """
    B, H, W, N = attention_maps.shape

    # Compute pairwise IoU for all token combinations
    iou_scores = []
    for i in range(N):
        for j in range(i + 1, N):
            intersection = torch.min(attention_maps[:, :, :, i], attention_maps[:, :, :, j]).sum(dim=[1, 2])
            union = (attention_maps[:, :, :, i] + attention_maps[:, :, :, j]).sum(dim=[1, 2])
            iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
            iou_scores.append(iou)

    # Average IoU scores across all pairs
    iou_scores = torch.stack(iou_scores, dim=0).mean(dim=0)

    return iou_scores, None


##### Smoothing attention maps ####


def batch_attention_maps_smoothing(attention_maps: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian smoothing to a batch of attention maps.
    :param attention_maps: A tensor of shape (B, H, W, N) where B is the batch size,
                           N is the number of tokens, and H, W are the map dimensions.
    :return: Smoothed attention maps of the same shape.
    """
    B, H, W, N = attention_maps.shape
    if N == 0:
        return attention_maps

    # Permute to (B, N, H, W) for Conv2d
    attention_maps_permuted = attention_maps.permute(0, 3, 1, 2)

    smoothing = GaussianSmoothing(channels=N).to(attention_maps.device)
    padded_maps = F.pad(attention_maps_permuted, (1, 1, 1, 1), mode="reflect")
    smoothed_maps_permuted = smoothing(padded_maps)

    # Permute back to (B, H, W, N)
    smoothed_maps = smoothed_maps_permuted.permute(0, 2, 3, 1)

    return smoothed_maps


def smooth_attention_map_single(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map


### Retention and Boxes ###
# boxdiff https://github.com/showlab/BoxDiff/blob/b0d5d3b534418aa3fc71b9a16e5b575c0b2ee3b6/pipeline/gligen_pipeline_boxdiff.py#L232
def constraint_attention_with_mask(attention_maps, token_indices, masks, p):
    max_indices_list_fg = []
    max_indices_list_bg = []

    for i in token_indices:
        # innerbox constraint
        k = (masks[i].sum() * p).long()
        max_indices_list_fg.append((attention_maps[i] * masks[i]).reshape(-1).topk(k)[0].mean())
        # outer-box constraint
        bg_mask = 1 - masks[i]
        k = (bg_mask.sum() * p).long()
        max_indices_list_bg.append((attention_maps[i] * bg_mask).reshape(-1).topk(k)[0].mean())
        # no corner constraint here
    losses_fg = [max(0, 1.0 - curr_max) for curr_max in max_indices_list_fg]
    losses_bg = [max(0, curr_max) for curr_max in max_indices_list_bg]
    return sum(losses_fg) + sum(losses_bg), None


def boxdiff_compute(attention_maps, token_indices, masks, bboxes, p, l):
    max_indices_list_fg = []
    max_indices_list_bg = []
    dist_x = []
    dist_y = []
    for i in token_indices:
        # innerbox constraint
        box = bboxes[i]
        print(f"box: {box}")
        k = (masks[i].sum() * p).long()
        max_indices_list_fg.append((attention_maps[i] * masks[i]).reshape(-1).topk(k)[0].mean())
        # outer-box constraint
        bg_mask = 1 - masks[i]
        k = (bg_mask.sum() * p).long()
        max_indices_list_bg.append((attention_maps[i] * bg_mask).reshape(-1).topk(k)[0].mean())
        # corner constraint
        gt_proj_x = torch.max(masks[i], dim=0)[0]
        gt_proj_y = torch.max(masks[i], dim=1)[0]
        corner_mask_x = torch.zeros_like(gt_proj_x)
        corner_mask_y = torch.zeros_like(gt_proj_y)

        # create gt according to the number L
        N = gt_proj_x.shape[0]
        corner_mask_x[max(box[0] - l, 0) : min(box[0] + l + 1, N)] = 1.0
        corner_mask_x[max(box[2] - l, 0) : min(box[2] + l + 1, N)] = 1.0
        corner_mask_y[max(box[1] - l, 0) : min(box[1] + l + 1, N)] = 1.0
        corner_mask_y[max(box[3] - l, 0) : min(box[3] + l + 1, N)] = 1.0
        dist_x.append((F.l1_loss(attention_maps[i].max(dim=0)[0], gt_proj_x, reduction="none") * corner_mask_x).mean())
        dist_y.append((F.l1_loss(attention_maps[i].max(dim=1)[0], gt_proj_y, reduction="none") * corner_mask_y).mean())

    return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y


def compute_loss_boxdiff_gsng(attention_maps, token_indices, masks, bboxes, p, l):
    max_indices_list_fg, max_indices_list_bg, dist_x, dist_y = boxdiff_compute(
        attention_maps, token_indices, masks, bboxes, p, l
    )
    losses_fg = [max(0, 1.0 - curr_max) for curr_max in max_indices_list_fg]
    losses_bg = [max(0, curr_max) for curr_max in max_indices_list_bg]
    return sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y), None


def compute_boxdiff_iteref(attention_maps, token_indices, masks, bboxes, p, l):
    max_indices_list_fg, _, _, _ = boxdiff_compute(attention_maps, token_indices, masks, bboxes, p, l)
    losses_fg = [max(0, 1.0 - curr_max) for curr_max in max_indices_list_fg]
    return max(losses_fg), None


# iou on mask
def iou_mask(attentions_maps, token_indices, masks):
    loss_iou = []
    for i in token_indices:
        loss_iou.append(1 - iou(attentions_maps[i], masks[i]))
    return sum(loss_iou) / len(loss_iou), None


def kl_divergence_mask(attention_maps, token_indices, masks):
    # positive loss attention_loss with mask
    pos_loss = []
    for i in token_indices:
        pos_loss.append(
            _symmetric_kl(
                attention_maps[i],
                torch.nn.functional.softmax(masks[i].reshape(-1), dim=0),
            )
        )
    pos_loss = sum(pos_loss) / len(token_indices)
    print(f"positive_loss: {pos_loss}")
    return pos_loss, None
    # return pos_loss, None
    neg_loss = []
    # negative loss between attention masks
    ai_aj = list(itertools.combinations(token_indices, 2))
    for ai, aj in ai_aj:
        neg_loss.append(_symmetric_kl(attention_maps[ai], attention_maps[aj]))
    neg_loss = -sum(neg_loss) / len(ai_aj)
    # neg_loss = 0
    print(f"negative_loss: {neg_loss}")
    loss = (neg_loss + pos_loss) / 2
    return loss, None


### SYNGEN ###
def _symmetric_kl(attention_map1, attention_map2):
    """The more the value is big, the more the two attention maps are different"""

    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence


def syngen_loss(attention_maps, subtrees_indices, all_indices, **kwargs):
    loss = 0.0

    for subtree_indices in subtrees_indices:
        # logger.info(f"Subtree indices: {subtree_indices}")
        noun, modifier = split_indices(subtree_indices)
        # logger.info(f"Subtree indices: {subtree_indices}")

        all_subtree_pairs = list(itertools.product(noun, modifier))
        if noun and not modifier:
            if isinstance(noun, list) and len(noun) == 1:
                processed_noun = noun[0]
            else:
                processed_noun = noun

            neg_loss = calculate_negative_loss(
                attention_maps,
                modifier,
                processed_noun,
                subtree_indices,
                all_indices,
            )
            # logger.info(f"Negative loss: {neg_loss}")
            loss += neg_loss

        else:
            positive_loss, negative_loss = _calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                all_indices,
            )

            loss += positive_loss
            loss += negative_loss

    return loss, None


def split_indices(related_indices: List[int]):
    noun = [related_indices[-1]]  # assumes noun is always last in the list
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier


def calculate_negative_loss(attention_maps, modifier, noun, subtree_indices, all_indices):
    outside_indices = _get_outside_indices(subtree_indices, all_indices)

    negative_noun_loss, num_noun_pairs = _calculate_outside_loss(attention_maps, noun, outside_indices)
    if outside_indices:
        negative_noun_loss = -sum(negative_noun_loss) / len(outside_indices)
    else:
        negative_noun_loss = 0

    if modifier:
        negative_modifier_loss, num_modifier_pairs = _calculate_outside_loss(attention_maps, modifier, outside_indices)
        if outside_indices:
            negative_modifier_loss = -sum(negative_modifier_loss) / len(outside_indices)
        else:
            negative_modifier_loss = 0

        negative_loss = (negative_modifier_loss + negative_noun_loss) / 2
    else:
        negative_loss = negative_noun_loss

    return negative_loss


def calculate_positive_loss(attention_maps, modifier, noun):
    src_indices = modifier
    dest_indices = noun

    if isinstance(src_indices, list) and isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[s], attention_maps[d])
            for (s, d) in itertools.product(src_indices, dest_indices)
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(dest_indices, list):
        wp_pos_loss = [_symmetric_kl(attention_maps[src_indices], attention_maps[d]) for d in dest_indices]
        positive_loss = max(wp_pos_loss)
    elif isinstance(src_indices, list):
        wp_pos_loss = [_symmetric_kl(attention_maps[s], attention_maps[dest_indices]) for s in src_indices]
        positive_loss = max(wp_pos_loss)
    else:
        positive_loss = _symmetric_kl(attention_maps[src_indices], attention_maps[dest_indices])

    return positive_loss


def _calculate_outside_loss(attention_maps, src_indices, outside_loss):
    negative_loss = []
    computed_pairs = set()
    pair_counter = 0

    for outside_idx in outside_loss:
        if isinstance(src_indices, list):
            wp_neg_loss = []
            for t in src_indices:
                pair_key = (t, outside_idx)
                if pair_key not in computed_pairs:
                    wp_neg_loss.append(_symmetric_kl(attention_maps[t], attention_maps[outside_idx]))
                    computed_pairs.add(pair_key)
            negative_loss.append(max(wp_neg_loss) if wp_neg_loss else 0)
            pair_counter += 1

        else:
            pair_key = (src_indices, outside_idx)
            if pair_key not in computed_pairs:
                negative_loss.append(_symmetric_kl(attention_maps[src_indices], attention_maps[outside_idx]))
                computed_pairs.add(pair_key)
                pair_counter += 1

    return negative_loss, pair_counter


def _get_outside_indices(subtree_indices, all_indices):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [map_idx for map_idx in all_indices if (map_idx not in flattened_subtree_indices)]
    return outside_indices


def _flatten_indices(related_indices):
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices


def _calculate_losses(
    attention_maps,
    all_subtree_pairs,
    subtree_indices,
    all_indices,
):
    positive_loss = []
    negative_loss = []
    for pair in all_subtree_pairs:
        noun, modifier = pair
        positive_loss.append(calculate_positive_loss(attention_maps, modifier, noun))
        negative_loss.append(calculate_negative_loss(attention_maps, modifier, noun, subtree_indices, all_indices))

    positive_loss = sum(positive_loss)
    negative_loss = sum(negative_loss)

    return positive_loss, negative_loss
