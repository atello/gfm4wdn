#
# Created on Wed Jan 23 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Store mask generation algorithm
# Version: v8 - random
# Note: It is similar to v7- random, but we add a boolean flag whether we zero required masks from context (i.e., target mask turns off (0) all positions of context that was masked (1))
# ------------------------------
#

import torch
import math


def generate_batch_mask(
    num_nodes: list[int], mask_rate: float, required_mask: torch.Tensor | None, do_mask_required_mask=True, **kwargs
) -> torch.Tensor:
    """generate random mask per graphs

    Args:
        num_nodes (list[int]): num_nodes per graph. You can gather this by torch.unique(data.batch, return_counts=True)[1]
        mask_rate (float): masking rate. E.g., 0.95 = 95% nodes are masked per graph
        required_mask (torch.Tensor): Before generating mask, apply required_mask tensor (has shape of #num_nodes) to the node indices array
        do_mask_required_mask (bool, optional): In combination with required_mask. If true, zero-ing positions which are formed by required_mask

    Returns:
        torch.Tensor: batch_mask has shape of (num_nodes)
    """
    assert mask_rate > 0.0
    N = sum(num_nodes)
    # tensor_num_nodes has shape [batch_size]
    cumsum_num_nodes = torch.cumsum(num_nodes, dim=0)  # type:ignore
    lbs = torch.nn.functional.pad(cumsum_num_nodes[:-1], [1, 0])  # torch.cat([torch.zeros(size=[1]),cumsum_num_nodes[:-1]])
    ubs = cumsum_num_nodes
    remaining_nodes = torch.arange(N)

    if required_mask is not None:
        nonzero_inverted_required_mask = (~required_mask).nonzero()
        remaining_nodes = remaining_nodes[nonzero_inverted_required_mask]

    def extract_graphs(lb, ub):
        filter_mask = torch.logical_and(remaining_nodes >= lb, remaining_nodes < ub)
        segment = remaining_nodes[filter_mask]
        gap = segment.shape[0]
        mask_num = max(math.floor(gap * mask_rate), 1)
        keep_num = gap - mask_num
        permuted_idx = torch.randperm(segment.nelement())[:keep_num]
        permuted_node = segment[permuted_idx]

        return permuted_node

    keep_nodes = torch.hstack(tuple(map(extract_graphs, lbs, ubs)))

    mask = torch.ones(N, dtype=torch.bool)
    mask[keep_nodes] = 0

    if not do_mask_required_mask and required_mask is not None:
        mask[required_mask.nonzero()] = 0
    return mask
