import math

import networkx as nx
import torch
from typing import Any, Literal, Optional, Tuple, Union, List

from torch import Tensor
from torch_geometric.utils import to_undirected, degree, add_random_edge, batched_negative_sampling, one_hot


def compute_node_degree(edge_index: torch.Tensor, num_nodes: int) -> Tensor:
    deg = degree(edge_index[0], num_nodes)
    neigh_degree = neighbor_degree_sum(edge_index, deg).reshape(-1, 1)

    # graph = nx.Graph()
    # graph.add_nodes_from(range(num_nodes))
    #
    # edges = [(i.item(), j.item()) for i, j in zip(edge_index[0], edge_index[1])]
    # graph.add_edges_from(edges)
    #
    # # closeness_centrality = torch.as_tensor([v for k, v in nx.closeness_centrality(graph).items()]).reshape(-1, 1)
    # betweenness_centrality = torch.as_tensor([v for k, v in nx.betweenness_centrality(graph, weight="roughness").items()]).reshape(-1, 1)
    # # clustering_coef = torch.as_tensor([v for k, v in nx.clustering(graph).items()])
    # pagerank = torch.as_tensor([v for k, v in nx.pagerank(graph, weight="roughness").items()]).reshape(-1, 1)
    #
    # ret = torch.cat([deg.reshape(-1, 1), neigh_degree, pagerank, betweenness_centrality], dim=-1)
    ret = torch.cat([deg.reshape(-1, 1), neigh_degree], dim=-1)
    return ret


def compute_centrality_metrics(edge_index: torch.Tensor, edge_attr: torch.Tensor,  num_nodes: int) -> Tensor:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    minval = edge_attr.min(dim=0)[0]
    max_val = edge_attr.max(dim=0)[0]
    edge_attr = (edge_attr - minval) / (max_val - minval)
    edge_attr = torch.floor(edge_attr * 100)

    edges = [(i.item(), j.item()) for i, j in zip(edge_index[0], edge_index[1])]
    edge_attr = {edge: val for edge, val in zip(edges, edge_attr[:, 2].tolist())}

    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph, edge_attr, "roughness")

    # closeness_centrality = torch.as_tensor([v for k, v in nx.closeness_centrality(graph).items()]).reshape(-1, 1)
    betweenness_centrality = torch.as_tensor([v for k, v in nx.betweenness_centrality(graph, weight="roughness").items()]).reshape(-1, 1)
    # clustering_coef = torch.as_tensor([v for k, v in nx.clustering(graph).items()])
    pagerank = torch.as_tensor([v for k, v in nx.pagerank(graph, weight="roughness").items()]).reshape(-1, 1)

    ret = torch.cat([pagerank, betweenness_centrality], dim=-1)
    return ret


def neighbor_degree_sum(edge_index, degrees):
    neighbor_deg_sum = torch.zeros_like(degrees)
    neighbor_deg_sum.scatter_add_(0, edge_index[0], degrees[edge_index[1]])

    return neighbor_deg_sum


def compute_node_degree_stats(degrees: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    degrees = torch.as_tensor(degrees)
    degree_stats = {
        "mean": degrees.mean(dim=0),
        "std": degrees.std(dim=0),
        "min": degrees.min(dim=0)[0],
        "max": degrees.max(dim=0)[0],
    }
    return degree_stats


class LambdaScheduler:
    def __init__(self, max_lambda=1.0, min_lambda=0.0, decay_rate=0.02):
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.decay_rate = decay_rate

    def get_lambda(self, epoch: int, total_epochs: int = 1000, variant: Literal["linear", "exp", "cos"] = "linear"):
        if variant == "linear":
            return max(0.1, 1 - epoch / total_epochs)  # gradually shift focus from pretraining to pressure
        elif variant == "exp":
            lambda_param = self.max_lambda * math.exp(-self.decay_rate * epoch)
            return max(self.min_lambda, lambda_param)
        elif variant == "cos":
            return (self.min_lambda +
                    0.5 * (self.max_lambda - self.min_lambda) * (1 + math.cos(math.pi * epoch / total_epochs)))


class DegreeNormalizer:
    def __init__(self, stats: dict[str, Tensor], device):
        self.mean = stats["mean"].to(device)
        self.std = stats["std"].to(device)
        self.min = stats["min"].to(device)
        self.max = stats["max"].to(device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class RunningStatisticNodeDegrees:
    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0.0
        self.eps = eps

    def update(self, x, training: bool = True):
        if training:
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(unbiased=False, dim=0, keepdim=True)
            batch_count = x.numel()

            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            new_var = M2 / total_count

            self.mean = new_mean
            self.var = new_var
            self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (self.var ** 0.5 + self.eps)

    def denormalize(self, x):
        return x * (self.var ** 0.5 + self.eps) + self.mean


def add_random_edge_batched(
        edge_index,
        batch: Union[Tensor, Tuple[Tensor, Tensor]],
        p: float = 0.5,
        force_undirected: bool = False,
        # num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
        training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float): Ratio of added edges to the existing edges.
            (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:

        >>> # Standard case
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0. or p > 1.:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 "
                         f"(got '{p}')")
    # if force_undirected and isinstance(num_nodes, (tuple, list)):
    #     raise RuntimeError("'force_undirected' is not supported for "
    #                        "bipartite graphs")

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    num_neg_samples = round(edge_index.size(1) * p)
    edge_index_to_add = batched_negative_sampling(
        edge_index=edge_index,
        batch=batch,
        num_neg_samples=num_neg_samples,
        force_undirected=force_undirected,
    )
    idx = torch.randperm(edge_index_to_add.size(1))[:num_neg_samples]
    edge_index_to_add = edge_index_to_add[:, idx]

    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

    return edge_index, edge_index_to_add


def add_random_edge_batched(
        edge_index,
        batch: Union[Tensor, Tuple[Tensor, Tensor]],
        p: float = 0.5,
        force_undirected: bool = False,
        # num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
        training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float): Ratio of added edges to the existing edges.
            (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:

        >>> # Standard case
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0. or p > 1.:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 "
                         f"(got '{p}')")
    # if force_undirected and isinstance(num_nodes, (tuple, list)):
    #     raise RuntimeError("'force_undirected' is not supported for "
    #                        "bipartite graphs")

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    num_neg_samples = round(edge_index.size(1) * p)
    edge_index_to_add = batched_negative_sampling(
        edge_index=edge_index,
        batch=batch,
        num_neg_samples=num_neg_samples,
        force_undirected=force_undirected,
    )
    idx = torch.randperm(edge_index_to_add.size(1))[:num_neg_samples]
    edge_index_to_add = edge_index_to_add[:, idx]

    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

    return edge_index, edge_index_to_add
