#
# Created on Wed Nov 20 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Training code for task Pressure Estimation
# ------------------------------
#
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable
import time
import numpy as np
import os
import math
from torch.nn import ReLU
from torch_geometric.nn import Linear

import torch.nn.functional as F
from gigantic_dataset.models.gnn_models import DownstreamTaskHead
from gigantic_dataset.utils.pretrain_utils import neighbor_degree_sum, compute_node_degree, RunningStatisticNodeDegrees, \
    add_random_edge_batched, compute_centrality_metrics
from torch_geometric.utils import degree, add_random_edge, batched_negative_sampling
from torch_geometric.utils import to_undirected
from torch_geometric.loader import RandomNodeLoader
from torch.nn.modules import Module
from torch.optim import Optimizer
from gigantic_dataset.utils.auxil_v8 import pretty_print
from gigantic_dataset.utils.train_utils import (
    generate_unique_name_from_config,
    print_metrics,
    save_checkpoint,
    log_metrics_on_wandb,
    apply_masks,
    print_single_metrics,
    wrapper_data_loader, prepare_finetune, prepare_finetune_dual_encoder,
)
from gigantic_dataset.utils.gen_random_mask_v8 import generate_batch_mask
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
import wandb
from gigantic_dataset.utils.train_protos import StartProfilerProto, ForwardProto, TrainOneEpochProto, TestOneEpochProto, \
    ConfigRef


class WandbStartProfiler(StartProfilerProto):
    def __call__(self, dataset_name: str = "", overriden_project_name: str = "", **kwargs: Any) -> tuple[str, str]:
        config = ConfigRef.config

        run_name, suffix, union_model_name = generate_unique_name_from_config(config, dataset_name)

        my_dict = config.as_dict()
        my_dict["run_name"] = run_name

        # start a new wandb run to track this script
        if config.log_method == "wandb":
            if wandb.run is not None:  # check if wandb is inititized
                wandb.finish(0, quiet=True)
            wandb.init(
                # set the wandb project where this run will be logged
                project=overriden_project_name if overriden_project_name != "" else config.project_name,
                name=run_name,
                # track hyperparameters and run metadata
                config=my_dict,
                settings=None,
            )
        if config.save_path != "":
            os.makedirs(config.save_path, exist_ok=True)

        pretty_print(my_dict)
        return run_name, suffix


# class SupervisedSingleForward(ForwardProto):
#     def __call__(
#             self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
#             **kwargs: Any
#     ) -> tuple[Any, Any, Any | None]:
#         assert data.x is not None and data.y is not None and isinstance(data.y, torch.Tensor)
#
#         out = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
#         y_pred = out
#         y_true = data.y
#         # exclude non-judging channels
#         if y_true.shape[-1] > 1 and take_first_channel:
#             y_true = y_true[..., 0]
#             y_true.unsqueeze_(dim=-1)
#             y_pred = y_pred[..., 0]
#             y_pred.unsqueeze_(dim=-1)
#
#         return (y_true, y_pred, out)


class SemiSingleForward(ForwardProto):
    def __call__(
            self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
            **kwargs: Any
    ) -> tuple[Any, Any, Any | None]:
        assert data.x is not None
        data.y = deepcopy(data.x)

        # data.x = data.x[:, :3]
        cols_to_zero = torch.tensor([0, 1])  # masked columns pressure and demand
        mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        data.x[mask_index[:, None], cols_to_zero] = 0
        # data.x[:, 4] = 0  # entire column neighborhood node degree zeroed

        data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce="mean")
        out, emb = models[0](x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)

        # out = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch,  # <----- ORIGINAL
        #                 edge_attr=data.edge_attr)

        y_pred = apply_masks(out, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore

        if y_true.shape[-1] > 1 and take_first_channel:
            y_true = y_true[..., 0]  # exclude non-judging channels
            y_true.unsqueeze_(dim=-1)
        return y_true, y_pred


class SemiSingleNNDegForward(ForwardProto):
    def __call__(
            self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
            **kwargs: Any
    ) -> tuple[Any, Any]:
        assert data.x is not None

        data.y = deepcopy(data.x[:, [0, 3, 4]])

        # if kwargs.get("is_inference", False):  # or kwargs.get("finetuning_dual", False):
        #     cols_to_mask = torch.tensor([0, 1])
        # else:
        cols_to_mask = torch.tensor([0, 1, 3])  # pressure, demand
        mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        data.x[mask_index[:, None], cols_to_mask] = 0

        # mask_degree = kwargs["batch_mask_degree"]
        # mask_degree_index = torch.nonzero(mask_degree, as_tuple=False).squeeze()
        # data.x[mask_degree_index[:, None], 3] = 0

        # data.x[:, 4] = 0  # all values for neighborhood node degree are zeroed

        data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce="mean")
        y_pred, emb = models[0](x=data.x[:, :4], edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

        y_pred = apply_masks(y_pred, [batch_mask])  # out[batch_mask] #type:ignore
        # neigh_degree_pred = apply_masks(neigh_degree, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore

        return y_pred, y_true  # deg_pred, neigh_degree_pred, y_true


class ForwardNNDegreeWOfeat(ForwardProto):
    def __call__(
            self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
            **kwargs: Any
    ) -> tuple[Any, Any, Any | None]:
        assert data.x is not None

        data.y = deepcopy(data.x[:, -2:])

        data.x[batch_mask, 0] = 0
        # data.x[:, 1] = 0  # all values for neighborhood node degree are zeroed

        # deg, neigh_degree = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch,
        #                               edge_attr=data.edge_attr)

        deg_pred, emb = models[0](x=data.x[:, 0], edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

        deg_pred = apply_masks(deg_pred, [batch_mask])  # out[batch_mask] #type:ignore
        # neigh_degree_pred = apply_masks(neigh_degree, [batch_mask])  # out[batch_mask] #type:ignore
        y_true = data.y[batch_mask]  # type:ignore

        return deg_pred, y_true  # deg_pred, neigh_degree_pred, y_true


# class MaskedSharedEncoder(ForwardProto):
#     def __call__(
#             self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
#             **kwargs: Any
#     ) -> tuple[Any, Any, Any, Any, Any, Any | None]:
#         assert data.x is not None
#
#         data.y = deepcopy(data.x)
#
#         cols_to_zero = torch.tensor([0, 1, 3])  # pressure, demand and node_degree
#         mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
#         data.x[mask_index[:, None], cols_to_zero] = 0
#         data.x[:, 4] = 0  # all values for neighborhood node degree are zeroed
#
#         out_pretrain, out_downstream = models[0](x=data.x, edge_index=data.edge_index, batch=data.batch,
#                                                  edge_attr=data.edge_attr)
#
#         y_pred_pressure = apply_masks(out_downstream, [batch_mask])  # out[batch_mask] #type:ignore
#         y_true_pressure = data.y[mask_index, 0]  # type:ignore
#
#         ret = (y_pred_pressure, y_true_pressure,)
#
#         if models[0].training:
#             y_pred_degree = apply_masks(out_pretrain[0], [batch_mask])  # out[batch_mask] #type:ignore
#             y_true_degree = data.y[mask_index, 3]  # type:ignore
#
#             y_pred_neigh_degree = apply_masks(out_pretrain[1], [batch_mask])  # out[batch_mask] #type:ignore
#             y_true_neigh_degree = data.y[mask_index, 4]  # type:ignore
#
#             ret = (y_pred_degree, y_true_degree, y_pred_neigh_degree, y_true_neigh_degree,) + ret
#
#         return ret


class MaskedDualEncoder(ForwardProto):
    def __call__(
            self, models: list[Module], data: Data, batch_mask: torch.Tensor, take_first_channel: bool = True,
            **kwargs: Any
    ) -> tuple[Any, Any, Any, Any, Any]:
        assert data.x is not None

        data.y = deepcopy(data.x[:, [0, 3, 4]])

        # if kwargs.get("is_inference", False):  # or kwargs.get("finetuning_dual", False):
        #     cols_to_mask = torch.tensor([0, 1])
        # else:
        cols_to_mask = torch.tensor([0, 1, 3])  # pressure, demand
        mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        data.x[mask_index[:, None], cols_to_mask] = 0

        # mask_degree = kwargs["batch_mask_degree"]
        # mask_degree_index = torch.nonzero(mask_degree, as_tuple=False).squeeze()
        # data.x[mask_degree_index[:, None], 3] = 0

        # data.x[:, 4] = 0  # all values for neighborhood node degree are zeroed

        data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce="mean")

        degrees, intermediate_pressure, pressure = models[0](x_struct=data.x[:, [3]], x_func=data.x[:, :3],
                                                             edge_index=data.edge_index,
                                                             # edge_index_undirected=data.edge_index_undirected,
                                                             batch=data.batch, edge_attr=data.edge_attr)

        deg_pred = apply_masks(degrees, [mask_index])  # out[batch_mask] #type:ignore
        deg_true = data.y[mask_index, -2:]

        # deg_pred = degrees
        # deg_true = data.y[:, -2:]

        int_pressure = apply_masks(intermediate_pressure, [batch_mask])  # out[batch_mask] #type:ignore
        y_pred_pressure = apply_masks(pressure, [batch_mask])  # out[batch_mask] #type:ignore
        y_true_pressure = data.y[mask_index, 0]  # type:ignore

        # if models[0].training:
        #     y_pred_degree = apply_masks(out_pretrain[0], [batch_mask])  # out[batch_mask] #type:ignore
        #     y_true_degree = data.y[mask_index, 3]  # type:ignore
        #
        #     y_pred_neigh_degree = apply_masks(out_pretrain[1], [batch_mask])  # out[batch_mask] #type:ignore
        #     y_true_neigh_degree = data.y[mask_index, 4]  # type:ignore
        #
        #     ret = ret + (y_pred_degree, y_true_degree, y_pred_neigh_degree, y_true_neigh_degree,)

        return deg_pred, deg_true, int_pressure, y_pred_pressure, y_true_pressure


# class TrainOneEpochSE(TrainOneEpochProto):
#     def __call__(
#             self,
#             models: list[Module],
#             optimizers: list[Optimizer],
#             loader: DataLoader,
#             criterion: Callable[..., Any],
#             metric_fn_dict: dict[str, Callable[..., Any]],
#             **kwargs: Any,
#     ) -> tuple[float, dict, Any]:
#         config = ConfigRef.config
#         func_ref = ConfigRef.ref
#         device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
#         mask_rate = config.mask_rate
#         use_data_batch = config.use_data_batch
#         for pt in models:
#             pt.train()
#             pt.to(device)
#         len_loader_dataset = len(loader.dataset)  # type:ignore
#         total_loss = 0
#         total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
#
#         epoch = kwargs.get("epoch", 0)
#         lambda_scheduler = kwargs.get("lambda_scheduler", None)
#         lambda_weight = lambda_scheduler.get_lambda(epoch=epoch - 1, total_epochs=config.epochs - 1, variant="linear")
#
#         # loader.dataset._which_subset = kwargs.get("which_subset", None)
#         for data in loader:  # Iterate in batches over the training dataset.
#             [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.
#
#             num_nodes = torch.unique(data.batch, return_counts=True)[1]
#             batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate,
#                                              required_mask=None)
#
#             non_blocking = False
#             data.x = data.x.to(device, non_blocking=non_blocking)
#             data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
#             data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None
#
#             data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
#             data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
#             data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
#
#             output = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
#             y_pred_degree = output[0]
#             y_true_degree = output[1]
#             y_pred_neigh_degree = output[2]
#             y_true_neigh_degree = output[3]
#             y_pred_pressure = output[4]
#             y_true_pressure = output[5]
#
#             degree_loss = criterion(y_pred_degree, y_true_degree)
#             neigh_degree_loss = criterion(y_pred_neigh_degree, y_true_neigh_degree)
#             pretraining_loss = degree_loss + neigh_degree_loss
#
#             pressure_loss = criterion(y_pred_pressure, y_true_pressure)
#             # tr_loss = pretraining_loss + pressure_loss
#             tr_loss = lambda_weight * pretraining_loss + (1 - lambda_weight) * pressure_loss
#
#             # print(f"lambda_weight: {lambda_weight} - {1 - lambda_weight}")
#
#             tr_loss.backward()  # Derive gradients.
#             [optimizer.step() for optimizer in optimizers]  # Update parameters based on gradients.
#
#             with torch.no_grad():
#                 total_loss += float(tr_loss) * data.num_graphs
#                 y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred_pressure, y_true_pressure,
#                                                                                **kwargs)
#
#                 output = normalizer.denormalize(normalized_output)
#                 target = normalizer.denormalize(normalized_target)
#
#                 for k, fn in metric_fn_dict.items():
#                     computed_metric = fn(y_pred_rescaled, y_true_rescaled)
#                     total_metric_dict[k] += computed_metric * data.num_graphs
#
#         with torch.no_grad():
#             dividend = max(1, len_loader_dataset)
#             metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
#             return total_loss / dividend, metric_dict
#
#
# class TestOneEpochSE(TestOneEpochProto):
#     def __call__(
#             self,
#             models: list[Module],
#             loader: DataLoader,
#             criterion: Callable[..., Any],
#             metric_fn_dict: dict[str, Callable[..., Any]],
#             **kwargs: Any,
#     ) -> tuple[float, dict, Any]:
#         config = ConfigRef.config
#         func_ref = ConfigRef.ref
#         device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
#         mask_rate = config.mask_rate
#         use_data_batch = config.use_data_batch
#
#         epoch = kwargs.get("epoch", 0)
#         # lambda_schedule = kwargs.get("lambda_schedule", None)
#         #
#         # lambda_weight = lambda_schedule(epoch) if lambda_schedule else 0.5
#
#         for pt in models:
#             pt.eval()
#             pt.to(device)
#         with torch.no_grad():
#             total_loss = 0
#             total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
#             len_loader_dataset = len(loader.dataset)  # type:ignore
#
#             # loader.dataset._which_subset = kwargs.get("which_subset", None)
#             for data in loader:
#                 # assert data.edge_index.max() < data.num_nodes
#
#                 num_nodes = torch.unique(data.batch, return_counts=True)[1]
#                 batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate,
#                                                  required_mask=None)
#
#                 non_blocking = False
#                 data.x = data.x.to(device, non_blocking=non_blocking)
#                 data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
#                 data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None
#
#                 data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
#                 data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
#                 data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
#
#                 output = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
#                 y_pred_pressure = output[0]
#                 y_true_pressure = output[1]
#
#                 # y_pred_degree = output[0]
#                 # y_true_degree = output[1]
#                 # y_pred_neigh_degree = output[2]
#                 # y_true_neigh_degree = output[3]
#                 # y_pred_pressure = output[4]
#                 # y_true_pressure = output[5]
#
#                 # degree_loss = criterion(y_pred_degree, y_true_degree)
#                 # neigh_degree_loss = criterion(y_pred_neigh_degree, y_true_neigh_degree)
#                 # pretraining_loss = degree_loss + neigh_degree_loss
#
#                 val_loss = criterion(y_pred_pressure, y_true_pressure)
#                 # val_loss = pretraining_loss + pressure_loss
#                 # val_loss = lambda_weight * pretraining_loss + (1 - lambda_weight) * pressure_loss
#
#                 # update metrics
#                 y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred_pressure, y_true_pressure,
#                                                                                **kwargs)
#
#                 total_loss += float(val_loss) * data.num_graphs
#                 for k, fn in metric_fn_dict.items():
#                     computed_metric = fn(y_pred_rescaled, y_true_rescaled)
#                     total_metric_dict[k] += computed_metric * data.num_graphs
#
#             dividend = max(1, len_loader_dataset)
#             metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
#             return total_loss / dividend, metric_dict

class TrainOneEpochDE(TrainOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            optimizers: list[Optimizer],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch
        for pt in models:
            pt.train()
            pt.to(device)
        len_loader_dataset = len(loader.dataset)  # type:ignore
        total_loss = 0
        total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}

        epoch = kwargs.get("epoch", 0)
        lambda_scheduler = kwargs.get("lambda_scheduler", None)
        lambda_scheduler_variant = kwargs.get("lambda_scheduler_variant", None)
        assert lambda_scheduler_variant is not None, "lambda_scheduler_variant is required"

        lambda_weight = lambda_scheduler.get_lambda(epoch=epoch - 1, total_epochs=config.epochs - 1, variant=lambda_scheduler_variant)

        scheduler = kwargs.get("scheduler", None)

        for batch_idx, data in enumerate(loader):  # Iterate in batches over the training dataset.
            [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.

            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))
                # print([optimizer.param_groups[0]['lr'] for optimizer in optimizers])

            num_nodes = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate, required_mask=None)

            # batch_mask_degree = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index,
            #                                         mask_rate=0.5,
            #                                         required_mask=None)

            non_blocking = False
            data.x = data.x.to(device, non_blocking=non_blocking)
            data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
            data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

            data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
            data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
            data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
            # data.edge_index_undirected = data.edge_index_undirected.to(device, non_blocking=non_blocking)

            deg_pred, deg_true, int_pressure, y_pred_pressure, y_true_pressure = func_ref.forward_fn(
                models=models,
                data=data,
                batch_mask=batch_mask,
                # batch_mask_degree=batch_mask_degree,
                **kwargs)

            deg_loss = criterion(deg_pred, deg_true)
            # consistency_loss = F.l1_loss(int_pressure, y_pred_pressure)
            pressure_loss = criterion(y_pred_pressure, y_true_pressure)

            # tr_loss = deg_loss + pressure_loss
            tr_loss = lambda_weight * deg_loss + (1 - lambda_weight) * pressure_loss

            tr_loss.backward()  # Derive gradients.
            [optimizer.step() for optimizer in optimizers]  # Update parameters based on gradients.

            with torch.no_grad():
                total_loss += float(tr_loss) * data.num_graphs
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred_pressure, y_true_pressure, **kwargs)

            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                total_metric_dict[k] += computed_metric * data.num_graphs

        with torch.no_grad():
            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict


class TestOneEpochDE(TestOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch

        # epoch = kwargs.get("epoch", 0)
        # lambda_schedule = kwargs.get("lambda_schedule", None)

        for pt in models:
            pt.eval()
            pt.to(device)
        with torch.no_grad():
            total_loss = 0
            total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
            len_loader_dataset = len(loader.dataset)  # type:ignore
            for data in loader:
                # assert data.edge_index.max() < data.num_nodes

                num_nodes = torch.unique(data.batch, return_counts=True)[1]
                batch_mask = generate_batch_mask(
                    num_nodes=num_nodes, edge_index=data.edge_index,
                    mask_rate=mask_rate, required_mask=None
                )

                # batch_mask_degree = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index,
                #                                         mask_rate=0.5,
                #                                         required_mask=None)

                non_blocking = False
                data.x = data.x.to(device, non_blocking=non_blocking)
                data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
                data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

                data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
                data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
                data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
                # data.edge_index_undirected = data.edge_index_undirected.to(device, non_blocking=non_blocking)

                deg_pred, deg_true, int_pressure, y_pred_pressure, y_true_pressure = func_ref.forward_fn(
                    models=models,
                    data=data,
                    batch_mask=batch_mask,
                    # batch_mask_degree=batch_mask_degree,
                    **kwargs)

                # deg_loss = criterion(deg_pred, deg_true)
                # consistency_loss = F.l1_loss(int_pressure, y_pred_pressure)
                # pressure_loss = criterion(y_pred_pressure, y_true_pressure)

                val_loss = criterion(y_pred_pressure, y_true_pressure)

                # val_loss = 0.5 * deg_loss + 0.5 * (int_pressure_loss + pressure_loss)
                # val_loss = deg_loss + consistency_loss + pressure_loss
                # val_loss = int_pressure_loss

                # val_loss = pretraining_loss + pressure_loss
                # val_loss = lambda_weight * pretraining_loss + (1 - lambda_weight) * pressure_loss

                # update metrics
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred_pressure, y_true_pressure,
                                                                               **kwargs)

                total_loss += float(val_loss) * data.num_graphs
                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
        return total_loss / dividend, metric_dict


class TrainOneEpochDegreeEncoder(TrainOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            optimizers: list[Optimizer],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch
        for pt in models:
            pt.train()
            pt.to(device)
        len_loader_dataset = len(loader.dataset)  # type:ignore
        total_loss = 0
        total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}

        # epoch = kwargs.get("epoch", 0)
        # lambda_scheduler = kwargs.get("lambda_scheduler", None)
        # lambda_weight = lambda_scheduler.get_lambda(epoch=epoch - 1, total_epochs=config.epochs - 1, variant="linear")

        # degree_normalizer = kwargs.get("degree_normalizer")

        for data in loader:  # Iterate in batches over the training dataset.
            [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.

            num_nodes = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate,
                                             required_mask=None)

            non_blocking = False
            data.x = data.x.to(device, non_blocking=non_blocking)
            data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
            data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

            data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
            data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
            data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)

            y_pred, y_true = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)

            tr_deg_loss = criterion(y_pred[:, 1:], y_true[:, 1:])
            tr_head_loss = criterion(y_pred[:, 0], y_true[:, 0])
            tr_loss = tr_deg_loss + tr_head_loss

            tr_loss.backward()  # Derive gradients.

            [optimizer.step() for optimizer in optimizers]  # Update parameters based on gradients.

            with torch.no_grad():
                total_loss += float(tr_loss) * data.num_graphs
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred[:, 0], y_true[:, 0], **kwargs)

                # y_pred_rescaled = degree_normalizer.denormalize(deg_pred)
                # y_true_rescaled = degree_normalizer.denormalize(y_true)

                # y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(
                #     torch.cat([deg_pred.unsqueeze(1), neigh_degree_pred.unsqueeze(1)], dim=1), y_true, **kwargs
                # )

                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

        with torch.no_grad():
            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict


class TestOneEpochDegreeEncoder(TestOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = kwargs.get("mask_rate", config.mask_rate)
        use_data_batch = config.use_data_batch

        for pt in models:
            pt.eval()
            pt.to(device)
        with torch.no_grad():
            total_loss = 0
            total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
            len_loader_dataset = len(loader.dataset)  # type:ignore

            # epoch = kwargs.get("epoch", 0)
            # lambda_scheduler = kwargs.get("lambda_scheduler", None)
            # lambda_weight = lambda_scheduler.get_lambda(epoch=epoch - 1, total_epochs=config.epochs - 1, variant="linear")

            # degree_normalizer = kwargs.get("degree_normalizer")

            for data in loader:
                # assert data.edge_index.max() < data.num_nodes

                num_nodes = torch.unique(data.batch, return_counts=True)[1]
                batch_mask = generate_batch_mask(
                    num_nodes=num_nodes, edge_index=data.edge_index,
                    mask_rate=mask_rate, required_mask=None
                )

                non_blocking = False
                data.x = data.x.to(device, non_blocking=non_blocking)
                data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
                data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

                data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
                data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
                data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)

                # kwargs["degree_stats"] = loader.dataset._node_degree_stats
                y_pred, y_true = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)
                val_loss = criterion(y_pred[:, 0], y_true[:, 0])

                # update metrics
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred[:, 0], y_true[:, 0], **kwargs)
                # y_pred_rescaled = degree_normalizer.denormalize(deg_pred)
                # y_true_rescaled = degree_normalizer.denormalize(y_true)

                # y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(
                #     torch.cat([deg_pred.unsqueeze(1), neigh_degree_pred.unsqueeze(1)], dim=1), y_true, **kwargs
                # )

                total_loss += float(val_loss) * data.num_graphs
                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict


class TrainOneEpoch(TrainOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            optimizers: list[Optimizer],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict,]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch
        for pt in models:
            pt.train()
            pt.to(device)
        len_loader_dataset = len(loader.dataset)  # type:ignore
        total_loss = 0
        total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}

        # loader.dataset._which_subset = kwargs.get("which_subset", None)
        for data in loader:  # Iterate in batches over the training dataset.
            [optimizer.zero_grad() for optimizer in optimizers]  # Clear gradients.

            num_nodes = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=mask_rate,
                                             required_mask=None)

            non_blocking = False
            data.x = data.x.to(device, non_blocking=non_blocking)
            data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
            data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

            data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
            data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
            data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
            # data.edge_index_undirected = data.edge_index_undirected.to(device, non_blocking=non_blocking)

            y_true, y_pred = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)

            tr_loss = criterion(y_pred, y_true)
            tr_loss.backward()  # Derive gradients.
            [optimizer.step() for optimizer in optimizers]  # Update parameters based on gradients.

            with torch.no_grad():
                total_loss += float(tr_loss) * data.num_graphs
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred, y_true, **kwargs)

                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

        with torch.no_grad():
            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
            return total_loss / dividend, metric_dict


class TestOneEpoch(TestOneEpochProto):
    def __call__(
            self,
            models: list[Module],
            loader: DataLoader,
            criterion: Callable[..., Any],
            metric_fn_dict: dict[str, Callable[..., Any]],
            **kwargs: Any,
    ) -> tuple[float, dict, list | None, list | None]:
        config = ConfigRef.config
        func_ref = ConfigRef.ref
        device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        mask_rate = config.mask_rate
        use_data_batch = config.use_data_batch
        is_inference = kwargs.get("is_inference", False)

        for pt in models:
            pt.eval()
            pt.to(device)
        with torch.no_grad():
            total_loss = 0
            total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
            len_loader_dataset = len(loader.dataset)  # type:ignore
            all_y_pred, all_y_true = [], []

            # loader.dataset._which_subset = kwargs.get("which_subset", None)
            for data in loader:
                # assert data.edge_index.max() < data.num_nodes

                num_nodes = torch.unique(data.batch, return_counts=True)[1]
                batch_mask = generate_batch_mask(
                    num_nodes=num_nodes, edge_index=data.edge_index,
                    mask_rate=mask_rate, required_mask=None
                )

                non_blocking = False
                data.x = data.x.to(device, non_blocking=non_blocking)
                data.y = data.y.to(device, non_blocking=non_blocking) if "y" in data else None
                data.edge_y = data.edge_y.to(device, non_blocking=non_blocking) if "edge_y" in data else None

                data.edge_attr = data.edge_attr.to(device, non_blocking=non_blocking) if "edge_attr" in data else None
                data.batch = data.batch.to(device, non_blocking=non_blocking) if use_data_batch else None
                data.edge_index = data.edge_index.to(device, non_blocking=non_blocking)
                # data.edge_index_undirected = data.edge_index_undirected.to(device, non_blocking=non_blocking)

                y_true, y_pred = func_ref.forward_fn(models=models, data=data, batch_mask=batch_mask, **kwargs)

                # if is_inference:
                #     all_y_true.extend(y_true)
                #     all_y_pred.extend(y_pred)

                val_loss = criterion(y_pred, y_true)
                # update metrics
                y_pred_rescaled, y_true_rescaled = func_ref.post_forward_tf_fn(y_pred, y_true, **kwargs)

                total_loss += float(val_loss) * data.num_graphs
                for k, fn in metric_fn_dict.items():
                    computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                    total_metric_dict[k] += computed_metric * data.num_graphs

            dividend = max(1, len_loader_dataset)
            metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}

            # ret = (total_loss / dividend, metric_dict,)
            # if is_inference:
            #     ret = ret + (all_y_pred, all_y_true,)
            #
            # return ret
            return total_loss / dividend, metric_dict


def train(
        models: list[torch.nn.Module],
        datasets: list[Dataset],
        train_metric_fn_dict: dict[str, Callable],
        val_metric_fn_dict: dict[str, Callable],
        **kwargs: Any,
) -> dict[str, Any]:
    sampling_strategy = kwargs.get("sampling_strategy", "batch")
    train_shuffle = kwargs.get("train_shuffle", True)
    finetuning = kwargs.get("finetuning", False)  # used only for fine tuning
    finetuning_dual = kwargs.get("finetuning_dual", False)  # used only for fine tuning
    config = ConfigRef.config
    func_ref = ConfigRef.ref

    # get default loaders
    train_loader = wrapper_data_loader(
        datasets[0], sampling_strategy=sampling_strategy, batch_size=config.batch_size, shuffle=train_shuffle,
        pin_memory=False
    )
    val_loader = wrapper_data_loader(datasets[1], sampling_strategy=sampling_strategy, batch_size=config.batch_size,
                                     shuffle=False, pin_memory=False)

    criterion = func_ref.load_criterion(**kwargs)
    optimizers = func_ref.load_optimizers(models=models, **kwargs)
    path = ""
    if finetuning:
        pass
        # stem = Linear(models[0].pressure_encoder.lin0.in_channels, models[0].pressure_encoder.lin0.out_channels)
        # head = Linear(models[0].pressure_encoder.lin1.in_channels, models[0].pressure_encoder.lin1.out_channels)
        # models[0].pressure_encoder.lin0 = stem
        # models[0].pressure_encoder.lin1 = head
        # optimizers[0] = prepare_finetune(model=models[0], lr_encoder=5e-5, lr_head=5e-4)

        # ### CONFIG SCHEDULER
        # kwargs["optimizer"] = optimizers[0]
        # kwargs["scheduler_variant"] = "CosineAnnealing"
        # kwargs["t_0"] = 10

    scheduler = func_ref.load_scheduler(**kwargs)
    kwargs["scheduler"] = scheduler

    # intial records
    best_loss = np.inf
    best_metric_dict = {}
    best_epoch = 0
    best_save_path = ""
    last_save_path = ""

    start_time = time.time()
    print("Start time:", datetime.fromtimestamp(start_time))
    print("*" * 80)

    # prev_lr = optimizers[0].param_groups[0]['lr']

    # degree_normalizer = RunningStatisticNodeDegrees()
    # kwargs["degree_normalizer"] = degree_normalizer

    for epoch in range(1, config.epochs + 1):

        # print(f"Training @epoch {epoch}...")
        kwargs["epoch"] = epoch
        tr_loss, tr_metric_dict = func_ref.train_one_epoch_fn(
            models=models,
            optimizers=optimizers,
            loader=train_loader,
            criterion=criterion,
            metric_fn_dict=train_metric_fn_dict,
            **kwargs,
        )

        # kwargs["which_subset"] = "validation"
        val_loss, val_metric_dict = func_ref.test_one_epoch_fn(
            models=models,
            loader=val_loader,
            criterion=criterion,
            metric_fn_dict=val_metric_fn_dict,
            **kwargs,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_metric_dict = val_metric_dict
            best_epoch = epoch
            # save training_checkpoint

            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict={i: optimizer.state_dict() if optimizer else None for i, optimizer in
                                       enumerate(optimizers)},
                epoch=best_epoch,
                loss=best_loss,
                val_metric_dict=best_metric_dict,
                norm_type=config.norm_type,
            )

            best_save_path = save_checkpoint(path=config.save_path, models=models, prefix="best",
                                             **save_kwargs)  # type:ignore

        if (epoch == 1 or (epoch % config.log_per_epoch) == 0) and not math.isnan(tr_loss):
            print_metrics(
                epoch=epoch,
                tr_loss=tr_loss,
                val_loss=val_loss,
                tr_metric_dict=tr_metric_dict,
                val_metric_dict=val_metric_dict,
            )
            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict={i: optimizer.state_dict() if optimizer else None for i, optimizer in
                                       enumerate(optimizers)},  # type:ignore
                epoch=epoch,  # type:ignore
                loss=tr_loss,  # type:ignore
                val_metric_dict=val_metric_dict,
                norm_type=config.norm_type,
            )
            last_save_path = save_checkpoint(path=config.save_path, models=models, prefix="last",
                                             **save_kwargs)  # type:ignore

        if config.log_method == "wandb":
            log_metrics_on_wandb(
                epoch=epoch,
                commit=True,
                train_loss=tr_loss,
                val_loss=val_loss,
                best_loss=best_loss,
                best_epoch=best_epoch,
                tr_metric_dict=tr_metric_dict,
                val_metric_dict=val_metric_dict,
            )

        if scheduler is not None:
            scheduler.step(val_loss)

    return {"best_save_path": best_save_path, "last_save_path": last_save_path}


def eval(
        models: list[torch.nn.Module],
        datasets: list[Dataset],
        test_metric_fn_dict: dict[str, Callable],
        **kwargs: Any,
) -> Any | None:
    config = ConfigRef.config
    func_ref = ConfigRef.ref

    test_loader = wrapper_data_loader(datasets[-1], sampling_strategy="batch", batch_size=config.batch_size,
                                      shuffle=False, pin_memory=False)

    assert isinstance(test_loader, DataLoader)

    # load reference
    criterion = func_ref.load_criterion(**kwargs)

    best_epoch = 0

    start_time = time.time()
    dt1 = datetime.fromtimestamp(start_time)
    print("Start time:", dt1)
    print("*" * 80)

    # kwargs["which_subset"] = "test"
    test_loss, test_metric_dict = func_ref.test_one_epoch_fn(
        models=models,
        loader=test_loader,
        criterion=criterion,
        metric_fn_dict=test_metric_fn_dict,
        config=config,
        **kwargs,
    )

    print_single_metrics(
        epoch=0,
        test_loss=test_loss,
        test_metric_dict=test_metric_dict,
    )

    if config.log_method == "wandb":
        log_metrics_on_wandb(
            epoch=0,
            commit=True,
            test_loss=test_loss,
            best_epoch=best_epoch,
            test_metric_dict=test_metric_dict,
        )

    end_time = time.time()
    dt2 = datetime.fromtimestamp(end_time)
    print("*" * 80)
    print("End time:", dt2)
    print("Executation time: ", dt2 - dt1)

    if config.log_method == "wandb":
        wandb.finish()
    return test_metric_dict
