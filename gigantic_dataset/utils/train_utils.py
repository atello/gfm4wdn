#
# Created on Wed Nov 20 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: store utils function for TRAINING ONLY!
# Required: torch and wandb
# ------------------------------
#
import math
from collections import OrderedDict

import torch
from copy import deepcopy
from typing import Any, Callable, Iterable, Literal, Optional, Sequence
from torch import save, load, cat, Tensor, abs, subtract, mean, sum, ravel, div, sqrt, clamp, divide, as_tensor
from torch import nn
import torch.nn.functional as F
from functools import partial
import wandb
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from gigantic_dataset.utils.configs import TrainConfig
from datetime import datetime

from torch_geometric.data import Dataset, Batch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader as GraphDataLoader

from torch.utils.data import DataLoader as InstanceDataLoader, TensorDataset
import sys

import re
from torch import arange

import os
import os.path as osp
import glob

import numpy as np
from scipy.stats import wilcoxon, ttest_rel


def find_latest_files(folder_path: str, pattern: str) -> list[str]:
    # construct the full search pattern
    search_pattern = os.path.join(folder_path, pattern)

    # search for matching files
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        return []

    # sort files by modification time (latest first)
    sorted_files = sorted(matching_files, key=osp.getmtime, reverse=True)

    return sorted_files


def check_if_exist_keys_in_checkpoint(path: str, key: str) -> bool:
    """check if keys are existed in checkpoint path. Note that this function return False if checkpoint path is invalid.

    Args:
        path (str): checkpoint path .pt
        key (str): query key

    Returns:
        bool: result if key exists in a valid checkpoint file
    """
    assert path[-4:] == ".pth" or path[-3:] == ".pt"
    try:
        print(f"Try to load path ({path})...")
        cp_dict = load(path)
    except FileNotFoundError:
        print(f"ERROR! Try to load path ({path})...unsuccessfully!")
        return False

    return key in cp_dict


def load_custom_checkpoint(path: str, keys: list[str]) -> dict:
    """load requesting keys from checkpoint .pt file. If a key is non-existed in checkpoint, we simply ignore it.

    Args:
        path (str): checkpoint path .pt
        keys (list[str]): list of querying keys

    Returns:
        dict: return a dict containing EXISITNG keys and their corresponding values
    """
    assert path[-4:] == ".pth" or path[-3:] == ".pt"
    try:
        print(f"Try to load path ({path})...")
        cp_dict = load(path)
        return {k: cp_dict[k] for k in keys if k in cp_dict}
    except FileNotFoundError:
        print(f"ERROR! Try to load path ({path})...unsuccessfully!")
        return {}
    except KeyError:
        print(f"ERROR! Loaded dict has no required keys. Input keys are {keys}")
        return {}


def load_weights(path: str, models: list[nn.Module], load_keys: list[str]) -> tuple[list[nn.Module], dict]:
    """support load multiple models and relevant data

    Args:
        path (str): checkpoint file
        models (list[torch.nn.Module]): ORDER-SENSITIVE.  model architectures to load weights into
        load_keys (list[str]): ORDER-SENSITIVE. key list indicate the corresponding model weights.

    Returns:
        tuple[list[torch.nn.Module], dict]: tuple of list of loaded model and relevant data as dict
    """
    assert path[-4:] == ".pth" or path[-3:] == ".pt"
    assert models is not None
    assert len(load_keys) == len(models)
    try:
        print(f"Try to load path ({path})...")
        cp_dict = load(path)
    except Exception as e:
        print(f"ERROR! Try to load path ({path})...unsuccessfully!")
        return models, {}

    for model, load_key in zip(models, load_keys):
        name = model.name if hasattr(model, "name") else type(model).__name__
        try:
            print(f"Try to load weights of model ({name}) with key ({load_key}) at path ({path})...")
            new_state_dict = OrderedDict()

            # print(f"LOADED_MODEL: {cp_dict[load_key].keys()}")
            # print(f"NEW_MODEL: {model.state_dict().keys()}")

            if cp_dict[load_key].keys() != model.state_dict().keys():
                for key, value in cp_dict[load_key].items():
                    if "blocks" in key and "lin.weight" in key:
                        # Replace 'lin.weight' with both 'lin_src.weight' and 'lin_dst.weight'
                        key_src = key.replace("lin.weight", "lin_src.weight")
                        key_dst = key.replace("lin.weight", "lin_dst.weight")
                        new_state_dict[key_src] = value.clone()
                        new_state_dict[key_dst] = value.clone()
                    if "presssure_encoder" in key:
                        new_key = key.replace("presssure_encoder", "pressure_encoder")
                        new_state_dict[new_key] = value.clone()
                    else:
                        new_state_dict[key] = value
                cp_dict[load_key] = new_state_dict
                # print(F"CONVERTED_KEYS: {new_state_dict.keys()}")
            model.load_state_dict(cp_dict[load_key])
            print(f"Succesfully load model!")
        except KeyError as e:
            # print(f"WARN! Loaded key ({load_key}) of model ({name}) is not existed in path ({path})! Use NEW MODEL INSTEAD! Log: {e}")
            print(
                f"Try to load weights of model ({name}) with key ({load_key}) at path ({path})...fail! Key is unfound! Use NEW MODEL INSTEAD! Log: {e}"
            )
        except RuntimeError as e:
            # print(f"WARN! Found loaded key ({load_key}) but doesnt fit model ({name})!  Use NEW MODEL INSTEAD! Log: {e}")
            print(
                f"Try to load weights of model ({name}) with key ({load_key}) at path ({path})...fail! Model doesn't fit weights! Use NEW MODEL INSTEAD! Log: {e}"
            )

    return models, cp_dict


def save_checkpoint(path: str, models: list[nn.Module], prefix: str, **kwargs: Any) -> str:
    """Support save checkpoint. User can leverage kwargs to store model and relevant data
    Proxy of torch.save
    Args:
        path (str): saved path

    Returns:
        str: saved path
    """

    # create a folder
    os.makedirs(path, exist_ok=True)

    # save training stuff
    training_path = osp.join(path, f"{prefix}_training_log.pt")
    save(kwargs, training_path)

    # save models
    for i, model in enumerate(models):
        model_path = osp.join(path, f"{prefix}_{model.name}_{i}.pt")
        model_weight = {getattr(model, "name"): model.state_dict()}
        save(model_weight, model_path)
    # save(kwargs, path)
    return path


def get_criterion(loss_name: str, device: str) -> Callable:
    """get loss by name"""
    if loss_name == "sce":

        def sce_loss(x, y, alpha=3):
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
            return loss.mean()

        criterion = sce_loss
    elif loss_name == "mse":
        criterion = nn.MSELoss(reduction="mean").to(device)
    elif loss_name == "mae":
        criterion = nn.L1Loss(reduction="mean").to(device)
    elif loss_name == "ce":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise KeyError(f"criterion {loss_name} is not supported")

    return criterion


def calculate_classified_accuracy(y_pred: Tensor, y_true: Tensor):
    if len(y_true.size()) > 1:
        y_true = y_true.max(1)[1].long()  # .squeeze().long()

    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def calculate_nse(y_pred, y_true, exponent=2):
    raveled_y_pred = ravel(y_pred)
    raveled_y_true = ravel(y_true)
    return 1.0 - div(
        sum(pow(raveled_y_pred - raveled_y_true, exponent)),
        sum(pow(raveled_y_true - mean(raveled_y_true), exponent)) + 1e-12,
    )


def calculate_rmse(y_pred, y_true):
    return sqrt(mean((y_pred - y_true) ** 2))


def calculate_rel_error(y_pred, y_true, epsilon: float = 1e-6):
    err = abs(subtract(y_true, y_pred))
    mask = abs(y_true) > 0.01
    rel_err = abs(divide(err[mask], y_true[mask] + epsilon))
    return mean(rel_err)


def calculate_rel_error2(y_pred, y_true, epsilon: float = 1e-4):
    flat_pred = y_pred.reshape([-1, 1])
    flat_true = y_true.reshape([-1, 1])
    err = abs(subtract(flat_true, flat_pred))
    mask = abs(flat_true) > 0.01
    rel_err = abs(divide(err[mask], flat_true[mask] + epsilon))
    return mean(rel_err)


def calculate_mape_per_node(y_pred, y_true, epsilon: float = 1e-4):
    flat_pred = y_pred.reshape([-1, 1])
    flat_true = y_true.reshape([-1, 1])
    err = abs(subtract(flat_true, flat_pred))
    mape = abs(divide(err, flat_true + epsilon))
    return mape


def calculate_mae_per_node(y_pred, y_true):
    flat_pred = y_pred.reshape([-1, 1])
    flat_true = y_true.reshape([-1, 1])
    mae = abs(subtract(flat_true, flat_pred))
    return mae


def calculate_accuracy(y_pred, y_true, threshold=0.2):
    mae = abs(subtract(y_true, y_pred))
    acc = (mae <= (y_true * threshold)).float()
    return mean(acc)


def calculate_correlation_coefficient(y_pred, y_true):
    vx = y_pred - mean(y_pred)
    vy = y_true - mean(y_true)

    cost = sum(vx * vy) / (sqrt(sum(vx ** 2)) * sqrt(sum(vy ** 2)))
    # cov    = torch.mul(y_pred-y_pred.mean(), y_true-y_true.mean()).mean()
    # std   = torch.sqrt(torch.mul(torch.square(y_pred-y_pred.mean()), torch.square(y_true-y_true.mean()))).mean()

    return clamp(cost, -1.0, 1.0)


def calculate_r2(y_pred, y_true):
    r = calculate_correlation_coefficient(y_pred, y_true)
    return r ** 2


def get_default_metric_fn_collection(prefix: str, task: Literal["supervised", "semi"] = "semi") -> dict:
    """util creating metric funtions

    Args:
        prefix (str): set a prefix name for tracking these experiment

    Returns:
        dict: contains functional name and callable functions
    """
    if task == "semi":
        metric_fn_dict = {
            f"{prefix}_error": calculate_rel_error2,
            f"{prefix}_0.1": partial(calculate_accuracy, threshold=0.1),
            f"{prefix}_corr": calculate_correlation_coefficient,
            f"{prefix}_r2": calculate_r2,
            f"{prefix}_mae": F.l1_loss,
            f"{prefix}_rmse": calculate_rmse,
            f"{prefix}_mynse": partial(calculate_nse, exponent=2),
        }
    else:
        metric_fn_dict = {
            f"{prefix}_acc": calculate_classified_accuracy,
        }
    return metric_fn_dict


def print_single_metrics(epoch: int, **kwargs: Any) -> None:
    formatter = f"Epoch: {epoch:03}, "  # f'Epoch: {epoch:0.3d}'
    for k, v in kwargs.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                formatter += f"{sk}: {sv:.4f}, "
        else:
            formatter += f"{k}: {v:.4f}, "
    print(formatter)


def print_metrics(epoch: int, tr_loss: float, val_loss: float, tr_metric_dict: dict, val_metric_dict: dict) -> None:
    """support pretty print string format

    Args:
        epoch (int): current epoch/ trial
        tr_loss (float): training loss
        val_loss (float): validation loss
        tr_metric_dict (dict): training metric dict including name, values
        val_metric_dict (dict): validation metric dict including name, values
    """
    metric_log = ""

    for k, v in tr_metric_dict.items():
        metric_log += f"{k}: {v:.4f}, "

    if val_metric_dict:
        for k, v in val_metric_dict.items():
            metric_log += f"{k}: {v:.4f}, "

    formatter = f"Epoch: {epoch:03d}, train loss: {tr_loss:.4f},"
    formatter += f"val_loss: {val_loss:.4f}," if val_loss else ""
    formatter += f" {metric_log}"
    print(formatter)


def log_metrics_on_wandb(epoch: int, commit: bool = True, is_epoch_a_trial=False, **kwargs):
    """support function allowing to push log to wandb server

    Args:
        epoch (int): deterministic epoch
        commit (bool, optional): if it is one of non-last incremental logs, set it to True. Defaults to True.
    """
    for k, v in kwargs.items():
        if isinstance(v, dict):
            wandb.log(v, commit=False)
        else:
            wandb.log({k: v}, commit=False)
    if is_epoch_a_trial:
        wandb.log({"trial": epoch}, commit=commit)
    else:
        wandb.log({"epoch": epoch}, commit=commit)


def load_secret(secret_path: str | None, secret_file_name: str = "mysecret") -> Optional[dict]:
    if secret_path is None:
        return None
    secret_extension = secret_path[-2:]
    if secret_extension == "py":
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(secret_file_name, secret_path)
        module = importlib.util.module_from_spec(spec)  # type:ignore
        sys.modules[secret_file_name] = module
        spec.loader.exec_module(module)  # type:ignore
        return module.secrets
    else:
        raise NotImplementedError()


def apply_masks(x: Tensor, masks: list[Tensor]) -> Tensor:
    """apply a mask onto a tensor x. If the list of masks is greater than 1, func returns filtered/ applied tensor x w.r.t. elements in this list

    Args:
        x (Tensor): a tensor should has shape  (batch_size * num_graphs, num_features)  or  (num_masks, batch_size * num_graphs, num_features)
        masks (list[Tensor]): a mask list whose length is num_masks. Each element must have shape (batch_size* num_graphs)

    Returns:
        Tensor: masked x has shape  (batch_size * num_graphs, num_features) | (num_masks, batch_size * num_graphs, num_features)
    """
    # x has shape (num_masks, batch_size * num_graphs, num_features)
    # masks has length = num_masks, masks's element has shape  (batch_size* num_graphs)

    num_masks = len(masks)
    if num_masks == 1:
        if len(x.size()) == 2:
            return x[masks[0]]
        else:
            return x.squeeze(0)[masks[0]]

    xs = []
    for i in range(num_masks):
        mask = masks[i]
        xs.append(x[i][mask])

    # out has shape = (num_masks, batch_size * num_graphs, num_features)
    return cat(xs, dim=0)


def apply_znorm(value: Tensor, denormalize: bool, mean: Tensor, std: Tensor, to_device: bool = False) -> Tensor:
    if to_device:
        mean = mean.to(value.get_device())
        std = std.to(value.get_device())

    if not denormalize:
        value = (value - mean) / std
    else:
        value = value * std + mean
    return value


class ZNormalize(BaseTransform):
    def __init__(self, denormalize: bool, mean: Sequence | Tensor, std: Sequence | Tensor, attr: str = "x",
                 device: Optional[str] = None) -> None:
        mean = mean.detach() if isinstance(mean, Tensor) else as_tensor(mean)
        std = std.detach() if isinstance(std, Tensor) else as_tensor(std)

        self.mean: Tensor = mean.to(device=device)
        self.std: Tensor = std.to(device=device)
        self.attr = attr
        self.denormalize = denormalize
        # self.eps :float = 1e-5

    def forward(self, data: Data) -> Data:
        # adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/normalize_features.html#NormalizeFeatures

        # reshape if value has different shape (len of shape > 2)
        attr_val: Tensor | None = getattr(data, self.attr)
        assert attr_val is not None
        len_attr_val_shape = len(attr_val.shape)
        if len_attr_val_shape != len(self.mean.shape):
            new_shape = [1 if i < len_attr_val_shape - 1 else self.mean.shape[-1] for i in range(len_attr_val_shape)]
            self.mean = self.mean.reshape(new_shape)
            self.std = self.std.reshape(new_shape)

        if attr_val.device != self.mean.device:
            self.mean = self.mean.to(attr_val.device)
            self.std = self.std.to(attr_val.device)

        attrs: list[str] = [self.attr]
        for store in data.stores:
            for key, value in store.items(*attrs):
                if value.numel() > 0:
                    value = self.transform(value=value, denormalize=self.denormalize)
                    store[key] = value
        return data

    def to(self, device: str):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def transform(self, value: Tensor, denormalize: bool, **kwargs) -> Tensor:
        train_default: bool = kwargs.get("train_default", False)
        is_pretraining: bool = kwargs.get("is_pretraining", False)
        shared_encoder: bool = kwargs.get("shared_encoder", False)
        dual_encoder: bool = kwargs.get("dual_encoder", False)

        if not denormalize:
            value = (value - self.mean) / self.std
        else:
            # if is_pretraining:
            #     pass
            # else:
            # if is_pretraining and not shared_encoder and not dual_encoder and self.mean.shape[1] != value.shape[1]:
            #     self.mean = self.mean[:, [0, 3, 4]]
            #     self.std = self.std[:, [0, 3, 4]]
            if (is_pretraining, shared_encoder or dual_encoder or train_default) and self.mean.dim() > 1:
                self.mean = self.mean[:, 0]
                self.std = self.std[:, 0]

            value = value * self.std + self.mean
        return value


class MinMaxNormalize(BaseTransform):
    def __init__(
            self, denormalize: bool, min_val: Sequence | Tensor, max_val: Sequence | Tensor, attr: str = "x",
            device: Optional[str] = None
    ) -> None:
        min_val = min_val.detach() if isinstance(min_val, Tensor) else as_tensor(min_val)
        max_val = max_val.detach() if isinstance(max_val, Tensor) else as_tensor(max_val)

        self.min_val: Tensor = min_val.to(device=device)
        self.max_val: Tensor = max_val.to(device=device)
        self.attr = attr
        self.denormalize = denormalize
        # self.eps :float = 1e-5

    def forward(self, data: Data) -> Data:
        # adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/normalize_features.html#NormalizeFeatures

        # reshape if value has different shape (len of shape > 2)
        attr_val: Tensor | None = getattr(data, self.attr)
        assert attr_val is not None
        len_attr_val_shape = len(attr_val.shape)
        if len_attr_val_shape != len(self.min_val.shape):
            new_shape = [1 if i < len_attr_val_shape - 1 else self.min_val.shape[-1] for i in range(len_attr_val_shape)]
            self.min_val = self.min_val.reshape(new_shape)
            self.max_val = self.max_val.reshape(new_shape)

        if attr_val.device != self.min_val.device:
            self.min_val = self.min_val.to(attr_val.device)
            self.max_val = self.max_val.to(attr_val.device)
        attrs: list[str] = [self.attr]
        for store in data.stores:
            for key, value in store.items(*attrs):
                if value.numel() > 0:
                    value = self.transform(value=value, denormalize=self.denormalize)
                    store[key] = value
        return data

    def to(self, device: str):
        self.max_val = self.max_val.to(device)
        self.min_val = self.min_val.to(device)

    def transform(self, value: Tensor, denormalize: bool, eps: float = 1e-16, **kwargs) -> Tensor:
        is_pretraining: bool = kwargs.get("is_pretraining", False)
        if is_pretraining:
            self.min_val = self.min_val[:, -2:]
            self.max_val = self.max_val[:, -2:]

        residual = self.max_val - self.min_val
        if not denormalize:
            value = (value - self.min_val) / (residual + eps)
        else:
            value = value * (residual + eps) + self.min_val
        return value


def generate_unique_name_from_config(
        config: TrainConfig,
        dataset_name: str = "",
) -> tuple[str, str, str]:
    # merge model names
    union_model_name = "-".join([model_config.name for model_config in config.model_configs])
    # gen suffix
    # suffix = datetime.today().strftime("%Y%m%d_%H%M")
    suffix = datetime.today().strftime("%Y%m%d_%H%M%S")
    # get current process ID for multiprocessing
    pid = os.getpid()
    suffix = f"{suffix}_{pid}"
    # concat all
    cat_names = [config.run_prefix, dataset_name, union_model_name, suffix]
    # filter null elements
    cat_names = [n for n in cat_names if n != ""]
    # finally merge
    run_name = "+".join(cat_names)
    return run_name, suffix, union_model_name


def string2bytes(size_str: str) -> int:
    # Mapping of units to their respective byte multipliers
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }
    # Extract the numeric part and the unit
    match = re.match(r"(\d+)([A-Za-z]+)", size_str.strip())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    number, unit = match.groups()
    number = int(number)
    unit = unit.upper()
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    # Calculate the bytes
    return number * units[unit]


def select_batch_size(dataset: Dataset, memory_limit: str) -> int:
    avail_memory_in_bytes = string2bytes(memory_limit)
    single_instace_in_bytes = sys.getsizeof(dataset[0])
    batch_size = avail_memory_in_bytes // single_instace_in_bytes
    return batch_size


class DoubleDataLoader(Iterable):
    scene_batch_size: int
    snapshot_batch_size: int
    shuffle: bool
    memory_limit: str

    def __init__(
            self, dataset: Dataset, memory_limit: str, snapshot_batch_size: int, scene_batch_size: int = 0,
            shuffle: bool = False, **kwargs: Any
    ) -> None:
        self.memory_limit = memory_limit
        self.snapshot_batch_size = snapshot_batch_size
        self.shuffle = shuffle
        self.scene_batch_size = select_batch_size(dataset,
                                                  self.memory_limit) if scene_batch_size <= 0 else scene_batch_size
        self.scene_loader = GraphDataLoader(dataset, batch_size=self.scene_batch_size, shuffle=shuffle, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        for i, batch_scenes in enumerate(self.scene_loader):
            assert isinstance(batch_scenes, Batch)
            batch_snapshot_ids = arange(0, batch_scenes.x.shape[0]).to(dtype=int)  # type: ignore
            dataset = TensorDataset(batch_snapshot_ids)
            snapshot_index_loader = InstanceDataLoader(dataset, batch_size=self.snapshot_batch_size)

            try:
                mini_batch_ids = next(iter(snapshot_index_loader))
                data_list = batch_scenes.to_data_list()
                yield Batch.from_data_list([data_list[mini_batch_ids] for i in mini_batch_ids])
            except StopIteration:
                pass


def wrapper_data_loader(
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        sampling_strategy: Literal["batch", "instance"] = "batch",
        **kwargs: Any,
) -> GraphDataLoader:
    if sampling_strategy == "batch":
        assert hasattr(dataset, "__getitems__"), "ERROR! Dataset doesnt support batch loading!"
        single_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(single_sampler, batch_size=batch_size, drop_last=drop_last)
        loader = GraphDataLoader(dataset, sampler=batch_sampler, **kwargs)
    else:
        loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return loader


def prepare_finetune(model, lr_encoder=5e-5, lr_head=1e-3):
    stem_head_params = []
    others = []
    for name, p in model.named_parameters():
        if 'pressure_encoder.lin0' in name or 'pressure_encoder.lin1' in name:
            stem_head_params.append(p)
        else:
            others.append(p)

    # Prepare optimizer with different LRs
    optimizer = torch.optim.Adam([
        {'params': stem_head_params, 'lr': lr_head},
        {'params': others, 'lr': lr_encoder},
    ])

    return optimizer


def prepare_finetune_dual_encoder(model, lr_shared=5e-4, lr=5e-3):
    # model.task_head.apply(reinit_weights)
    # model.task_head = DownstreamTaskHead(32)

    # Prepare optimizer with different LRs
    optimizer = torch.optim.Adam([
        # {'params': model.lin0.parameters(), 'lr': lr_shared},
        {'params': model.structure_encoder.parameters(), 'lr': lr_shared},
        # {'params': model.pressure_encoder.parameters(), 'lr': lr_shared},
        # {'params': model.decoder.parameters(), 'lr': lr_head},
    ], lr=lr, weight_decay=0.00001)

    return optimizer


def compute_statistic_test(y_pred_a, y_pred_b, y_true):
    # compute per-sample absolute errors (for MAE comparison)
    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)

    # difference in errors
    # error_diffs = errors_a - errors_b

    # Wilcoxon Signed-Rank Test (non-parametric)
    result = wilcoxon(errors_a, errors_b)
    print(f"Wilcoxon test: statistic={result.statistic}, p={result.pvalue}")

    # Paired t-test (parametric)
    result = ttest_rel(errors_a, errors_b)
    print(f"Paired t-test: statistic={result.statistic}, p={result.p_value}")
