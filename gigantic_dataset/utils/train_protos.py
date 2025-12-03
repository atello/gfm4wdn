#
# Created on Sun Nov 24 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Create functional prototols for training a DL model
# ------------------------------
#

import os
import timeit
from typing import Any, Callable, Literal, Optional, Protocol

# from gigantic_dataset.core.datasets_nsf import GidaV6_NSF
from gigantic_dataset.utils.configs import TrainConfig, ModelConfig, GidaConfig  # , GidaNSFConfig,
from gigantic_dataset.core.datasets_large import GidaV6
from gigantic_dataset.utils.train_utils import get_criterion, ZNormalize, MinMaxNormalize, find_latest_files, load_custom_checkpoint
from torch import nn
from torch import Tensor
from torch.optim import Optimizer, lr_scheduler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from torch_geometric.transforms import BaseTransform, Compose

from torch import load
from torch.optim import Adam
import os.path as osp
from datetime import datetime
import time

from gigantic_dataset.utils.func_ref import FuncRef


class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]

    def initialize_and_start_profiler(self, *args, **kwargs):
        """
        Override this method in subclasses to perform initialization.
        This method will be called explicitly when needed.
        """
        pass


class ConfigRef(Singleton):
    config: TrainConfig
    ref: FuncRef

    @staticmethod
    def initialize_and_start_profiler(config: TrainConfig, ref: FuncRef) -> None:
        ConfigRef.config = config
        ConfigRef.ref = ref
        # load references
        if ConfigRef.ref is not None:
            run_name, suffix = ConfigRef.ref.start_profiler_fn()
            if ConfigRef.config.save_path == "":
                ConfigRef.config.save_path = os.path.join("gigantic_dataset/experiments_logs", run_name)
                os.makedirs(ConfigRef.config.save_path, exist_ok=False)


class StartProfilerProto(Protocol):
    def __call__(self, **kwargs: Any) -> tuple[str, str]:
        raise NotImplementedError()


class ForwardProto(Protocol):
    def __call__(self, models: list[nn.Module], data: Data, batch_mask: Tensor, **kwargs: Any) -> tuple[Any, Any, Any | None]:
        raise NotImplementedError()


class TrainOneEpochProto(Protocol):
    def __call__(
        self,
        models: list[nn.Module],
        optimizers: list[Optimizer],
        loader: DataLoader,
        criterion: Callable[..., Any],
        metric_fn_dict: dict[str, Callable[..., Any]],
        **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        raise NotImplementedError()


class TestOneEpochProto(Protocol):
    def __call__(
        self,
        models: list[nn.Module],
        loader: DataLoader,
        criterion: Callable[..., Any],
        metric_fn_dict: dict[str, Callable[..., Any]],
        **kwargs: Any,
    ) -> tuple[float, dict, Any]:
        raise NotImplementedError()


class TrainProto(Protocol):
    def __call__(
        self,
        models: list[nn.Module],
        datasets: list[Dataset],
        train_metric_fn_dict: dict[str, Callable],
        val_metric_fn_dict: dict[str, Callable],
        **kwargs: Any,
    ) -> Any | None:
        raise NotImplementedError()


class EvalProto(Protocol):
    def __call__(
        self,
        models: list[nn.Module],
        datasets: list[Dataset],
        test_metric_fn_dict: dict[str, Callable],
        **kwargs: Any,
    ) -> Any | None:
        raise NotImplementedError()


class PostForwardTransformProto(Protocol):
    def __call__(self, y_pred: Tensor, y_true: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()


class LoadModelProto(Protocol):
    def __call__(self, in_dims: list[int], out_dims: list[int], **kwds: Any) -> list[nn.Module]:
        raise NotImplementedError()


class LoadDatasetsProto(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> list[Dataset]:
        raise NotImplementedError()


class LoadOptimizersProto(Protocol):
    def __call__(self, models: list[nn.Module], **kwds: Any) -> list[Optimizer]:
        raise NotImplementedError()


class LoadCriterionProto(Protocol):
    def __call__(self, **kwds: Any) -> Callable:
        raise NotImplementedError()


class LoadSchedulerProto(Protocol):
    def __call__(self, **kwds: Any) -> Optional[Any]:
        raise NotImplementedError()


###################### SUPPORT SOME DEFAULT FUNCTIONS #################################################


def default_start_profiler(model_name: str = "", **kwargs: Any) -> tuple[str, str]:
    suffix = datetime.today().strftime("%Y%m%d_%H%M")
    return "", suffix


# def gather_statistic(
#     gida_config: GidaConfig, which_array: Literal["node", "edge", "label", "edge_label"] = "node"
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     """we parse the datset to a GiDa interface and gather stats one.
#     Note that the actual data interface could be simplier (and faster)
#
#     Args:
#         gida_config (GidaConfig): gida config
#
#     Returns:
#         tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray] | tuple[Tensor, Tensor, Tensor, Tensor]: min_val, max_val, mean_val, std_val
#
#     """
#     actual_num_records: int | None = gida_config.num_records
#     gida_param_dict = gida_config.as_dict()
#     gida_param_dict["num_records"] = None
#     full_gida = GidaV5(**gida_param_dict)
#
#     train_set = full_gida.get_set(full_gida.train_ids, num_records=actual_num_records)
#
#     min_val, max_val, mean_val, std_val = train_set.gather_statistic_v3(which_array=which_array, to_tensor=True)  # type:ignore
#
#     return min_val, max_val, mean_val, std_val  # type:ignore


def load_stat_tuple_from_path(
    train_set: GidaV6,
    custom_stats_tuple_pt_path: str,
    which_array: Literal["node", "edge", "label", "edge_label"] = "node",
    num_batches: int = 10,
    verbose: bool = True,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    formated_prefix = f"{which_array}_"

    min_val_key = formated_prefix + "min_val"
    max_val_key = formated_prefix + "max_val"
    mean_val_key = formated_prefix + "mean_val"
    std_val_key = formated_prefix + "std_val"

    keys = [min_val_key, max_val_key, mean_val_key, std_val_key]

    stat_tuple: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None] = (None, None, None, None)
    has_custom_stats_path: bool = False
    if custom_stats_tuple_pt_path != "" and os.path.isfile(custom_stats_tuple_pt_path):
        # load stat_tuples given custom_stats_tuple_path
        my_dict: dict[str, Tensor] = load_custom_checkpoint(custom_stats_tuple_pt_path, keys=keys)
        # set a boolean flag indicating the checkpoint existance
        has_custom_stats_path = True
        if my_dict:
            # effort to convert legacy codes
            if min_val_key not in my_dict and which_array == "node":
                min_val_key = "min_val"
                max_val_key = "max_val"
                mean_val_key = "mean_val"
                std_val_key = "std_val"
            stat_tuple = (my_dict[min_val_key], my_dict[max_val_key], my_dict[mean_val_key], my_dict[std_val_key])

    if stat_tuple[0] is None and stat_tuple[2] is None:
        assert isinstance(train_set, GidaV6)
        stat_tuple = train_set.gather_statistic(which_array=which_array, to_tensor=True, num_batches=num_batches)  # type:ignore

        # we only save STATs if has_custom_stats_path is False (while training). Otherwise, we only infer but not override the save (while testing)
        if not has_custom_stats_path:
            # save to use
            save_dict: dict[str, Any] = {
                min_val_key: stat_tuple[0],
                max_val_key: stat_tuple[1],
                mean_val_key: stat_tuple[2],
                std_val_key: stat_tuple[3],
            }
            # support legacy keys
            if which_array == "node":
                save_dict["min_val"] = stat_tuple[0]
                save_dict["max_val"] = stat_tuple[1]
                save_dict["mean_val"] = stat_tuple[2]
                save_dict["std_val"] = stat_tuple[3]

            train_set.save_dataset_checkpoint(**save_dict)

    min_val: Tensor | None = stat_tuple[0]
    max_val: Tensor | None = stat_tuple[1]
    mean_val: Tensor | None = stat_tuple[2]
    std_val: Tensor | None = stat_tuple[3]

    if verbose:
        if has_custom_stats_path:
            print("#" * 40 + f"STATISTICS of <{which_array}>- gathered from file <{custom_stats_tuple_pt_path}>" + "#" * 40, flush=True)  # type:ignore
        else:
            print(
                "#" * 40 + f"STATISTICS of <{which_array}>- gathered from train set with size of <{train_set.length}> records" + "#" * 40, flush=True
            )  # type:ignore
        print(f"min_val = {min_val}", flush=True)
        print(f"max_val = {max_val}", flush=True)
        print(f"mean_val = {mean_val}", flush=True)
        print(f"std_val = {std_val}", flush=True)
        print("#" * 140, flush=True)

    return (min_val, max_val, mean_val, std_val)


def create_transform_from_path(
    norm_type: str,
    train_set: GidaV6,
    custom_stats_tuple_pt_path: str,
    which_array: Literal["node", "edge", "label", "edge_label"] = "node",
    num_batches: int = 10,
    verbose: bool = True,
    device: str | None = None,
) -> BaseTransform | None:
    (min_val, max_val, mean_val, std_val) = load_stat_tuple_from_path(
        train_set=train_set, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path, which_array=which_array, num_batches=num_batches, verbose=verbose
    )

    which_array2data_prop = {
        "node": "x",
        "edge": "edge_attr",
        "label": "y",
        "edge_label": "y",
    }

    my_dict = {
        "minmax": MinMaxNormalize(denormalize=False, min_val=min_val, max_val=max_val, attr=which_array2data_prop[which_array], device=device),  # type:ignore
        "znorm": ZNormalize(denormalize=False, mean=mean_val, std=std_val, attr=which_array2data_prop[which_array], device=device),  # type:ignore
    }

    return my_dict.get(norm_type, None)  # type:ignore


def pre_proccessing(
    full_gida: GidaV6,
    gida_config: GidaConfig,
    custom_stats_tuple_pt_path: str = "",
    num_batches: int = 10,
    verbose: bool = True,
) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
    """
    This utils function helps split the datasets into subsets for train/val/test.<br />
    NOTE: The normalization could be affected by the length of selected training set.<br />
    NOTE: GidaV6 doesn't support loading stats. You must declare via `stat_tuple`. Unless we compute them based on train set.<br />

    Concretely, the subset size is ALWAYS pre-defined before splitting with FIXED ratio (60-20-20).<br />
    Still, the size is still outrageously large, so we can selectively take the preferred amount using `gida_config.num_records`<br />
    If this parameter is set, we recompute the number of taken records, and get the amount from extracted set IN ORDER if `gida_config.subset_shuffle== False`, or, otherwise, random.<br />
    .. code-block:: text
        PREDEFINED                              COMPUTED
        -------------                         ---------------
        |  Fixed    |                         |  New train  |
        |  Train    | -> new_amount * 60% =   |       ******|
        |           |                         |*************|
        -------------                         ---------------
        | Fixed Val | -> new_amount * 20% =   |NVal*********|
        -------------                         ---------------
        | Fixed Test| -> new_amount * 20% =   |NTest********|
        -------------                         ---------------
        (*) represents the skipping so that the new set is often smaller (or equal) the corresponding predefined one.
    ...
    Args:
        full_gida (GidaV6): gida object
        gida_config (GidaConfig): gida config
        custom_stats_tuple_pt_path (str, optional): if non-empty, stats are loaded from this .pt path. Defaults set to empty.
        num_batches (int, optional): For computing statistic if undefined, the train set is split into `num_batches` and fed into Dask sequentially to prevent OOM. Defaults to 10.
        verbos (bool, optional): print debug info. Default is True.
    Returns:
        tuple[Dataset | None, Dataset | None, Dataset | None]: train, valid, test dataset
    """  # noqa: E501
    # actual_num_records: int = gida_config.num_records if gida_config.num_records is not None else full_gida.length
    # compute #samples per set
    # train_samples = int(actual_num_records * full_gida.split_ratios[0])
    # val_samples = int(actual_num_records * full_gida.split_ratios[1])
    # test_samples = actual_num_records - train_samples - val_samples

    train_samples = val_samples = test_samples = None

    if gida_config.split_set in ["train", "all"]:
        train_set = full_gida.get_set(full_gida.train_ids, num_records=train_samples, sampling_strategy="sequential")
    else:
        train_set = None

    valid_set = test_set = None
    # norm_type is used, we must add Transform
    norm_type = ConfigRef.config.norm_type
    which_arrays = ConfigRef.config.norm_on
    if norm_type in ["znorm", "minmax"]:
        my_tfs: list[BaseTransform] = []
        for which_array in which_arrays:
            tf = create_transform_from_path(
                norm_type=ConfigRef.config.norm_type,
                train_set=train_set,  # type: ignore
                custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
                which_array=which_array,
                num_batches=num_batches,
                verbose=verbose,
                device=None,  # <- we won't parse into device at this time (but at runtime)
            )
            assert tf is not None
            my_tfs.append(tf)

        single_tf = Compose(my_tfs) if len(my_tfs) > 1 else my_tfs[0]
    else:
        single_tf = None

    if gida_config.split_set in ["train", "all"]:
        assert train_set is not None
        train_set.transform = single_tf

    if gida_config.split_set in ["val", "all"]:
        valid_set = full_gida.get_set(full_gida.val_ids, num_records=val_samples, sampling_strategy="interval", transform=single_tf)

    if gida_config.split_set in ["test", "all"]:
        test_set = full_gida.get_set(full_gida.test_ids, num_records=test_samples, sampling_strategy="interval", transform=single_tf)

    return train_set, valid_set, test_set

    # train_set = full_gida.get_set(full_gida.train_ids, num_records=train_samples)
    # assert train_set is not None and isinstance(train_set, GidaV6)

    # valid_set = test_set = None
    # # norm_type is used, we must add Transform
    # norm_type = ConfigRef.config.norm_type
    # which_arrays = ConfigRef.config.norm_on
    # if norm_type in ["znorm", "minmax"]:
    #     my_tfs: list[BaseTransform] = []
    #     for which_array in which_arrays:
    #         tf = create_transform_from_path(
    #             norm_type=ConfigRef.config.norm_type,
    #             train_set=train_set,
    #             custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
    #             which_array=which_array,
    #             num_batches=num_batches,
    #             verbose=verbose,
    #             device=None,  # <- we won't parse into device at this time (but at runtime)
    #         )
    #         assert tf is not None
    #         my_tfs.append(tf)
    #
    #     if len(my_tfs) > 1:
    #         composer = Compose(my_tfs)
    #         train_set.transform = composer
    #     else:
    #         train_set.transform = my_tfs[0]
    #
    # if gida_config.split_set in ["val", "all"]:
    #     valid_set = full_gida.get_set(full_gida.val_ids, num_records=val_samples, transform=train_set.transform)
    # if gida_config.split_set in ["test", "all"]:
    #     test_set = full_gida.get_set(full_gida.test_ids, num_records=test_samples, transform=train_set.transform)
    #
    # return train_set if gida_config.split_set in ["train", "all"] else None, valid_set, test_set


def load_gida_datasets(
    gida_config: GidaConfig,
    custom_stats_tuple_pt_path: str = "",
    custom_subset_shuffle_pt_path: str = "",
    is_training: bool = True,
) -> list[Dataset]:
    """Function supports loading gida datasets with GiDa Interface.<br />
    NOTE: If you want pre-define stats tuple when computing normalization, override `custom_stats_tuple_pt_path` with the dataset_log.pt of elsewhere training config. If not, we reuse the `dataset_log.pt` from `save_path` of the CURRENT Training Config. <br />
    NOTE: If testing on different network and ensuring on the same subset of test set, user should override `custom_subset_shuffle_pt_path` in `gida_config`. <br />
    By default, we use `stat_tuple`, which is stored in `dataset_log.pt` in the folder `save_path` of training config accessed via a singleton interface, to do normalization.<br />
    NOTE: pre_processing only normalize 'x' features<br />
    Args:
        gida_config (GidaConfig): gida confi
        custom_stats_tuple_pt_path (str, optional): For loading stat tuple, this parameter is for external loading statistic tuple from elsewhere. In such a case, add the path of `dataset_log.pt` of the testing network here. If `custom_stats_tuple_path==""`, we reuse the `dataset_log.pt` from `save_path` of Trainning Config. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional): For loading subset shuffle ids, this parameter is for external loading statistic tuple from elsewhere. In such a case, add the path of `dataset_log.pt` of the testing network here. If `custom_subset_shuffle_path==""`, we reuse the `dataset_log.pt` from `save_path` of Trainning Config. Defaults to "".

    Returns:
        list[Dataset]: _description_
    """
    train_set, valid_set, test_set = None, None, None

    # default path (if empty, we create)
    if is_training:
        defaut_dataset_log_pt_path = osp.join(ConfigRef.config.save_path, "gida_dataset_log.pt")
        gida_config.dataset_log_pt_path = defaut_dataset_log_pt_path

    # call interface and implicitly load subset shuffle ids from the custom path
    gida_param_dict = gida_config.as_dict()
    full_gida = GidaV6(**gida_param_dict)

    # we perform subset_shuffle prior the custom file. If it is empty, we try to load from dataset_log.pt. If it is failed, we create some and save to dataset_log.pt
    full_gida.process_subset_shuffle(custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, create_and_save_to_dataset_log_if_nonexist=True)

    # full_gida.process_subset_shuffle_custom(custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path,
    #                                         sampling_strategy=gida_config.sampling_strategy,
    #                                         create_and_save_to_dataset_log_if_nonexist=True)

    train_set, valid_set, test_set = pre_proccessing(
        full_gida=full_gida,
        gida_config=gida_config,
        custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
    )

    ret_datasets = []
    if train_set is not None:
        ret_datasets.append(train_set)
    if valid_set is not None:
        ret_datasets.append(valid_set)
    if test_set is not None:
        ret_datasets.append(test_set)
    return ret_datasets  # type:ignore


def default_load_optimizers(
    models: list[nn.Module],
    load_from: Literal["best_in_save_path", "last_in_save_path", "per_model_weight_path"] = "best_in_save_path",
    is_shared_optim: bool = False,
    **kwds: Any,
) -> list[Optimizer]:
    config = ConfigRef.config
    if len(models) == 1:
        optimizers = [Adam(models[0].parameters(), lr=config.lr, weight_decay=config.weight_decay)]
    else:
        if is_shared_optim:
            param_groups = [{"params": (p for n, p in model.named_parameters())} for model in models]
            optimizers = [Adam(param_groups, lr=config.lr, weight_decay=config.weight_decay)]  # TODO: add eps
        else:
            optimizers = [Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) for model in models]

    try:
        if load_from in ["best_in_save_path", "last_in_save_path"] and config.save_path != "":
            prefix = load_from.split("_")[0]
            latest_file = osp.join(config.save_path, prefix + "_training_log.pt")
            cp_dict = load(latest_file)
            optimizers_state_dict = cp_dict["optimizers_state_dict"]
            for i, optim_state in optimizers_state_dict.items():
                optimizers[i].load_state_dict(optim_state)
                print(f"Load optimizer at position [{i}] successfully!")
        else:
            assert len(models) == len(config.model_configs), (
                f"ERROR! #models({len(models)}) should equal to #model_configs({len(config.model_configs)})"
            )
            for i, model_config in enumerate(config.model_configs):
                if model_config.weight_path != "":
                    assert not is_shared_optim, "ERROR! if load_from is 'per_model_weight_path', 'is_shared_optim' must be set to False"
                    cp_dict = load(model_config.weight_path)
                    optimizers_state_dict = cp_dict["optimizers_state_dict"]
                    optimizers[i].load_state_dict(optimizers_state_dict[0])

    except Exception as e:
        print(f"WARNING! Load Optimizer failed! Use new optimizers instead! Exception got: {e}")

    return optimizers  # type:ignore


def default_load_criterion(**kwds: Any) -> Callable:
    return get_criterion(ConfigRef.config.criterion, ConfigRef.config.device)


def default_load_scheduler(**kwds: Any) -> Optional[Any]:

    optimizer = kwds.pop("optimizer", None)
    variant = kwds.pop("scheduler_variant", None)

    if variant == "ReduceLROnPlateau":
        mode = kwds.pop("mode", 'min')
        factor = kwds.pop("factor", 0.5)
        patience = kwds.pop("patience", 5)
        min_lr = kwds.pop("min_lr", 1e-6)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            threshold=0.001,
            factor=factor,  # Reduce LR by half
            patience=patience,  # Wait 5 epochs without improvement
            min_lr=min_lr,  # Don't go lower than this
        )
    elif variant == "CosineAnnealing":
        t_0 = kwds.pop("t_0", 10)
        t_mult = kwds.pop("t_mult", 2)
        eta_min = kwds.pop("eta_min", 1e-6)

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,  # Number of epochs before the first restart
            T_mult=t_mult,  # Multiply T_i by this after a restart (optional)
            eta_min=eta_min  # Minimum learning rate
        )
    else:
        scheduler = None

    return scheduler


def default_post_foward_transform(y_pred: Tensor, y_true: Tensor, tf: Callable[..., Tensor] | None = None, **kwargs: Any) -> tuple[Tensor, Tensor]:
    if tf is not None:
        y_pred = tf(y_pred, **kwargs)
        y_true = tf(y_true, **kwargs)

    return y_pred, y_true
