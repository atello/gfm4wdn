#
# Created on Sat May 10 2025
# Copyright (c) 2025 Andrés Tello
# --------------------------------------------------------------
# Purpose: Run model pretraining on node degree, and
# neighborhood node degree and downstream task with a shared
# GNN-based encoder.
# --------------------------------------------------------------
#

from __future__ import annotations
from copy import deepcopy
import os
from typing import Any, Callable
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.transforms import Compose

from gigantic_dataset.utils.configs import TrainConfig, GidaConfig
from gigantic_dataset.core.train import (
    train, eval,  WandbStartProfiler, TrainOneEpochDE, TestOneEpochDE, MaskedDualEncoder
)

from gigantic_dataset.core.run import extract_dataset_name
from gigantic_dataset.utils.pretrain_utils import LambdaScheduler

from gigantic_dataset.utils.train_protos import (
    load_gida_datasets,
    default_post_foward_transform,
    default_load_criterion,
    default_load_optimizers,
    default_load_scheduler,
)
from gigantic_dataset.utils.train_utils import (
    MinMaxNormalize,
    ZNormalize,
    get_default_metric_fn_collection,
    find_latest_files,
)
from gigantic_dataset.models.gnn_models import LoadGraphWaterSE, LoadGraphWaterDualEncoder, \
    LoadGraphWaterDualEncoderFinetune, LoadGraphWaterDualEncoderPretrainedStructural, \
    LoadGraphWaterDualEncoderPretrainedStructuralFunctional, LoadDualEncoderGFM, LoadDualEncoderGFMFineTune

from torch_geometric.data import Data
from gigantic_dataset.utils.train_protos import ConfigRef

from gigantic_dataset.utils.func_ref import FuncRef
from functools import partial

import numpy as np


# def lambda_schedule(epoch):
#     return max(0.2, 1 - epoch / 50)  # gradually shift focus from pretraining to pressure


def dual_encoder_train(
        gida_yaml_path: str,
        train_yaml_path: str,
        save_path: str = "",
        custom_stats_tuple_pt_path: str = "",
        custom_subset_shuffle_pt_path: str = "",
) -> Any:
    """prepare for the pressure estimation task on gida

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.save_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.save_path`. Defaults to "".
    Returns:
        Any: return dict if possible
    """
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    train_config.save_path = save_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=MaskedDualEncoder(),
        train_one_epoch_fn=TrainOneEpochDE(),
        test_one_epoch_fn=TestOneEpochDE(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
            custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, is_training=True
        ),
        # load_models=LoadGraphWaterDualEncoderPretrainedStructural(),
        load_models=LoadDualEncoderGFM(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )
    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)

    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1  # <--------------------- CHECK
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        reduced_train_tf = deepcopy(train_tf)
        if in_node_dim > 1 and out_node_dim == 1:
            if isinstance(reduced_train_tf, ZNormalize):
                reduced_train_tf.mean = reduced_train_tf.mean[..., 0]
                reduced_train_tf.std = reduced_train_tf.std[..., 0]
            elif isinstance(reduced_train_tf, MinMaxNormalize):
                reduced_train_tf.min_val = reduced_train_tf.min_val[..., 0]
                reduced_train_tf.max_val = reduced_train_tf.max_val[..., 0]

        if isinstance(reduced_train_tf, Compose):
            reduced_train_tf = reduced_train_tf.transforms[0]
        reduced_train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(reduced_train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(in_dims=[in_node_dim], out_dims=[out_node_dim])

    lambda_scheduler = LambdaScheduler()

    # run train
    ret_dict = func_ref.train_fn(
        datasets=datasets,
        models=models,
        train_metric_fn_dict=get_default_metric_fn_collection(prefix="train", task="semi"),
        val_metric_fn_dict=get_default_metric_fn_collection(prefix="val", task="semi"),
        dual_encoder=True,
        lambda_scheduler=lambda_scheduler,
        lambda_scheduler_variant="cos",
        finetuning_dual=False,
    )
    # temporarily comment for fast check
    # TODO: If you wish to switch wandb project, we must re-call start profiler fn and override the project name
    func_ref.start_profiler_fn(dataset_name=dataset_name,
                               overriden_project_name=train_config.project_name.replace("train", "test"))

    # test
    func_ref.eval_fn(
        datasets=datasets,
        models=models,
        test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
        dual_encoder=True,
        finetuning_dual=False,
    )

    return ret_dict


def dual_encoder_train_pretrained_encoders(
        gida_yaml_path: str,
        train_yaml_path: str,
        save_path: str = "",
        custom_stats_tuple_pt_path: str = "",
        custom_subset_shuffle_pt_path: str = "",
) -> Any:
    """prepare for the pressure estimation task on gida

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.save_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.save_path`. Defaults to "".
    Returns:
        Any: return dict if possible
    """
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    train_config.save_path = save_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=MaskedDualEncoder(),
        train_one_epoch_fn=TrainOneEpochDE(),
        test_one_epoch_fn=TestOneEpochDE(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
            custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, is_training=True
        ),
        load_models=LoadGraphWaterDualEncoderPretrainedStructuralFunctional(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )
    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)

    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1  # <--------------------- CHECK
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        reduced_train_tf = deepcopy(train_tf)
        if in_node_dim > 1 and out_node_dim == 1:
            if isinstance(reduced_train_tf, ZNormalize):
                reduced_train_tf.mean = reduced_train_tf.mean[..., 0]
                reduced_train_tf.std = reduced_train_tf.std[..., 0]
            elif isinstance(reduced_train_tf, MinMaxNormalize):
                reduced_train_tf.min_val = reduced_train_tf.min_val[..., 0]
                reduced_train_tf.max_val = reduced_train_tf.max_val[..., 0]

        reduced_train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(reduced_train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(in_dims=[in_node_dim], out_dims=[out_node_dim], is_pretraining=True)

    # run train
    ret_dict = func_ref.train_fn(
        datasets=datasets,
        models=models,
        train_metric_fn_dict=get_default_metric_fn_collection(prefix="train", task="semi"),
        val_metric_fn_dict=get_default_metric_fn_collection(prefix="val", task="semi"),
        # is_pretraining=True,
        dual_encoder=True,
        # lambda_schedule=lambda_schedule,
        finetuning_dual=False,
    )
    # temporarily comment for fast check
    # TODO: If you wish to switch wandb project, we must re-call start profiler fn and override the project name
    func_ref.start_profiler_fn(dataset_name=dataset_name,
                               overriden_project_name=train_config.project_name.replace("train", "test"))

    # test
    func_ref.eval_fn(
        datasets=datasets,
        models=models,
        test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
        # is_pretraining=True,
        dual_encoder=True,
        # lambda_schedule=lambda_schedule,
        finetuning_dual=False,
    )

    return ret_dict


def dual_encoder_inference(
    gida_yaml_path: str,
    train_yaml_path: str,
    save_path: str = "",
    custom_stats_tuple_pt_path: str = "",
    custom_subset_shuffle_pt_path: str = "",
) -> None:
    """for inference only <br />

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.save_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.save_path`. Defaults to "".
    Returns:
        list[str]: return save_path where storing model weights and training stuff.
    """  # noqa: E501
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    if save_path != "":
        train_config.save_path = save_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=MaskedDualEncoder(),
        train_one_epoch_fn=TrainOneEpochDE(),
        test_one_epoch_fn=TestOneEpochDE(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path, custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, is_training=False
        ),
        # load_datasets=load_dask_datasets,
        load_models=LoadDualEncoderGFM(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )
    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)

    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    print(f"sample.x = {sample.x}")
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        train_tf = deepcopy(train_tf)

        if isinstance(train_tf, Compose):
            train_tf = train_tf.transforms[0]

        train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(model_configs=train_config.model_configs, in_dims=[in_node_dim], out_dims=[out_node_dim])

    # test
    test_metrics_array = []
    for _ in range(10):
        metrics = func_ref.eval_fn(
            datasets=datasets,
            models=models,
            test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
            dual_encoder=True,
            is_inference=True,
        )
        test_metrics_array.append([metrics[k].item() for k, v in metrics.items()])

    mean_val = np.mean(test_metrics_array, axis=0)
    std_val = np.std(test_metrics_array, axis=0)

    for idx, item in enumerate(metrics.items()):
        print(f"{item[0]}: {mean_val[idx]:.4f} ±{std_val[idx]:.3f}")


def finetune_dual_encoder(
        gida_yaml_path: str,
        train_yaml_path: str,
        save_path: str = "",
        custom_stats_tuple_pt_path: str = "",
        custom_subset_shuffle_pt_path: str = "",
) -> Any:
    """prepare for the pressure estimation task on gida

    Args:
        gida_yaml_path (str): gida path, corresponding to parameter set of GiDa Interface
        train_yaml_path (str): training config, parameter set for training stuff
        save_path (str, optional): to override train_config.save_path. Leave blank to auto-gen save path (and folder). Defaults to "".
        custom_stats_tuple_pt_path (str, optional): Custom .pt file to LOAD (READ-ONLY) stats tuple. If empty, we load stats from the default dataset log in `train_config.save_path`. Defaults to "".
        custom_subset_shuffle_pt_path (str, optional):Custom .pt file to LOAD (READ-ONLY) subset shuffle ids. If empty, we load ids from the default dataset log in `train_config.save_path`. Defaults to "".
    Returns:
        Any: return dict if possible
    """
    # load dataset config
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    # load train config
    train_config = TrainConfig()
    train_config._parsed = True
    train_config._from_yaml(train_yaml_path, unsafe_load=True)
    train_config.save_path = save_path

    # to flush to terminal
    setattr(train_config, "data", gida_config.as_dict())

    # add function references, where we mix and match training code.
    dataset_name = extract_dataset_name(gida_config)
    func_ref = FuncRef(
        start_profiler_fn=partial(WandbStartProfiler(), dataset_name=dataset_name),
        forward_fn=MaskedDualEncoder(),
        train_one_epoch_fn=TrainOneEpochDE(),
        test_one_epoch_fn=TestOneEpochDE(),
        train_fn=train,
        eval_fn=eval,
        post_forward_tf_fn=default_post_foward_transform,
        load_criterion=default_load_criterion,
        load_datasets=partial(
            load_gida_datasets, custom_stats_tuple_pt_path=custom_stats_tuple_pt_path,
            custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path
        ),
        load_models=LoadDualEncoderGFMFineTune(),
        load_optimizers=default_load_optimizers,
        load_scheduler=default_load_scheduler,
    )
    # initialize the ConfigRef which we can call from anywhere
    ConfigRef.initialize_and_start_profiler(config=train_config, ref=func_ref)

    # we first load dataset
    datasets: list[Dataset] = func_ref.load_datasets(gida_config)
    print(f"len={datasets[0].len()}")

    # gather in_dim and out_dim for loading models
    sample: Data = datasets[0][0]  # type: ignore
    in_node_dim = sample.x.shape[-1]  # type:ignore
    if sample.y is not None:
        out_node_dim = sample.y.shape[-1]  # type:ignore
    elif hasattr(sample, "edge_y") and sample.edge_y is not None:
        out_node_dim = sample.edge_y.shape[-1]  # type:ignore
    else:
        if in_node_dim > 1:  # extra attrs
            out_node_dim = 1  # <--------------------- CHECK
        else:
            out_node_dim = in_node_dim

    # take transform from gida to perform inverse normalization
    if train_config.norm_type != "unused":
        train_tf: ZNormalize | MinMaxNormalize | None = datasets[0].transform  # type:ignore
        assert train_tf is not None
        reduced_train_tf = deepcopy(train_tf)
        if in_node_dim > 1 and out_node_dim == 1:
            if isinstance(reduced_train_tf, ZNormalize):
                reduced_train_tf.mean = reduced_train_tf.mean[..., 0]
                reduced_train_tf.std = reduced_train_tf.std[..., 0]
            elif isinstance(reduced_train_tf, MinMaxNormalize):
                reduced_train_tf.min_val = reduced_train_tf.min_val[..., 0]
                reduced_train_tf.max_val = reduced_train_tf.max_val[..., 0]

        if isinstance(reduced_train_tf, Compose):
            reduced_train_tf = reduced_train_tf.transforms[0]

        reduced_train_tf.to(device=ConfigRef.config.device)
        inverse_apply_fn: Callable = partial(reduced_train_tf.transform, denormalize=True)
        func_ref.post_forward_tf_fn = partial(func_ref.post_forward_tf_fn, tf=inverse_apply_fn)

    # start model
    models = func_ref.load_models(in_dims=[in_node_dim], out_dims=[out_node_dim])

    lambda_scheduler = LambdaScheduler()

    # run train
    ret_dict = func_ref.train_fn(
        datasets=datasets,
        models=models,
        train_metric_fn_dict=get_default_metric_fn_collection(prefix="train", task="semi"),
        val_metric_fn_dict=get_default_metric_fn_collection(prefix="val", task="semi"),
        # is_pretraining=False,
        dual_encoder=True,
        lambda_scheduler=lambda_scheduler,
        lambda_scheduler_variant="cos",
        finetuning=True,
    )
    # temporarily comment for fast check
    # TODO: If you wish to switch wandb project, we must re-call start profiler fn and override the project name
    func_ref.start_profiler_fn(dataset_name=dataset_name,
                               overriden_project_name=train_config.project_name.replace("train", "test"))

    # test
    func_ref.eval_fn(
        datasets=datasets,
        models=models,
        test_metric_fn_dict=get_default_metric_fn_collection(prefix="test", task="semi"),
        # is_pretraining=True,
        dual_encoder=False,
        # finetuning_dual=True,
        # lambda_schedule=lambda_schedule,
    )

    return ret_dict
