#
# Created on Thu Mar 14 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Test simgen
# ------------------------------
#
import warnings
import zarr
import numpy as np
import pandas as pd
import tempfile
import math
import json
import logging
from datetime import datetime  # type: ignore
import time
import os

from typing import Any, Callable, Literal, Set, Union, Optional, Generator
from dataclasses import fields
from collections import OrderedDict
from copy import deepcopy

import wntr
from wntr.network import WaterNetworkModel

import wntr.network.elements as wntre
from itertools import compress
from gigantic_dataset.core.simgen import (
    get_curve_parameters,
    get_pattern_parameters,
    get_object_dict_by_config,
    get_value_internal,
    get_default_value_from_global,
    convert_to_float_array,
    generate,
    init_water_network,
    Strategy,
)
from gigantic_dataset.utils.profiler import WDNProfiler
from gigantic_dataset.utils.configs import TuneConfig, SimConfig
from gigantic_dataset.utils.auxil_v8 import (
    upper_bound_IQR,
    lower_bound_IQR,
    get_object_name_list_by_component,
    internal_lower_bound_IQR,
    internal_upper_bound_IQR,
    is_node_simulation_output,
    is_node,
)


def get_dummy_values(
    param_name: str,
    old_records: list[np.ndarray],
    strategy: Strategy,
    is_curve: bool,
    is_pattern: bool,
) -> tuple[Union[None, list], bool]:
    if len(old_records) <= 0:
        return None, False
    if strategy == "keep":
        return None, True
    elif strategy == "sampling":
        if is_curve:
            xs = [v[: (len(v) // 2)] for v in old_records]
            ys = [v[(len(v) // 2) :] for v in old_records]
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            xmin = np.min(xs).astype(float)
            xmax = np.max(xs).astype(float)
            ymin = np.min(ys).astype(float)
            ymax = np.max(ys).astype(float)
            num_curves = old_records[0].shape[-1]
            # [xmin, xmax, ymin, ymax, num_curves]
            return [xmin, xmax, ymin, ymax, num_curves], True
        else:
            old_values_array = np.vstack(old_records)
            min_val = np.min(old_values_array).astype(float)
            max_val = np.max(old_values_array).astype(float)
            return [min_val, max_val], True
    elif strategy == "perturbation":
        if is_curve:
            xs = [v[: (len(v) // 2)] for v in old_records]
            ys = [v[(len(v) // 2) :] for v in old_records]
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)

            std_x = np.std(xs).astype(float)
            std_y = np.std(ys).astype(float)
            return [std_x, std_y], True
        else:
            old_values_array = np.vstack(old_records)
            std_val = np.std(old_values_array).astype(float)
            return [std_val], True
    elif strategy == "adg":
        if param_name == "base_demand":
            seasonal = 1 if np.random.random() > 0.5 else 0
            frequence = np.random.choice([2, 4, 12, 52, 365]).astype(float)
            scale = np.max(np.concatenate(old_records, axis=-1).flatten())
            return [seasonal, frequence, scale], True
        else:
            return None, False
    elif strategy == "adg_v2":
        if param_name == "base_demand":
            scale = np.max(np.concatenate(old_records, axis=-1).flatten())
            return [scale], True
        else:
            return None, False
    elif strategy == "series":
        selected_idx = np.random.permutation(len(old_records))[0]
        ret_list = old_records[selected_idx].tolist()
        return ret_list, True
    else:
        return None, False


def get_old_records(
    obj_dict: list,
    wn: WaterNetworkModel,
    param_name: str,
    time_dim: int,
    is_curve: bool,
    is_pattern: bool,
    skip_names: list[str] = [],
) -> list[np.ndarray]:
    num_points_list = []
    temp_list = []
    for obj_name, obj in obj_dict:
        if obj_name in skip_names:
            continue
        old_value = get_value_internal(obj, param_name, duration=time_dim, timestep=-1)  # timestep= -1 will take base_value
        if old_value is None:  # get a default value from global config
            old_value = get_default_value_from_global(param_name, wn)
        if old_value is not None:
            if not isinstance(old_value, np.ndarray):
                obj_array = convert_to_float_array(old_value)
            else:
                obj_array = old_value

            if is_curve:
                obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

            if is_curve or is_pattern:
                num_points = obj_array.shape[-1]
                num_points_list.append(num_points)
            temp_list.append(obj_array)
    return temp_list


def add_dummy_value(
    tune_config: TuneConfig,
    wn: WaterNetworkModel,
    time_dim: int,
    new_strategy: Strategy,
) -> OrderedDict:
    param_nested_dict = OrderedDict()

    for field in fields(tune_config):
        name = field.name
        tmps = name.split("_")
        param_name = "_".join(tmps[:-1])
        strategy_or_values = tmps[-1]

        if param_name not in param_nested_dict:
            param_nested_dict[param_name] = OrderedDict()
        param_nested_dict[param_name][strategy_or_values] = getattr(tune_config, name)

    for param_name in param_nested_dict:
        # bl_strategy, bl_value = (
        #     param_nested_dict[param_name]["strategy"],
        #     param_nested_dict[param_name]["values"],
        # )

        is_curve = param_name in get_curve_parameters()
        is_pattern = param_name in get_pattern_parameters()
        # param_dtype = get_dtype_by_param_name(param_name)

        # obj_names = get_object_name_list_by_config(tune_config,wn)
        obj_dict = list(get_object_dict_by_config(tune_config, wn))

        old_records = get_old_records(
            obj_dict=obj_dict,
            wn=wn,
            param_name=param_name,
            time_dim=time_dim,
            is_curve=is_curve,
            is_pattern=is_pattern,
        )

        new_values, is_legal = get_dummy_values(
            param_name=param_name,
            old_records=old_records,
            strategy=new_strategy,
            is_curve=is_curve,
            is_pattern=is_pattern,
        )
        if is_legal:
            (
                param_nested_dict[param_name]["strategy"],
                param_nested_dict[param_name]["values"],
            ) = new_strategy, new_values

    return param_nested_dict


def create_dummy_configs_by_component(
    dummy_folder: str,
    dummy_prefix: str,
    blueprint_yaml_path: str,
    unsafe_load: bool = True,
    strategy_list: list[str] = ["sampling", "series", "keep", "perturbation", "adg"],
) -> list[str]:
    config = SimConfig().parse_args([])
    config._from_yaml(yaml_path=blueprint_yaml_path, unsafe_load=unsafe_load)

    if not os.path.exists(dummy_folder):
        os.mkdir(dummy_folder)

    assert os.path.isdir(dummy_folder)

    if dummy_prefix is None or dummy_prefix == "":
        dummy_prefix = os.path.basename(config.inp_paths[0])
        dummy_prefix = dummy_prefix[:-5]

    field_list = list(vars(config))
    tune_list = [field for field in field_list if "tune" in field]
    tune_name = tune_list[0]
    tune_config = getattr(config, tune_name)

    inp_path = config.inp_paths[0]
    wn = WaterNetworkModel(inp_path)
    time_dim = 1 if config.duration <= 1 else config.duration
    dummy_config_paths = []
    for i in range(len(strategy_list)):
        my_tune_config = deepcopy(tune_config)
        new_dict = add_dummy_value(
            tune_config=my_tune_config,
            wn=deepcopy(wn),
            time_dim=time_dim,
            new_strategy=strategy_list[i],  # type:ignore
        )

        tmp_config = SimConfig().parse_args([])
        for param_name, nested_dict in new_dict.items():
            for k, v in nested_dict.items():
                name_attr = f"{param_name}_{k}"
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, list):
                    v = [float(x) for x in v]
                setattr(my_tune_config, name_attr, v)

        setattr(tmp_config, tune_name, my_tune_config)
        dummy_config_path = os.path.join(
            dummy_folder,
            f"{dummy_prefix}_{tune_name.replace('_tune','')}_{strategy_list[i]}_{i}.yaml",
        )
        tmp_config._to_yaml(yaml_path=dummy_config_path)

        dummy_config_paths.append(dummy_config_path)
    return dummy_config_paths


def find_optimal_config(
    blueprint_yaml_path: str,
    report_json_path: str,
    skip_compo_params: list[str] = [],
    log_path: str = "gigantic_dataset/log",
    reinforce_params: bool = False,
    relax_q3_condition: bool = False,
    acceptance_lo_threshold: float = 0.4,
    acceptance_up_threshold: float = 1.0,
    junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    max_iters: int = 10,
    switching_factor: float = 0.5,
    custom_order_keys: list[str] = [],
    custom_skip_keys: list[str] = [],
    **kwargs,
) -> str:
    tmp_name = os.path.basename(blueprint_yaml_path)[:-5]
    postfix = datetime.today().strftime("%Y%m%d_%H%M")

    def setup_logger(logger_name, log_file, level=logging.INFO) -> logging.Logger:
        my_logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        my_logger.setLevel(level)
        my_logger.addHandler(fileHandler)
        my_logger.addHandler(streamHandler)
        return my_logger

    os.makedirs(log_path, exist_ok=True)
    log_file_path = f"{log_path}/findoptimalconfig_{tmp_name}_{postfix}.log"
    logger = setup_logger(tmp_name, log_file_path)
    logger.info(f"log_path: {log_file_path}")
    logger.info(f"blueprint_yaml_path: {blueprint_yaml_path}")
    logger.info(f"report_json_path: {report_json_path}")
    logger.info(f"reinforce_params: {reinforce_params}")
    logger.info(f"relax_q3_condition: {relax_q3_condition}")
    logger.info(f"acceptance_lo_threshold: {acceptance_lo_threshold}")
    logger.info(f"acceptance_up_threshold: {acceptance_up_threshold}")
    logger.info(f"junc_demand_strategy: {junc_demand_strategy}")
    logger.info(f"max_iters: {max_iters}")

    def ndarray2list(arr: np.ndarray | list) -> list:
        if isinstance(arr, np.ndarray):
            ret = arr.tolist()
        elif isinstance(arr, list):
            ret = [float(x) for x in arr]
        else:
            raise NotImplementedError(f"arr has type {type(arr)} is not supported")
        return ret

    def generate_values(
        stat_dict: dict, global_stat_dict: dict, num_trials: int, verbose: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray], float, float, bool]:
        max_val: float = stat_dict["max"]
        min_val: float = stat_dict["min"]
        std_val: float = stat_dict["std"]
        mean_val: float = stat_dict["mean"]

        if min_val == max_val:
            global_max_val: float = global_stat_dict["max"]
            global_min_val: float = global_stat_dict["min"]
            start_min_val = (min_val - global_min_val) / 2
            start_max_val = (global_max_val - max_val) / 2

            if abs(start_max_val - start_min_val) <= 0:
                if verbose:
                    logging.info("Skipped due to zero minmax gap in both local and global setting! Use keep strategy and null values")
                return [], [], mean_val, std_val, False

            min_vals = np.linspace(
                start_min_val, min_val, num_trials
            ).tolist()  # np.arange(start_min_val, min_val, abs( min_val - start_min_val) / HALF_MAX_TRIALS_PER_PARAM )
            max_vals = np.linspace(
                start_max_val, max_val, num_trials
            ).tolist()  # np.arange(max_val,start_max_val, abs( start_max_val - max_val) / HALF_MAX_TRIALS_PER_PARAM )[1:][::-1]
        else:
            mid_point = abs(max_val - min_val) / 2
            min_vals = np.linspace(
                min_val, mid_point, num_trials
            ).tolist()  # np.arange(min_val, mid_point, abs( mid_point - min_val) / HALF_MAX_TRIALS_PER_PARAM )
            max_vals = np.linspace(
                mid_point, max_val, num_trials
            ).tolist()  # np.arange(mid_point, max_val,  abs( max_val - mid_point) / HALF_MAX_TRIALS_PER_PARAM )[1:][::-1]

        if std_val <= 1e-6:
            std_val: float = global_stat_dict["std"]
            if std_val <= 1e-6:
                if verbose:
                    logging.info("Skipped due to zero std in both local and global setting! Use keep strategy and null values!")
                return min_vals, max_vals, mean_val, std_val, False

        return min_vals, max_vals, mean_val, std_val, True

    assert blueprint_yaml_path[-4:] == "yaml"
    assert report_json_path[-4:] == "json"

    tuned_config = SimConfig().parse_args([])
    tuned_config._from_yaml(blueprint_yaml_path, unsafe_load=True)

    with open(report_json_path, "r") as f:
        report_dict = json.load(f, object_pairs_hook=OrderedDict)

    wdn_name = os.path.basename(tuned_config.inp_paths[0])[:-4]

    wn = wntr.network.WaterNetworkModel(tuned_config.inp_paths[0])
    # call before init to gather unchanged metrics from the baseline
    baseline_demand = collect_all_params(
        frozen_wn=wn,
        time_from="wn",
        config=None,
        sim_output_keys=["demand"],
        output_only=True,
        duration=None,
        time_step=1,
    )["demand"].flatten()
    baseline_dmd_ubiqr = upper_bound_IQR(baseline_demand)  # np.quantile(baseline_demand, 0.75)

    wn.reset_initial_values()
    wn = init_water_network(
        wn,
        duration=tuned_config.duration,
        time_step=tuned_config.time_step,
        remove_controls=True,
        remove_curve=False,
        remove_patterns=False,
    )

    all_params = list(report_dict.keys())

    if len(custom_order_keys) <= 0:
        filtered_params = [param_name for param_name in all_params if wdn_name in report_dict[param_name]]

        logger.info("Filtering...")
        logger.info(f"Filtered / total: {len(filtered_params)} / {len(all_params)}")
    else:
        filtered_params = custom_order_keys

    if len(custom_skip_keys) > 0:
        # respect order while skipping
        filtered_params = [k for k in filtered_params if k not in custom_skip_keys]

    logger.info(f"Selected: {filtered_params}")

    MAX_TRIALS_PER_PARAM = max_iters
    STRATEGY_SWITCHING_TIME: int = int(MAX_TRIALS_PER_PARAM * switching_factor)

    DEFAULT_FREQUENCY = 365
    DEFAULT_SEASONAL = 0
    perturb_base = 4
    # sampling_base =2

    # in-order param selection
    failed_params = []
    for index, compo_param in enumerate(filtered_params):
        if compo_param[-2:] == "_x":  # we will proceed _y
            continue

        logger.info("*" * 80)
        logger.info(f"Tuning {compo_param}...")

        if compo_param in skip_compo_params:
            logger.info(f"Skipped! Due to compo_param {compo_param} in the defined sipped list")
            continue

        component = str(compo_param).split("+")[0].lower()
        param = str(compo_param).split("+")[1]

        if param[-2:] in ["_y", "_x"]:
            param = param[:-2]

        is_curve = param in get_curve_parameters()
        # is_pattern = param in get_pattern_parameters()

        if not reinforce_params:
            tmp_tune = getattr(tuned_config, f"{component}_tune")
            tmp_vals = getattr(tmp_tune, f"{param}_values")
            if tmp_vals is not None:
                logger.info("Skipped due to non-null values")
                continue

        stat_dict = report_dict[compo_param][wdn_name]
        global_stat_dict = report_dict[compo_param]["global"]

        assert stat_dict
        assert global_stat_dict

        if not is_curve:
            min_vals, max_vals, mean_val, std_val, is_valid = generate_values(
                stat_dict=stat_dict,
                global_stat_dict=global_stat_dict,
                num_trials=STRATEGY_SWITCHING_TIME if param != "base_demand" else MAX_TRIALS_PER_PARAM,
                verbose=True,
            )
            if not is_valid:
                continue
        else:
            num_points = math.floor(stat_dict["len"] / stat_dict["num_objects"])  # int(stat_dict['len'] / stat_dict['num_objects'])
            min_vals_y, max_vals_y, mean_val_y, std_val_y, is_valid = generate_values(
                stat_dict=stat_dict,
                global_stat_dict=global_stat_dict,
                num_trials=STRATEGY_SWITCHING_TIME,
                verbose=True,
            )
            if not is_valid:
                continue

            # get x
            compo_param_x = str(compo_param).replace("_y", "_x")
            stat_dict_x = report_dict[compo_param_x][wdn_name]
            global_stat_dict_x = report_dict[compo_param_x]["global"]
            min_vals_x, max_vals_x, mean_val_x, std_val_x, is_valid = generate_values(
                stat_dict=stat_dict_x,
                global_stat_dict=global_stat_dict_x,
                num_trials=STRATEGY_SWITCHING_TIME,
                verbose=True,
            )
            if not is_valid:
                continue

        is_found = False
        for trial in range(MAX_TRIALS_PER_PARAM):
            if param != "base_demand":
                if trial < STRATEGY_SWITCHING_TIME:
                    strategy = "sampling"
                    if not is_curve:
                        values = [min_vals[trial], max_vals[trial]]  # type:ignore
                    else:
                        values = [
                            min_vals_x[trial],  # type:ignore
                            max_vals_x[trial],  # type:ignore
                            min_vals_y[trial],  # type:ignore
                            max_vals_y[trial],  # type:ignore
                            num_points,  # type:ignore
                        ]
                else:
                    strategy = "perturbation"
                    if not is_curve:
                        values = [std_val / perturb_base ** (trial - STRATEGY_SWITCHING_TIME)]  # type:ignore
                    else:
                        values = [
                            std_val_x / perturb_base ** (trial - STRATEGY_SWITCHING_TIME),  # type:ignore
                            std_val_y / perturb_base ** (trial - STRATEGY_SWITCHING_TIME),  # type:ignore
                        ]  # type:ignore

            else:
                if trial < STRATEGY_SWITCHING_TIME:
                    strategy = junc_demand_strategy
                    if junc_demand_strategy == "adg":
                        values = [DEFAULT_SEASONAL, DEFAULT_FREQUENCY, max_vals[trial]]  # type:ignore
                    else:
                        values = [max_vals[trial]]  # type:ignore

                else:
                    strategy = "perturbation"
                    values = [std_val / perturb_base ** (trial - STRATEGY_SWITCHING_TIME)]  # type:ignore

            trial_config = deepcopy(tuned_config)
            component_tune = getattr(trial_config, f"{component}_tune")
            setattr(component_tune, f"{param}_strategy", strategy)
            setattr(component_tune, f"{param}_values", values)
            # output_path

            with tempfile.TemporaryDirectory() as output_temp_path:
                trial_config.output_path = output_temp_path
                trial_config.verbose = False
                # trial_config.num_cpus = 1
                output_paths = generate(trial_config)
                success_runs = get_success_runs(output_path=output_paths[0], config=trial_config)

                g = zarr.open_group(output_paths[0], mode="r")
                if "demand" in g.keys():
                    gen_dmd_ubiqr = upper_bound_IQR(
                        g["demand"][:]  # type:ignore
                    )  # np.quantile(g['demand'][:].flatten(), 0.75) # type:ignore
                else:
                    gen_dmd_ubiqr = 0.0

            sucess_ratio = float(success_runs) / trial_config.num_samples

            if (
                sucess_ratio >= acceptance_lo_threshold
                and sucess_ratio <= acceptance_up_threshold
                and (gen_dmd_ubiqr >= baseline_dmd_ubiqr or relax_q3_condition)
            ):
                is_found = True
                tuned_component_tune = getattr(tuned_config, f"{component}_tune")

                setattr(tuned_component_tune, f"{param}_strategy", strategy)
                setattr(tuned_component_tune, f"{param}_values", ndarray2list(values))

                setattr(tuned_config, f"{component}_tune", tuned_component_tune)

                logger.info(
                    f"Trial {trial+1} / {max_iters} on param {compo_param}: SUCCESS! Success Ratio: {sucess_ratio}! Gen UB IQR: {gen_dmd_ubiqr} >= BL UB IQR: {baseline_dmd_ubiqr}  | Strategy: {strategy} | values = {values}"
                )
                rp_tag = "_rp" if reinforce_params else ""
                new_yaml_path = blueprint_yaml_path[:-5] + f"_{index}" + f"{rp_tag}" + ".yaml"
                tuned_config._to_yaml(new_yaml_path)
                break
            else:
                logger.info(
                    f"Trial {trial+1} / {max_iters} on param {compo_param}: FAIL! Success Ratio: {sucess_ratio}! Gen UB IQR: {gen_dmd_ubiqr} < BL UB IQR: {baseline_dmd_ubiqr}  | Strategy: {strategy} | values = {values}"
                )

        if not is_found:
            failed_params.append(compo_param)
            logger.info(f"Failed to find optimum value for param {compo_param}")

    new_yaml_path = blueprint_yaml_path[:-5] + "_final" + ".yaml"
    tuned_config._to_yaml(new_yaml_path)

    if len(failed_params) > 0:
        logger.info(f"{tmp_name}: Params that were failed to find the optimal values: {failed_params}")
    return new_yaml_path


def compute_stats_and_beauty_print(flatten_arr: np.ndarray, indent: int = 16, precision: int = 8) -> None:
    """The flatten arr should fit your RAM.
    Do not use this function for too large array. Consider `summary2` function for large case.

    Args:
        flatten_arr (np.ndarray): a flatten arr
        indent (int, optional): indent space. Defaults to 16.
        precision (int, optional): floating point precision. Defaults to 8.
    """
    mean, std = np.mean(flatten_arr), np.std(flatten_arr)
    min, max = np.min(flatten_arr), np.max(flatten_arr)
    q25, q75 = np.quantile(flatten_arr, 0.25), np.quantile(flatten_arr, 0.75)
    q05, q99 = np.quantile(flatten_arr, 0.05), np.quantile(flatten_arr, 0.99)
    lb_iqr, ub_iqr = lower_bound_IQR(flatten_arr), upper_bound_IQR(flatten_arr)
    cv = std / mean
    mean_coef = np.corrcoef(flatten_arr)

    my_dict: dict[str, Any] = {
        "Mean": mean,
        "Std": std,
        "Min": min,
        "Max": max,
        "CV": cv,
        "MCoef": mean_coef,
        "Q25": q25,
        "Q75": q75,
        "Q05": q05,
        "Q99": q99,
        "LB IQR": lb_iqr,
        "UB IQR": ub_iqr,
    }
    beauty_print(my_dict=my_dict, indent=indent, precision=precision)


def beauty_print(my_dict: dict[str, Any], indent: int = 16, precision: int = 8) -> None:
    for k, v in my_dict.items():
        if isinstance(v, np.ndarray):
            print(f"{k}:   {v.item():>{indent}.{precision}f}")
        else:
            print(f"{k}:   {v:>{indent}.{precision}f}")


def summary(arr: zarr.Array | np.ndarray, config: SimConfig, name: str | None) -> None:
    """Summary information and compute statistic\\
    DO NOT USE THIS FUNCTION FOR LARGE ZARR.ARRAY! This blows out your memory!\\
    In the large ase, you should use `summary2` instead

    Args:
        arr (zarr.Array | np.ndarray): support zarray and nparray
        config (SimConfig): simulation config
        name (str | None): name of your dataset.
    """
    if isinstance(arr, zarr.Array):
        my_arr: np.ndarray = arr[:]  # TODO: <- IT MUST CAUSE OOM
    else:
        my_arr = arr

    flatten_arr = my_arr.flatten()
    if name is None:
        name = ""
    print("#" * 40 + f"{name}" + "#" * 40)

    success = my_arr.shape[0]
    total = config.num_samples
    indent = 16
    precision = 8
    if success > 0:
        compute_stats_and_beauty_print(flatten_arr, indent, precision)
        print(f"Succes/ Total: {success}/{total}")
    else:
        print("Array is empty!")

    print("#" * 80)


def summary2(
    arr: zarr.Array | np.ndarray,
    config: SimConfig,
    attrs: dict[str, Any],
    name: str | None,
    chunk_limit: str = "1024 MiB",
    exclude_skip_nodes_from_config: bool = True,
) -> None:
    import dask.array.core as dac
    from dask.array.percentile import percentile
    from dask.base import compute

    DYNAMIC_PARAMS = [
        "pressure",
        "head",
        "demand",
        "flowrate",
        "velocity",
        "headloss",
        "friction_factor",
        "reservoir_head_pattern_name",
        "junction_base_demand",
        "powerpump_base_speed",
        "powerpump_energy_pattern",
        #'headpump_base_speed',
        "headpump_energy_pattern",
    ]

    def get_component(attr: str) -> tuple[str, bool]:
        has_component = "_" in attr
        is_dynamic = is_node_simulation_output(attr)
        component = attr.split("_")[0] if has_component else "node" if is_dynamic else "link"
        return component, is_dynamic

    dask_arr: dac.Array = dac.from_zarr(url=arr)

    success = arr.shape[0]
    total = config.num_samples
    indent = 30
    precision = 8
    epsilon = 1e-8

    if success > 0:
        print(f"Succes/ Total: {success}/{total}")

        if exclude_skip_nodes_from_config:
            assert name is not None
            skip_names: list[str] = config.skip_names

            component, is_dynamic = get_component(attr=name)

            object_names = attrs["onames"][component]

            node_mask: np.ndarray = np.asarray([name not in skip_names for name in object_names], dtype=bool)

            if is_dynamic or component in DYNAMIC_PARAMS or component in get_curve_parameters():
                num_samples = dask_arr.shape[0]
                num_objects = len(object_names)
                dask_arr = dask_arr.reshape((num_samples, -1, num_objects), limit=chunk_limit)

            if not is_node(component):
                adj_list: list[tuple[str, str, str]] = attrs["adj_list"]
                edge_mask: np.ndarray = np.asarray(
                    [not set([src_node, dst_node]).isdisjoint(skip_names) for src_node, dst_node, _ in adj_list], dtype=bool
                )
                dask_arr = dask_arr[..., edge_mask]
            else:
                dask_arr = dask_arr[..., node_mask]

        # flatten the array with limit memory
        flatten_arr = dask_arr.reshape((-1), limit=chunk_limit)
        # since we don't have built-in quantile function, we replace it with percentile
        # Note that percentile divide data to 100, while quantile divide data to a number of equal-size subsets.

        my_dict: dict[str, Any] = {
            "Mean": flatten_arr.mean(),
            "Std": flatten_arr.std(),
            "Min": flatten_arr.min(),
            "Max": flatten_arr.max(),
            "Q25": percentile(flatten_arr, q=25)[0],
            "Q75": percentile(flatten_arr, q=75)[0],
            "Q05": percentile(flatten_arr, q=5)[0],
            "Q99": percentile(flatten_arr, q=99)[0],
        }

        (ret_dict,) = compute(my_dict)
        ret_dict = {k: float(v) for k, v in ret_dict.items()}
        mean, std, q25, q75 = ret_dict["Mean"], ret_dict["Std"], ret_dict["Q25"], ret_dict["Q75"]

        cv = std / (mean + epsilon)

        lb_iqr, ub_iqr = internal_lower_bound_IQR(q25, q75), internal_upper_bound_IQR(q25, q75)

        ret_dict["CV"] = cv
        ret_dict["LB IQR"] = lb_iqr
        ret_dict["UB IQR"] = ub_iqr

        beauty_print(my_dict=ret_dict, indent=indent, precision=precision)

    else:
        print("Array is empty!")

    print("#" * 80)


def summary3(
    arr: zarr.Array | np.ndarray,
    config: SimConfig,
    attrs: dict[str, Any],
    name: str | None,
    chunk_limit: str = "1024 MiB",
    exclude_skip_nodes_from_config: bool = True,
) -> None:
    import dask.array.core as dac
    from dask.array.percentile import percentile
    from dask.base import compute

    DYNAMIC_PARAMS = [
        "pressure",
        "head",
        "demand",
        "flowrate",
        "velocity",
        "headloss",
        "friction_factor",
        "reservoir_head_pattern_name",
        "junction_base_demand",
        "powerpump_base_speed",
        "powerpump_energy_pattern",
        #'headpump_base_speed',
        "headpump_energy_pattern",
    ]

    def get_component(attr: str) -> tuple[str, bool]:
        has_component = "_" in attr
        is_dynamic = is_node_simulation_output(attr)
        component = attr.split("_")[0] if has_component else "node" if is_dynamic else "link"
        return component, is_dynamic

    dask_arr: dac.Array = dac.from_zarr(url=arr)

    success = arr.shape[0]
    total = config.num_samples
    indent = 30
    precision = 8
    epsilon = 1e-8

    if success > 0:
        from gigantic_dataset.core.datasets_large import GidaV5

        print(f"Succes/ Total: {success}/{total}")
        component, is_dynamic = get_component(attr=name)  # type:ignore
        node_attrs = []
        edge_attrs = []
        if is_node(component):
            node_attrs.append(name)
        else:
            node_attrs.append("demand")
            edge_attrs.append(name)

        if exclude_skip_nodes_from_config:
            assert name is not None
            skip_names: list[str] = config.skip_names

            component, is_dynamic = get_component(attr=name)

            object_names = attrs["onames"][component]

            tmp_masks = [name not in skip_names for name in object_names]
            node_mask: np.ndarray = np.asarray(tmp_masks, dtype=bool)
            # filtered_object_names = list(compress(object_names, tmp_masks))
            if is_dynamic or component in DYNAMIC_PARAMS or component in get_curve_parameters():
                num_samples = dask_arr.shape[0]
                num_objects = len(object_names)
                dask_arr = dask_arr.reshape((num_samples, -1, num_objects), limit=chunk_limit)

            if not is_node(component):
                adj_list: list[tuple[str, str, str]] = attrs["adj_list"]
                edge_mask: np.ndarray = np.asarray(
                    [not set([src_node, dst_node]).isdisjoint(skip_names) for src_node, dst_node, _ in adj_list], dtype=bool
                )
                dask_arr = dask_arr[:, :, edge_mask]
            else:
                dask_arr = dask_arr[:, :, node_mask]
            # print(f"dask_arr shape = {dask_arr.shape}")
            # print(f"node_mask shape = {node_mask.shape}")

            # testing_arr = dask_arr.reshape([dask_arr.shape[0] * dask_arr.shape[1], 35], limit=chunk_limit)

            # dask_arr_argmin = testing_arr.argmin(axis=1)
            # dask_arr_argmin = dask_arr_argmin.compute()
            # print(f"dask_arr_argmin = {dask_arr_argmin}")

        flatten_arr = dask_arr.reshape((-1), limit=chunk_limit)

        my_dict: dict[str, Any] = {
            "Mean": flatten_arr.mean(),
            "Std": flatten_arr.std(),
            "Min": flatten_arr.min(),
            "Max": flatten_arr.max(),
            "Q25": percentile(flatten_arr, q=25)[0],
            "Q75": percentile(flatten_arr, q=75)[0],
            "Q05": percentile(flatten_arr, q=5)[0],
            "Q99": percentile(flatten_arr, q=99)[0],
        }

        (ret_dict,) = compute(my_dict)

        ret_dict = {k: float(v) for k, v in ret_dict.items()}
        mean, std, q25, q75 = ret_dict["Mean"], ret_dict["Std"], ret_dict["Q25"], ret_dict["Q75"]

        cv = std / (mean + epsilon)

        lb_iqr, ub_iqr = internal_lower_bound_IQR(q25, q75), internal_upper_bound_IQR(q25, q75)

        ret_dict["CV"] = cv
        ret_dict["LB IQR"] = lb_iqr
        ret_dict["UB IQR"] = ub_iqr

        beauty_print(my_dict=ret_dict, indent=indent, precision=precision)

    else:
        print("Array is empty!")

    print("#" * 80)


def report(
    output_path: str,
    only_keys: list[str] = ["demand", "pressure"],
    config: SimConfig | None = None,
    report_baseline: bool = False,
    report_all_params: bool = False,
    chunk_limit: str = "1 GB",
):
    """printout useful statistic. Support large dataset. \\
    If it takes too long, consider increasing chunk limit to 10,240 MiB (10GB).\\
    Require: dask

    Args:
        output_path (str): path to zarr or zip file
        only_keys (list[str]): only reported keys if this list is non-empty. Defauts to [demand, pressure]
        config (SimConfig | None, optional): simulation config; if None, we use attrs from zarr file. Defaults to None.
        report_baseline (bool, optional): Indicate whether we report baseline INP file. We assume that the baseline is stored at config.input_path. Defaults to False.
        report_all_params (bool, optional): Flag indicates whether we report both in/out hydraulic parameters. Defaults to False.
        chunk_limit (str, optional): expected size of a chunk in mb to load very large dataset. Defaults to "1 GB".
    """
    assert output_path[-4:] == ".zip" or output_path[-4:] == "zarr"
    group = zarr.open_group(output_path, mode="r")
    arr_names = list(group.array_keys())  # type:ignore
    attrs: dict[str, Any] = group.attrs.asdict()  # type:ignore
    print(f"attrs = {attrs}")
    if len(arr_names) > 0:
        print(f"Considering... {arr_names}")

        if config is None:
            # we get it from the group.attrs
            config = SimConfig()
            config._parsed = True
            config = config.from_dict(attrs, True)

        for arr_name in arr_names:
            if arr_name in config.sim_outputs or report_all_params:
                if len(only_keys) <= 0 or arr_name in only_keys:
                    arr: zarr.Array | zarr.Group = group[arr_name]
                    assert isinstance(arr, zarr.Array)
                    print(f"{arr_name}: arr shape = {arr.shape}")
                    summary3(
                        arr=arr,
                        config=config,
                        attrs=attrs,
                        name=arr_name,  # type:ignore
                        chunk_limit=chunk_limit,
                        exclude_skip_nodes_from_config=True,
                    )
                    # summary(arr=arr, config=config, name=arr_name)  # type:ignore
        if report_baseline:
            print("@" * 80 + "BASELINE" + "@" * 80)
            assert len(config.inp_paths) == 1
            baseline_inp_path = config.inp_paths[0]
            baseline_param_dict = collect_all_params(
                frozen_wn=wntr.network.WaterNetworkModel(baseline_inp_path),
                time_from="wn",
                config=config,
                sim_output_keys=config.sim_outputs,
                exclude_skip_nodes_from_config=True,
                output_only=True,
                duration=None,
                time_step=1,
            )
            indent = 16
            precision = 8
            for k in baseline_param_dict:
                if k in config.sim_outputs or report_all_params:
                    if len(only_keys) <= 0 or k in only_keys:
                        print("#" * 40 + "BASELINE-" + f"{k}" + "#" * 40)
                        compute_stats_and_beauty_print(baseline_param_dict[k].flatten(), indent, precision)
    else:
        print("Empty dataset!!!")


def get_success_runs(output_path: str, config: SimConfig) -> int:
    """Only for small dataset

    Args:
        output_path (str): path of the output file
        config (SimConfig: simulation config

    Returns:
        int: number of successful runs
    """
    assert output_path[-4:] == ".zip" or output_path[-4:] == "zarr"

    def contain_sim_outputs(
        checking_key: str,
        output_list: list[
            Literal[
                "pressure",
                "head",
                "demand",
                "flowrate",
                "velocity",
                "headloss",
                "friction_factor",
            ]
        ],
    ) -> bool:
        for output in output_list:
            if output in checking_key:
                return True
        return False

    group = zarr.open_group(output_path, mode="r")
    arr_names = list(group.array_keys())  # type:ignore
    for arr_name in arr_names:
        if contain_sim_outputs(arr_name, config.sim_outputs):  # type:ignore
            arr = group[arr_name][:]  # type:ignore
            if arr is not None:
                return arr.shape[0]  # type:ignore
    return 0


def collect_input_params(
    time_from: Literal["wn", "config"],
    frozen_wn: WaterNetworkModel | None = None,
    input_path: str = "",
    config: SimConfig | None = None,
    duration: int | None = None,
    time_step: int | None = 1,
    exclude_skip_nodes_from_config: bool = False,
    skip_names: list[str] = [],
    edge_skip_names: list[str] = [],
) -> Generator[tuple[str, str, np.ndarray, int], None, None]:
    """
    Return a generator that yields a tuple of (component_name, param_name, param_values, num_objs)
    Args:
        time_from (Literal[&quot;wn&quot;, &quot;config&quot;]): when duration or time_step is unavailable, we gather them based on 1) wn or 2) config
        frozen_wn (WaterNetworkModel): water network model instance. If None, load it from input_path
        input_path (str): optional. Only be used (and should be not None) if frozen_wn is None.
        config (SimConfig | None): simulation config. if time_from == 'config', it should be not None
        duration (int | None): if None, auto-pick based on time_from. Default is None
        time_step (int | None): if None, auto-pick based on time_from. Default is 1
        skip_nodes_from_config (bool): if True, we exclude skip_nodes from config. Else, we don't perform skipping
        skip_names (list[str]): if empty, we auto-pick config.skip_nodes
        edge_skip_names (list[str]): if empty, we auto-pick based on skip_names and wn
    Yields:
        Generator[tuple[str, str, np.ndarray, int], None, None]: a tuple of (component_name, param_name, param_values, num_objs)
    """
    if frozen_wn is None:
        assert input_path != ""
        wn = WaterNetworkModel(input_path)
    else:
        wn = deepcopy(frozen_wn)

    duration = get_duration_from(wn=wn, config=config, time_from=time_from) if duration is None else duration
    time_step = get_time_step_from(wn=wn, config=config, time_from=time_from) # if time_step is None else time_step

    if duration is None or duration <= 0:
        duration = time_step

    wn = init_water_network(
        wn,
        duration=duration,
        time_step=time_step,
        remove_controls=True,
        remove_curve=False,
        remove_patterns=False,
    )
    time_dim = duration

    # compo_mask_dict: dict[str, list[bool]] = {}
    union_skip_names: list[str] = []
    if exclude_skip_nodes_from_config:
        assert config is not None
        if len(skip_names) <= 0:
            skip_names = config.skip_names
        if len(edge_skip_names) <= 0:
            for skip_node_name in skip_names:
                link_names: list[str] = wn.get_links_for_node(skip_node_name)
                edge_skip_names.extend(link_names)
        union_skip_names = list(set(skip_names).union(edge_skip_names))

    if config is None:
        config = SimConfig()
        config._parsed = True

    tune_list = [field for field in config._to_dict() if "tune" in field]
    # tune_list = [field for field in list(vars(config)) if "tune" in field]

    for tune_name in tune_list:
        tune_config = getattr(config, tune_name)
        param_names = ["_".join(field.name.split("_")[:-1]) for field in fields(tune_config) if "strategy" in field.name]
        component_name = tune_name.replace("_tune", "")
        for param_name in param_names:
            is_curve = param_name in get_curve_parameters()
            is_pattern = param_name in get_pattern_parameters()
            obj_tup_list: list[tuple] = list(get_object_dict_by_config(tune_config, wn))  # type:ignore

            old_records = get_old_records(
                obj_dict=obj_tup_list,  # type:ignore
                wn=wn,
                param_name=param_name,
                time_dim=time_dim,
                is_curve=is_curve,
                is_pattern=is_pattern,
                skip_names=union_skip_names,
            )

            if len(old_records) <= 0:
                # print(f'WDN: {wdn_name} | Component: {tune_name.replace("_tune","")} | param: {param_name} has none values!')
                continue

            if is_curve:
                param_values_x = []
                param_values_y = []
                for t in old_records:
                    half_length = len(t) // 2
                    xs = t[0:half_length]
                    ys = t[half_length:]
                    assert len(xs) == len(ys)
                    param_values_x.append(xs)
                    param_values_y.append(ys)

                param_values_x = np.concatenate(param_values_x)

                yield (component_name, param_name + "_x", param_values_x, len(obj_tup_list))
                param_values_y = np.concatenate(param_values_y)
                yield (component_name, param_name + "_y", param_values_y, len(obj_tup_list))

            else:
                param_values = np.concatenate(old_records)
                yield (component_name, param_name, param_values, len(obj_tup_list))


def collect_all_input_params(
    input_path: str,
    config: SimConfig,
    time_from: Literal["wn", "config"],
    duration: int | None = None,
    time_step: int | None = 1,
    exclude_skip_nodes_from_config: bool = False,
) -> dict[str, np.ndarray]:
    results_dict = {}
    for component_name, param_name, param_values, num_objs in collect_input_params(
        input_path=input_path,
        frozen_wn=None,
        time_from=time_from,
        config=config,
        duration=duration,
        time_step=time_step,
        exclude_skip_nodes_from_config=exclude_skip_nodes_from_config,
    ):
        key = component_name.replace("_", "") + "_" + param_name
        results_dict[key] = param_values

    return results_dict


def get_duration_from(
    wn: WaterNetworkModel | None,
    config: SimConfig | None,
    time_from: Literal["config", "wn"] = "config",
) -> int:
    """To gather duration, we decide based on time_from.
    If time_from == config, we gather duration from config
    else
        we retrieve maximum length of demand patterns and choose it as duration.
        In case there has no pattern, we choose duration in the network settings.

    Args:
        wn (WaterNetworkModel | None): _description_
        config (SimConfig | None): _description_
        time_from (Literal[&quot;config&quot;, &quot;wn&quot;], optional): _description_. Defaults to "config".

    Raises:
        NotImplementedError: _description_

    Returns:
        int: _description_
    """
    if time_from == "config":
        assert config is not None
        return config.duration
    else:
        assert wn is not None
        # first choose duration based on pattern max length
        pattern_lengths = []
        for p_name in wn.pattern_name_list:
            p = wn.patterns[p_name]
            if isinstance(p, wntre.Pattern):
                p_length = len(p.multipliers)
                pattern_lengths.append(p_length)
            else:
                raise NotImplementedError(f"pattern type {type(p)} is not supported")

        if len(pattern_lengths) > 0:
            return max(pattern_lengths)
        else:
            # if none of patterns, we just take duration in wn settings.
            return wn.options.time.duration


def get_time_step_from(
    wn: WaterNetworkModel | None,
    config: SimConfig | None,
    time_from: Literal["wn", "config"] = "config",
) -> int:
    if time_from == "wn":
        assert wn is not None
        return wn.options.time.hydraulic_timestep
    elif time_from == "config":
        assert config is not None
        return config.time_step


def collect_all_params(
    frozen_wn: WaterNetworkModel,
    time_from: Literal["wn", "config"],
    config: SimConfig | None = None,
    sim_output_keys: list = [
        "pressure",
        "head",
        "demand",
        "flowrate",
        "velocity",
        "headloss",
        "friction_factor",
    ],
    exclude_skip_nodes_from_config: bool = True,
    output_only: bool = True,
    duration: int | None = None,
    time_step: int | None = 1,
) -> dict:
    """
    Args:
        frozen_wn (WaterNetworkModel): instance of WDNs
        time_from (Literal[&quot;wn&quot;, &quot;config&quot;], optional): This indicates source to gather time information (duration and time_step).
        config (SimConfig) : simulation config.
        sim_output_keys (list, optional): selective keys. Defaults to [ "pressure", "head", "demand", "flowrate", "velocity", "headloss", "friction_factor", ].
        exclude_skip_nodes_from_config (bool): eliminate skip_nodes and adjacent edges gathered from config in collecting params. Config must be not None.
        output_only (bool): flag indicates whether it only collect outputs or both output and input
        duration (int | None): If none, auto-pick w.r.t. time_from
        time_step (int | None): if none, auto-pick w.r.t. time_from

    Returns:
        dict: dict contains results of selected sim_output_keys
    """
    wn = deepcopy(frozen_wn)
    wn.reset_initial_values()

    if exclude_skip_nodes_from_config:
        assert config is not None, "ERROR! Config must exist to gather skip nodes, but it is None"
        skip_names: list[str] = config.skip_names
        edge_skip_names = []
        for skip_node_name in skip_names:
            link_names: list[str] = wn.get_links_for_node(skip_node_name)
            edge_skip_names.extend(link_names)
    else:
        skip_names: list[str] = []
        edge_skip_names: list[str] = []

    results_dict = {}
    if not output_only:
        for component_name, param_name, param_values, num_objs in collect_input_params(
            frozen_wn=wn,
            time_from=time_from,
            config=config,
            duration=duration,
            time_step=time_step,
            exclude_skip_nodes_from_config=exclude_skip_nodes_from_config,
            skip_names=skip_names,
            edge_skip_names=edge_skip_names,
        ):
            key = component_name.replace("_", "") + "_" + param_name
            results_dict[key] = param_values

    # gather outputs
    sim = wntr.sim.EpanetSimulator(wn=wn)
    with tempfile.TemporaryDirectory(prefix="explore-dir-temp") as temp_dir_name:
        results: wntr.sim.SimulationResults = sim.run_sim(file_prefix=temp_dir_name, version=2.2)
        for k in results.node:  # type:ignore
            if k in sim_output_keys:
                df: pd.DataFrame = results.node[k]  # type:ignore
                if len(skip_names) > 0:
                    df = df.drop(skip_names, axis=1, errors="ignore")  # type:ignore
                results_dict[k] = df.to_numpy()

        for k in results.link:  # type:ignore
            if k in sim_output_keys:
                df: pd.DataFrame = results.link[k]  # type:ignore
                if len(edge_skip_names) > 0:
                    df = df.drop(edge_skip_names, axis=1, errors="ignore")  # type:ignore
                results_dict[k] = df.to_numpy()
    return results_dict


def collect_global_statistic_data(
    inp_paths: list[str] = [],
    configs: list[SimConfig] = [],
    time_from: Literal["wn", "config"] = "wn",
    export_path: str = "profiler_report.json",
    duration: int | None = None,
    time_step: int | None = 1,
    exclude_skip_nodes_from_config: bool = False,
):
    # time_dim = 1 if config.duration <= 1 else config.duration
    # tune_list = [field for field in list(vars(config)) if "tune" in field]

    if len(inp_paths) <= 0:
        assert len(configs) > 0
        inp_paths = [config.inp_paths[0] for config in configs]

    assert len(inp_paths) > 0

    profiler = WDNProfiler(None)  # saved in mem
    num_valid_networks = 0
    for i, inp_path in enumerate(inp_paths):
        wdn_name = os.path.basename(inp_path)[:-4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_valid = False
            for component_name, param_name, param_values, num_objs in collect_input_params(
                frozen_wn=None,
                input_path=inp_path,
                time_from=time_from,
                config=configs[i] if len(configs) > i else None,
                duration=duration,
                time_step=time_step,
                exclude_skip_nodes_from_config=exclude_skip_nodes_from_config,
            ):
                profiler.collect(
                    param_name=param_name,
                    component_name=component_name,
                    wdn_name=wdn_name,
                    param_values=param_values,
                    num_objects=num_objs,
                    track_global=True,
                )
                is_valid = True

            if is_valid:
                num_valid_networks += 1

    print(f"Number of valid networks = {num_valid_networks}")

    profiler.collect_globally(global_key="global")

    profiler.export(export_path)


def lookup_skip_nodes_and_parse_to_config(
    inp_path: str,
    config: SimConfig | None = None,
    lookup_words: list[str] = ["Pump"],
    skip_words: list[str] = [],
    skip_reservoir: bool = True,
    verbose: bool = True,
) -> SimConfig:
    if config is None:
        config = SimConfig()
        config._parsed = True
    wn = wntr.network.WaterNetworkModel(inp_path)
    pump_node_names = []
    for name in wn.junction_name_list:
        if any([w in name for w in lookup_words]):
            if len(skip_words) <= 0 or all([w not in name for w in skip_words]):
                pump_node_names.append(name)

    if pump_node_names:
        pump_node_names = sorted(pump_node_names)
        if verbose:
            for name in pump_node_names:
                print(f"- {name}")
        config.skip_names.extend(pump_node_names)

    if skip_reservoir:
        config.skip_names.extend(wn.reservoir_name_list)

    return config


def check_wdns(folder_path: str, verbose: bool = True) -> tuple[list[str], list[str]]:
    inp_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name[-4:].lower() == ".inp"]
    failed_list = []
    success_list = []
    for inp in inp_paths:
        wdn_name = os.path.basename(inp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wntr.network.WaterNetworkModel(inp)
            if verbose:
                print(f"{wdn_name} is loaded successfully")
            success_list.append(inp)
        except AssertionError:
            if verbose:
                print(f"{wdn_name} : This is not input file")
            failed_list.append(inp)
        except Exception as e:
            if verbose:
                print(f"{wdn_name} : {e}")
            failed_list.append(wdn_name)
    return success_list, failed_list


def check_wdn_and_collect_stats(export_path: str = "profiler_report.json", wdn_folder_path: str = r"gigantic_dataset\inputs\public"):
    inp_paths, _ = check_wdns(wdn_folder_path)

    configs = [lookup_skip_nodes_and_parse_to_config(inp_path, skip_reservoir=False) for inp_path in inp_paths]

    collect_global_statistic_data(
        inp_paths=inp_paths,
        configs=configs,
        exclude_skip_nodes_from_config=True,
        export_path=export_path,
    )


def create_blueprint_config(
    inp_path: str,
    blueprint_path: str,
    report_json_path: Optional[str] = r"profiler_report_new.json",
    junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    init_demand_key: Literal["min", "max", "q3", "q1", "mean"] = "q3",
    junc_elevation_strategy: Literal["keep", "terrain"] = "terrain",
    pipe_diameter_strategy: Literal["keep", "substitute", "factor"] = "substitute",
    auto_add_skip_nodes: bool = False,
    **kwargs: Any,
):
    config = SimConfig().parse_args([])
    config.inp_paths = [
        os.path.relpath(inp_path),
    ]

    if kwargs is not None:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    if report_json_path is not None:
        with open(report_json_path) as file:
            report_dict: dict[str, Any] = json.load(file)

        network_name = os.path.basename(inp_path)[:-4]

        component = "junction"
        param = "base_demand"
        if junc_demand_strategy == "adg":
            junc_demand_values = [
                0,
                365,
                report_dict[f"{component}+{param}"][network_name][init_demand_key],
            ]
        elif junc_demand_strategy == "adg_v2":
            junc_demand_values = [report_dict[f"{component}+{param}"][network_name][init_demand_key]]
        else:
            raise NotImplementedError(f"junc_demand_strategy is not expected!Found junc_demand_strategy = {junc_demand_strategy}")

        component_tune = getattr(config, f"{component}_tune")
        setattr(component_tune, f"{param}_strategy", junc_demand_strategy)
        setattr(component_tune, f"{param}_values", junc_demand_values)

        if junc_elevation_strategy == "terrain":
            junc_ele_values = [
                4 + np.random.rand() * (25 - 4),  # noise in range [4,25]
                17,  # fixed terrain width
            ]

            setattr(component_tune, "elevation_strategy", junc_elevation_strategy)
            setattr(component_tune, "elevation_values", junc_ele_values)
        # pipe
        if pipe_diameter_strategy != "keep":
            component = "pipe"
            param = "diameter"
            if pipe_diameter_strategy == "substitute":
                pipe_diameter_values = [
                    0.2,  # max noise
                ]
            elif pipe_diameter_strategy == "factor":
                pipe_diameter_values = [
                    0.8,  # min scale
                    1.4,  # max scale
                    0.2,  # max noise
                ]
            component_tune = getattr(config, f"{component}_tune")
            setattr(component_tune, f"{param}_strategy", pipe_diameter_strategy)
            setattr(component_tune, f"{param}_values", pipe_diameter_values)

    if auto_add_skip_nodes:
        config = lookup_skip_nodes_and_parse_to_config(inp_path, config=config)
    config._to_yaml(yaml_path=blueprint_path)


def find_optimal_config_wrapper(
    strategy_fn: Callable,
    inp_path: str,
    folder_yaml_path: str = r"gigantic_dataset/arguments",
    report_json_path: str = r"gigantic_dataset/profiler_report_new.json",
    yaml_path: Optional[str] = None,
    reinforce_params: bool = False,
    max_iters: int = 2,
    population_size: int = 10,
    num_cpus: int = 10,
    junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    **kwargs: Any,
) -> str:
    assert os.path.exists(report_json_path), f"{report_json_path} is not existed"

    if yaml_path is None:
        base_name = os.path.basename(inp_path)[:-4]
        file_name = f"{base_name}.yaml"
        yaml_path = os.path.join(folder_yaml_path, file_name)
        assert os.path.isdir(folder_yaml_path)
        create_blueprint_config(
            inp_path=inp_path,
            blueprint_path=yaml_path,
            report_json_path=report_json_path,
            alter_demand_if_null=True,
            init_demand_key="q3",
            # **kwargs,
        )
        blueprint_yaml_path = os.path.join(folder_yaml_path, file_name)
    else:
        # folder_yaml_path = os.path.dirname(yaml_path)
        # file_name = os.path.basename(yaml_path)
        # base_name = file_name.split('_')[0]
        blueprint_yaml_path = yaml_path

    latest_yaml_path = ""

    start = time.time()

    latest_yaml_path = strategy_fn(
        blueprint_yaml_path=blueprint_yaml_path,
        report_json_path=report_json_path,
        reinforce_params=reinforce_params,
        max_iters=max_iters,
        population_size=population_size,
        num_cpus=num_cpus,
        junc_demand_strategy=junc_demand_strategy,
        **kwargs,
    )
    gap = time.time() - start

    print(f"Finish after {gap} seconds. Latest yaml path: {latest_yaml_path}")

    return latest_yaml_path
