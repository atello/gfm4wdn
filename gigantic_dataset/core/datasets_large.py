#
# Created on Thu May 16 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: (Dec 12 2024) BACK UP OF DATASETS_LARGE.PY
# ------------------------------

import copy
import glob
import json
import math
import os
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from itertools import compress
from typing import Any, Literal, Optional, Sequence, Union

import dask.array.core as dac
import numpy as np
import pandas as pd
import torch
import zarr
from dask.base import compute
from torch import Tensor, from_numpy, save, load
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import to_undirected
from wntr.network import WaterNetworkModel

from gigantic_dataset.utils.auxil_v8 import (
    get_adj_list,
    get_object_name_list_by_component,
    is_node_simulation_output,
    shuffle_list,
    masking_list,
)
from gigantic_dataset.utils.pretrain_utils import compute_node_degree

LARGE_NUMBER = 10000

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def parse2data(x: np.ndarray, edge_index: np.ndarray, edge_attr: Optional[np.ndarray], y: Optional[np.ndarray],
               edge_y: Optional[np.ndarray], node_degrees: Optional[torch.Tensor]) -> Data:
    data_dict: dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = torch.as_tensor(edge_index, dtype=torch.long)
    data_dict["x"] = torch.as_tensor(x, dtype=torch.float)

    if edge_attr is not None:
        data_dict["edge_attr"] = torch.as_tensor(edge_attr, dtype=torch.float)
    if y is not None:
        data_dict["y"] = torch.as_tensor(y, dtype=torch.float)
    if edge_y is not None:
        data_dict["edge_y"] = torch.as_tensor(edge_y, dtype=torch.float)
    if node_degrees is not None:
        data_dict["x"] = torch.cat([data_dict["x"], node_degrees], dim=-1)

    data = Data.from_dict(data_dict)
    return data


class GidaV6(Dataset):
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
    # sugar-coating popular merge-able params (order insensitive)
    Node_Elevation = ("reservoir_base_head", "junction_elevation", "tank_elevation")
    Link_Initial_Status = (
        "pipe_initial_status",
        "power_pump_initial_status",
        "headpump_initial_status",
        "prv_initial_status",
        "tcv_initial_status",
    )  # some valves are missing as we haven't had them yet
    Pump_Curve = ("headpump_pump_curve_name", "powerpump_pump_curve_name")

    class Root:
        def get_file_type(self, zip_file_path: str) -> str:
            try:
                zarr.open(store=zip_file_path, mode="r")
            except Exception:
                return "csv"
            return "zarr"

        def _get_component(self, attr: str) -> str:
            has_component = "_" in attr
            component = attr.split("_")[0] if has_component else "node" if is_node_simulation_output(attr) else "link"
            return component

        def __init__(self, name: str, zip_file_path: str, num_cpus: Optional[int] = None) -> None:
            self.file_type = self.get_file_type(zip_file_path)
            self.zip_file_path = zip_file_path
            self.name = name
            if self.file_type == "zarr":
                self.root = zarr.open(store=zip_file_path, mode="r")
                assert isinstance(self.root, zarr.Group)
                self.attrs = self.root.attrs
                self.array_keys: list[str] = list(self.root.array_keys())  # type:ignore

            elif self.file_type == "csv":
                self.temp_folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    # Extract all the contents into the specified directory
                    zip_ref.extractall(self.temp_folder.name)
                self.root = {}
                self.array_keys: list[str] = [file_name[:-4] for file_name in os.listdir(self.temp_folder.name) if file_name[-4:] == ".csv"]
                json_paths = glob.glob(os.path.join(self.temp_folder.name, "*.json"))
                assert len(json_paths) > 0

                with open(json_paths[0]) as f:
                    self.attrs = json.load(f)

            self.time_dim: int = self.attrs["duration"] // self.attrs["time_step"]
            self.node_mask: Optional[np.ndarray] = None
            self.edge_mask: Optional[np.ndarray] = None
            self.adj_mask: Optional[np.ndarray] = None
            self.edge_index: Optional[np.ndarray] = None
            self.node_names: list[str] = []
            self.edge_names: list[str] = []
            self.num_cpus: Optional[int] = num_cpus

            self.node_components: list[str] = []
            self.edge_components: list[str] = []

            self.sorted_node_attrs: list[str | tuple] = []
            self.node_has_asterisks: list[bool] = []
            self.sorted_edge_attrs: list[str | tuple] = []
            self.edge_has_asterisks: list[bool] = []
            self.sorted_label_attrs: list[str | tuple] = []
            self.label_has_asterisks: list[bool] = []
            self.sorted_edge_label_attrs: list[str | tuple] = []
            self.edge_label_has_asterisks: list[bool] = []

        def get_array_by_key(self, key: str) -> zarr.Array:
            if isinstance(self.root, zarr.Group):
                assert key in self.root
                ret_arr = self.root[key]
                assert ret_arr is not None and isinstance(ret_arr, zarr.Array)
                ret_arr = ret_arr.astype(np.float32)

            elif isinstance(self.root, dict):
                if key not in self.root:
                    assert key in self.array_keys
                    csv_path = key + ".csv"
                    df = pd.read_csv(os.path.join(self.temp_folder.name, csv_path), sep=",", header=None)
                    chunks = (self.attrs["batch_size"], df.shape[-1])
                    self.root[key] = zarr.array(data=df.to_numpy(dtype=np.float32), chunks=chunks)

                ret_arr = self.root[key]
            else:
                raise NotImplementedError("root instance is unknown! ")

            return ret_arr

        def compute_first_size(self) -> int:
            if len(self.sorted_node_attrs) <= 0:
                return 0
            node_attr = self.sorted_node_attrs[0]
            if isinstance(node_attr, tuple):
                node_attr = node_attr[0]
            my_zarr = self.get_array_by_key(node_attr)
            num_samples = my_zarr.shape[0]
            assert num_samples is not None and isinstance(num_samples, int)
            return num_samples

    def __init__(
        self,
        zip_file_paths: list[str],
        node_attrs: list[str | tuple],
        edge_attrs: list[str | tuple] = [],
        label_attrs: list[str | tuple] = [],
        edge_label_attrs: list[str | tuple] = [],
        input_paths: list[str] = [],
        num_records: Optional[int] = None,
        selected_snapshots: Optional[int] = None,
        verbose: bool = True,
        split_type: Literal["temporal", "scene"] = "scene",
        split_set: Literal["train", "val", "test", "all"] = "all",
        split_fixed: bool = True,
        split_per_network: bool = True,
        skip_nodes_list: list[list[str]] = [],
        skip_types_list: list[list[str]] = [],
        unstackable_pad_value: Any = 0,
        bypass_skip_names_in_config: bool = False,
        chunk_limit: str = "120 GB",
        overwatch: bool = False,
        batch_axis_choice: Literal["scene", "temporal", "snapshot"] = "scene",
        num_cpus: int = 1,
        time_sampling_rate: int = 1,
        do_cache: bool = False,
        subset_shuffle: bool = False,
        dataset_log_pt_path: str = r"",
        run_mode: Literal[
            "pretrain", "train_default", "shared_enc", "pretrain-inference", "dual_encoder"] = "train_default",
        sampling_strategy: Literal["default", "random", "subsampling"] = "default",
        **kwargs,
    ) -> None:
        """Correponding to configs.py/GiDaConfig.

        Args:
            zip_file_paths (list[str]): where you load .zip (.zarr) arrays. Support multi-network.
            node_attrs (list[str  |  tuple]): node attributes. if str, we query directly. Otherwise, we hstack attributes. Support (*) operator.
            edge_attrs (list[str  |  tuple], optional): edge attributes. Support (*) operator. Defaults to [].
            label_attrs (list[str  |  tuple], optional): label attributes. Support (*) operator. Defaults to [].
            edge_label_attrs (list[str  |  tuple], optional): edge label (only used when label is non-empty). Support (*) operator. Defaults to [].
            input_paths (list[str], optional): To query networks' basis info. Defaults to [].
            num_records (Optional[int], optional): Set None to use all. Defaults to None.
            selected_snapshots (Optional[int], optional): Deprecated. Defaults to None.
            verbose (bool, optional): flag indicates whether logging debug info. Defaults to True.
            split_type (Literal[&quot;temporal&quot;, &quot;scene&quot;], optional): Deprecated. Defaults to "scene".
            split_set (Literal[&quot;train&quot;, &quot;val&quot;, &quot;test&quot;, &quot;all&quot;], optional): to split subsets. Defaults to "all".
            split_per_network (bool, optional): If True, foreach network, we split train, valid, test individually (Useful for multiple network joint-training). Otherwise, we concatenate all networks into a single to-be-splitted array
            skip_nodes_list (list[list[str]], optional): add extra skipped node names, otherwise, load skip_nodes in zarr zip files. Defaults to [].
            skip_types_list (list[list[str]], optional): massively skip by node types (juctions, reservoir, tank). Defaults to [].
            unstackable_pad_value (Any, optional): pad array by this value, when array is unmatched/unstacked. Defaults to 0.
            bypass_skip_names_in_config (bool, optional): Whether bypass default skipped node names in zarrs. Mutual inclusive w skip_nodes_list, skip_types_list. Defaults to False.
            chunk_limit (str, optional): Deprecated. Defaults to "120 GB".
            overwatch (bool, optional): Deprecated. Defaults to False.
            batch_axis_choice (Literal[&quot;scene&quot;, &quot;temporal&quot;, &quot;snapshot&quot;], optional): split subset by axis. Defaults to "scene".
            num_cpus (int, optional): Support multi-cpu data loading w the number of cpus (nightly). Defaults to 1.
            time_sampling_rate (int, optional): Deprecated. Defaults to 1.
            do_cache (bool, optional): flag indicates dump everything in mem. Very fast but OOM can happen. Defaults to False.
            subset_shuffle (bool, optional): flag indicates whether subset is shuffled. After splitting WHOLE TRAIN/VAL/TEST sets, we perform subset shuffling per set. Defaults to True.
            dataset_log_pt_path (str, optional): .pt path where storing subset indices and statistic (when you call `gather_statistic`). If `dataset_log_pt_path=""`, we WON'T LOAD SHUFFLE IDs.  Defaults to ''.
            run_mode (Literal[&quot;pretrain&quot;, &quot;train_default&quot;, &quot;shared_enc&quot;], optional): Defaults to "train_default": original implementation GATRes.
        """  # noqa: E501
        self.input_paths = input_paths
        self.zip_file_paths = zip_file_paths
        self._is_init_root = False
        self.node_attrs = [att if isinstance(att, str) else tuple(att) for att in node_attrs] if node_attrs else []
        self.edge_attrs = [att if isinstance(att, str) else tuple(att) for att in edge_attrs] if edge_attrs else []
        self.label_attrs = [att if isinstance(att, str) else tuple(att) for att in label_attrs] if label_attrs else []
        self.edge_label_attrs = [att if isinstance(att, str) else tuple(att) for att in
                                 edge_label_attrs] if edge_label_attrs else []

        if "headpump_base_speed" in self.edge_attrs or "headpump_base_speed" in self.label_attrs or "headpump_base_speed" in self.edge_label_attrs:
            print("WARNING! headpump_base_speed attribute is under validation phase!")

        assert run_mode in ["pretrain", "pretrain-inference", "train_default",
                            "shared_enc",
                            "dual_encoder"], "run_mode must be either 'pretrain' or 'train_default' or 'shared_enc'."

        self.verbose = verbose
        self.num_records = num_records
        self.selected_snapshots = selected_snapshots
        self.split_set = split_set
        self.skip_nodes_list = skip_nodes_list
        self.skip_types_list = skip_types_list
        self.split_type: Literal["temporal", "scene"] = split_type
        self.split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
        self.split_fixed = split_fixed
        self.split_per_network = split_per_network
        self._arrays: list[tuple[zarr.Array, zarr.Array, zarr.Array | None, zarr.Array | None, zarr.Array | None]] = []
        self._index_map: dict[int, tuple[int | None, int | None]] = {}
        self._network_map: dict[int, int] = {}
        self._roots: list[GidaV6.Root] = []
        self._node_degree_map: dict[int, Tensor | None] = {}
        self._node_degree_stats: dict[str, Tensor | None] = {}
        self.bypass_skip_names_in_config = bypass_skip_names_in_config
        self.unstackable_pad_value = unstackable_pad_value
        self.chunk_limit = chunk_limit
        self.batch_axis_choice: Literal["scene", "temporal", "snapshot"] = batch_axis_choice
        self.overwatch = overwatch
        self.num_cpus = num_cpus
        self.time_sampling_rate = time_sampling_rate
        self.do_cache = do_cache
        self.subset_shuffle = subset_shuffle
        self.train_shuffle_ids, self.val_shuffle_ids, self.test_shuffle_ids = [], [], []
        self.dataset_log_pt_path = dataset_log_pt_path
        self.run_mode = run_mode
        self.sampling_strategy = sampling_strategy
        # allow user customize function here
        self.custom_process()

        transform = kwargs.get("transform", None)
        super().__init__(None, transform=transform)

        # subset shuffle load/write depends on `subset_shuffle` and `dataset_log_pt_path`
        # self.process_subset_shuffle()
        if self._indices is None:
            self.update_indices()

    def _get_num_samples(self) -> int:
        if self.split_fixed:
            return self.num_records if self.num_records is not None else self.length
        else:
            return self.length

    def custom_process(self) -> None:
        # load arrays from zip file (and input_paths)
        self.length, self._index_map, self._network_map, self._num_samples_per_network_list = self.compute_indices(
            zip_file_paths=self.zip_file_paths,
            input_paths=self.input_paths,
        )

        self.train_ids, self.val_ids, self.test_ids = self.compute_subset_ids_by_ratio(self.split_ratios, num_samples=self._get_num_samples())

    def update_indices(self) -> None:
        # update _indices
        attempted_ids = self.train_ids + self.val_ids + self.test_ids
        if len(attempted_ids) > 0:
            self._indices = attempted_ids
        else:
            assert self._index_map is not None and len(self._index_map) > 0
            self._indices = list(self._index_map.keys())

    def _internal_subset_shuffle(
        self, sampling_strategy: Literal["default", "random"] = "default"
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        if sampling_strategy == "default":
            new_train_ids, train_shuffle_ids = shuffle_list(self.train_ids)
            new_val_ids, val_shuffle_ids = shuffle_list(self.val_ids)
            new_test_ids, test_shuffle_ids = shuffle_list(self.test_ids)
            return new_train_ids, new_val_ids, new_test_ids, train_shuffle_ids, val_shuffle_ids, test_shuffle_ids
        elif sampling_strategy == "random":
            raise NotImplementedError
            # train_ids = np.asarray(self.train_ids)
            # val_ids = np.asarray(self.val_ids)
            # test_ids = np.asarray(self.test_ids)

            # train_shuffle_ids = []
            # val_shuffle_ids = []
            # test_shuffle_ids = []
            # start_train_idx = start_val_idx = start_test_idx = 0
            # for network_length in self._num_samples_per_network_list:
            #     network_num_trains = int(network_length * self.split_ratios[0])
            #     network_num_vals = int(network_length * self.split_ratios[1])
            #     network_num_tests = network_length - network_num_trains - network_num_vals

            #     # extract network subset, shuffle, reassign
            #     tmp_subset = train_ids[start_train_idx : start_train_idx + network_num_trains]
            #     tmp_shuffle_ids = np.random.permutation(network_num_trains)
            #     tmp_subset = tmp_subset[tmp_shuffle_ids]
            #     train_ids[start_train_idx : start_train_idx + network_num_trains] = tmp_subset
            #     train_shuffle_ids.extend(start_train_idx + tmp_shuffle_ids)

            #     tmp_subset = val_ids[start_val_idx : start_val_idx + network_num_vals]
            #     tmp_shuffle_ids = np.random.permutation(network_num_vals)
            #     tmp_subset = tmp_subset[tmp_shuffle_ids]
            #     val_ids[start_val_idx : start_val_idx + network_num_vals] = tmp_subset
            #     val_shuffle_ids.extend(start_val_idx + tmp_shuffle_ids)

            #     tmp_subset = test_ids[start_test_idx : start_test_idx + network_num_tests]
            #     tmp_shuffle_ids = np.random.permutation(network_num_tests)
            #     tmp_subset = tmp_subset[tmp_shuffle_ids]
            #     test_ids[start_test_idx : start_test_idx + network_num_tests] = tmp_subset
            #     test_shuffle_ids.extend(start_test_idx + tmp_shuffle_ids)

            #     # update start indices
            #     start_train_idx += network_num_trains
            #     start_val_idx += network_num_vals
            #     start_test_idx += network_num_tests

            # # assert np.allclose(train_ids, np.asarray(self.train_ids)[train_shuffle_ids])
            # # assert np.allclose(val_ids, np.asarray(self.val_ids)[val_shuffle_ids])
            # # assert np.allclose(test_ids, np.asarray(self.test_ids)[test_shuffle_ids])
            # return train_ids.tolist(), val_ids.tolist(), test_ids.tolist(), train_shuffle_ids, val_shuffle_ids, test_shuffle_ids

        else:
            raise NotImplementedError

    def process_subset_shuffle(self, custom_subset_shuffle_pt_path: str = "", create_and_save_to_dataset_log_if_nonexist: bool = True):
        if self.subset_shuffle:
            if custom_subset_shuffle_pt_path != "":
                self._load_shuffle_indices_from_disk_internal(path=custom_subset_shuffle_pt_path, sanity_check=True)
            else:
                # we first check whether the pt file exists. If yes, we load train/val/test ids
                if not self.load_shuffle_indices_from_disk(sanity_check=True):
                    # otherwise, we perform shuffle and store ids in a saving folder

                    new_train_ids, new_val_ids, new_test_ids, train_shuffle_ids, val_shuffle_ids, test_shuffle_ids = self._internal_subset_shuffle(
                        sampling_strategy="default"
                    )

                    if self.dataset_log_pt_path != "" and create_and_save_to_dataset_log_if_nonexist:
                        self.save_shuffle_indices_to_disk(
                            self.train_ids, self.val_ids, self.test_ids, train_shuffle_ids, val_shuffle_ids, test_shuffle_ids
                        )
                    else:
                        print(
                            "WARN! Subset shuffle indices cannot be saved as `dataset_log_pt_path` is empty or `do_save_to_dataset_log` is set to False in Gida Interface! You cannot re-load these ids in the inference or next train!",  # noqa: E501
                            flush=True,
                        )

                    self.train_ids = new_train_ids
                    self.val_ids = new_val_ids
                    self.test_ids = new_test_ids
            self.update_indices()

    def save_shuffle_indices_to_disk(
        self,
        ori_train_ids: list[int],
        ori_val_ids: list[int],
        ori_test_ids: list[int],
        train_shuffle_ids: list[int],
        val_shuffle_ids: list[int],
        test_shuffle_ids: list[int],
    ) -> None:
        assert self.dataset_log_pt_path != "", "ERROR! dataset_log_pt_path should not be empty!"
        assert self.dataset_log_pt_path[-4:] == ".pth" or self.dataset_log_pt_path[-3:] == ".pt", (
            f"ERROR! dataset_log_pt_path is invalid. Get self.dataset_log_pt_path = ({self.dataset_log_pt_path})"
        )
        my_dict = {
            "train_ids": ori_train_ids,
            "val_ids": ori_val_ids,
            "test_ids": ori_test_ids,
            "train_shuffle_ids": train_shuffle_ids,
            "val_shuffle_ids": val_shuffle_ids,
            "test_shuffle_ids": test_shuffle_ids,
            "zip_file_paths": self.zip_file_paths,
        }

        if os.path.exists(self.dataset_log_pt_path):
            # prevent overriden
            cp_dict: dict = load(self.dataset_log_pt_path)
            cp_dict.update(my_dict)
            save(cp_dict, self.dataset_log_pt_path)
        else:
            save(my_dict, self.dataset_log_pt_path)

    def _load_shuffle_indices_from_disk_internal(self, path: str, sanity_check: bool = True) -> bool:
        if os.path.isfile(path) and (path[-4:] == ".pth" or path[-3:] == ".pt"):
            my_dict = load(path)

            if sanity_check:
                try:
                    assert set(my_dict["zip_file_paths"]) == set(self.zip_file_paths)
                    assert set(my_dict["train_ids"]) == set(self.train_ids)
                    assert set(my_dict["val_ids"]) == set(self.val_ids)
                    assert set(my_dict["test_ids"]) == set(self.test_ids)
                except Exception:  # Include assertionerror and keyerror :)
                    return False
            if "train_shuffle_ids" in my_dict:
                self.train_ids = masking_list(self.train_ids, my_dict["train_shuffle_ids"])
            if "val_shuffle_ids" in my_dict:
                self.val_ids = masking_list(self.val_ids, my_dict["val_shuffle_ids"])
            if "test_shuffle_ids" in my_dict:
                self.test_ids = masking_list(self.test_ids, my_dict["test_shuffle_ids"])
            return True
        return False

    def load_shuffle_indices_from_disk(self, sanity_check: bool = True) -> bool:
        return self._load_shuffle_indices_from_disk_internal(path=self.dataset_log_pt_path, sanity_check=sanity_check)

    def compute_subset_ids_by_ratio(self, split_ratios: tuple[float, float, float], num_samples: int) -> tuple[list[int], list[int], list[int]]:
        train_ids, val_ids, test_ids = [], [], []
        len_of_list = len(self._num_samples_per_network_list)
        # if not split per network or existing a single network only, we split based on flatten ids
        if not self.split_per_network:  # or len_of_list == 1:  ######### <------------- This was changed
            left = int(self.length * split_ratios[0])
            right = int(left + self.length * split_ratios[1])

            flatten_ids = np.asarray(list(self._index_map.keys()))

            flatten_ids = flatten_ids.tolist()
            train_ids = flatten_ids[:left]
            val_ids = flatten_ids[left:right]
            test_ids = flatten_ids[right:]
            if self.split_fixed:
                selected_trains = int(num_samples * split_ratios[0])
                selected_vals = int(num_samples * split_ratios[1])
                selected_test = num_samples - selected_trains - selected_vals

                train_ids = train_ids[:selected_trains]
                val_ids = val_ids[:selected_vals]
                test_ids = test_ids[:selected_test]
        else:
            # to split per network, we compute train/val/test individually
            # degree of freedom will be (len_of_list - 1)
            expected_train_samples = int(self.length * split_ratios[0])
            expected_valid_samples = int(self.length * split_ratios[1])
            expected_test_samples = self.length - expected_train_samples - expected_valid_samples
            flatten_ids = np.asarray(list(self._index_map.keys()))

            # num_samples_per_network = num_samples // len(self._num_samples_per_network_list)
            num_samples_per_network = self.num_records // len(self._num_samples_per_network_list)
            current_nid = 0
            for i, network_num_samples in enumerate(self._num_samples_per_network_list):
                network_flatten_ids = flatten_ids[current_nid : current_nid + network_num_samples]

                if self.batch_axis_choice == "snapshot":
                    # with snapshots, we still split by scence to ensure the scenario independence
                    # f_0-> (n_0, t_0), f_1 -> (n_0, t_1), ..., f_T -> (n_0, t_T), f_T+1 -> (n_1, t_0), ...
                    time_dim = self._roots[i].time_dim
                    num_scenes = len(network_flatten_ids) // time_dim
                    left = int(num_scenes * split_ratios[0])
                    right = int(left + num_scenes * split_ratios[1])

                    left = left * time_dim
                    right = right * time_dim

                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]
                else:
                    left = int(network_num_samples * split_ratios[0])
                    right = int(left + network_num_samples * split_ratios[1])
                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]

                if i == len_of_list - 1:
                    network_train_ids = network_train_ids[: expected_train_samples - len(train_ids)]
                    network_val_ids = network_val_ids[: expected_valid_samples - len(val_ids)]
                    network_test_ids = network_test_ids[: expected_test_samples - len(test_ids)]

                if self.split_fixed:
                    selected_trains = int(num_samples_per_network * split_ratios[0])
                    selected_vals = int(num_samples_per_network * split_ratios[1])
                    selected_test = num_samples_per_network - selected_trains - selected_vals

                    if self.do_cache:
                        network_train_ids = network_train_ids[np.random.permutation(len(network_train_ids))]
                        network_val_ids = network_val_ids[np.random.permutation(len(network_val_ids))]
                        network_test_ids = network_test_ids[np.random.permutation(len(network_test_ids))]
                    else:
                        samples_per_chunch = time_dim * 10
                        total_samples_train = math.ceil(selected_trains / samples_per_chunch) * samples_per_chunch
                        total_samples_vals = math.ceil(selected_vals / samples_per_chunch) * samples_per_chunch
                        total_samples_test = math.ceil(selected_test / samples_per_chunch) * samples_per_chunch
                        # if num_samples_per_network < samples_per_bucket:

                        network_train_ids = network_train_ids[np.random.permutation(total_samples_train)]
                        network_val_ids = network_val_ids[np.random.permutation(total_samples_vals)]
                        network_test_ids = network_test_ids[np.random.permutation(total_samples_test)]

                    network_train_ids = network_train_ids[:selected_trains]
                    network_val_ids = network_val_ids[:selected_vals]
                    network_test_ids = network_test_ids[:selected_test]

                train_ids.extend(network_train_ids.tolist())
                val_ids.extend(network_val_ids.tolist())
                test_ids.extend(network_test_ids.tolist())
                current_nid += network_num_samples

        return train_ids, val_ids, test_ids

    def _compute_subset_ids(
        self,
        split_ratios: tuple[float, float, float],
        num_samples: int,
        custom_subset_shuffle_pt_path: str,
        create_and_save_to_dataset_log_if_nonexist: bool,
    ) -> tuple[list[int], list[int], list[int]]:
        train_ids, val_ids, test_ids = [], [], []
        len_of_list = len(self._num_samples_per_network_list)
        num_trains = int(num_samples * split_ratios[0])
        num_vals = int(num_samples * split_ratios[1])
        num_tests = num_samples - num_trains - num_vals
        if not self.split_per_network or len_of_list == 1:
            left = int(self.length * split_ratios[0])
            right = int(left + self.length * split_ratios[1])
            flatten_ids = np.asarray(list(self._index_map.keys()))
            train_ids = flatten_ids[:left]
            val_ids = flatten_ids[left:right]
            test_ids = flatten_ids[right:]
            len_train_ids = len(train_ids)
            len_val_ids = len(val_ids)
            len_test_ids = len(test_ids)
            if self.subset_shuffle:
                train_ids = train_ids[np.random.permutation(len_train_ids)]
                val_ids = val_ids[np.random.permutation(len_val_ids)]
                test_ids = test_ids[np.random.permutation(len_test_ids)]
            # second-level triming
            train_ids = train_ids[:num_trains].tolist()
            val_ids = val_ids[:num_vals].tolist()
            test_ids = test_ids[:num_tests].tolist()
        else:
            # to split per network, we compute train/val/test individually
            # degree of freedom will be (len_of_list - 1)
            expected_train_samples = int(self.length * split_ratios[0])
            expected_valid_samples = int(self.length * split_ratios[1])
            expected_test_samples = self.length - expected_train_samples - expected_valid_samples
            flatten_ids = np.asarray(list(self._index_map.keys()))

            indi_num_trains = num_trains // len_of_list
            indi_num_vals = num_vals // len_of_list
            indi_num_tests = num_tests // len_of_list

            current_nid = 0
            for i, network_num_samples in enumerate(self._num_samples_per_network_list):
                network_flatten_ids = flatten_ids[current_nid : current_nid + network_num_samples]

                if self.batch_axis_choice == "snapshot":
                    # with snapshots, we still split by scence to ensure the scenario independence
                    # f_0-> (n_0, t_0), f_1 -> (n_0, t_1), ..., f_T -> (n_0, t_T), f_T+1 -> (n_1, t_0), ...
                    time_dim = self._roots[i].time_dim
                    num_scenes = len(network_flatten_ids) // time_dim
                    left = int(num_scenes * split_ratios[0])
                    right = int(left + num_scenes * split_ratios[1])

                    left = left * time_dim
                    right = right * time_dim

                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]

                else:
                    left = int(network_num_samples * split_ratios[0])
                    right = int(left + network_num_samples * split_ratios[1])
                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]

                len_train_ids = len(train_ids)
                len_val_ids = len(val_ids)
                len_test_ids = len(test_ids)
                if i == len_of_list - 1:
                    network_train_ids = network_train_ids[: expected_train_samples - len_train_ids]
                    network_val_ids = network_val_ids[: expected_valid_samples - len_val_ids]
                    network_test_ids = network_test_ids[: expected_test_samples - len_test_ids]

                # second-level triming
                if self.subset_shuffle:
                    network_train_ids = network_train_ids[np.random.permutation(len_train_ids)]
                    network_val_ids = network_val_ids[np.random.permutation(len_val_ids)]
                    network_test_ids = network_test_ids[np.random.permutation(len_test_ids)]

                if i < len_of_list - 1:
                    network_train_ids = network_train_ids[:indi_num_trains]
                    network_val_ids = network_val_ids[:indi_num_vals]
                    network_test_ids = network_test_ids[:indi_num_tests]
                else:
                    network_train_ids = network_train_ids[: num_trains - len(train_ids)]
                    network_val_ids = network_val_ids[: num_vals - len(val_ids)]
                    network_test_ids = network_test_ids[: num_tests - len(test_ids)]
                # TODO: Dealing with sampling strategy
                train_ids.extend(network_train_ids.tolist())
                val_ids.extend(network_val_ids.tolist())
                test_ids.extend(network_test_ids.tolist())
                current_nid += network_num_samples

        return train_ids, val_ids, test_ids

    def compute_indices(
        self, zip_file_paths: list[str], input_paths: list[str] = []
    ) -> tuple[int, dict[int, tuple[int | None, int | None]], dict[int, int], list[int]]:
        # this is must-have since the size of networks is different.
        index_map: dict[int, tuple[int | None, int | None]] = {}
        network_map: dict[int, int] = {}
        num_samples_per_network_list: list[int] = []
        flatten_index = 0
        self.load_roots(zip_file_paths, input_paths)

        root_sizes: list[int] = [r.compute_first_size() for r in self._roots]
        if self.batch_axis_choice == "scene":
            sum_of_root_sizes = sum(root_sizes)
        elif self.batch_axis_choice == "temporal":
            sum_of_root_sizes = sum([r.time_dim for r in self._roots])
        else:  # snapshot
            sum_of_root_sizes = sum([r.time_dim * root_sizes[i] for i, r in enumerate(self._roots)])

        total_num_samples: int = sum_of_root_sizes  # if self.num_records is None else min(sum_of_root_sizes, self.num_records)
        num_samples_per_network = total_num_samples // len(self._roots)

        for network_index, root in enumerate(self._roots):
            # if self.run_mode != "pretrain" and self.run_mode != "train_default":
            if self.run_mode != "train_default":
                edge_index_undirected = to_undirected(torch.tensor(root.edge_index))
                # root.edge_index, root.edge_attr = to_undirected(edge_index=torch.as_tensor(root.edge_index),
                #                                                 edge_attr=torch.as_tensor(root.edge_attr),
                #                                                 reduce="mean")
                self._node_degree_map[network_index] = compute_node_degree(edge_index=torch.as_tensor(edge_index_undirected), num_nodes=root.node_mask.sum())

            if self.batch_axis_choice == "scene":
                # arr WILL have shape <merged>(#scenes, #nodes_or_#links, #statics + time_dims * #dynamics)
                if self.split_fixed:
                    relative_scene_ids = np.arange(root_sizes[network_index])
                    relative_scene_ids = relative_scene_ids[:num_samples_per_network]
                    num_samples = len(relative_scene_ids)  # min(num_samples_per_network, root_sizes[network_index])
                else:
                    num_samples = root.compute_first_size() #if self.num_records is None else min(self.num_records, root.compute_first_size())
                    relative_scene_ids = np.arange(num_samples)
                tuples = (relative_scene_ids, None)
            elif self.batch_axis_choice == "temporal":
                if self.split_fixed:
                    relative_time_ids = np.arange(root.time_dim)
                    relative_time_ids = relative_time_ids[:num_samples_per_network]
                    num_samples = len(relative_time_ids)
                else:
                    num_samples = root.time_dim
                    relative_time_ids = np.arange(num_samples)
                tuples = (None, relative_time_ids)

            elif self.batch_axis_choice == "snapshot":
                if self.split_fixed:
                    time_dim = root.time_dim
                    num_scenes = root_sizes[network_index]
                    relative_scene_ids = np.arange(num_scenes).repeat(time_dim)  # .reshape([-1, 1])
                    relative_time_ids = np.tile(np.arange(time_dim), reps=num_scenes)  # .reshape([-1, 1])
                    # relative_scene_ids = relative_scene_ids[:num_samples_per_network]  ## comment this
                    # relative_time_ids = relative_time_ids[:num_samples_per_network]  ## comment this
                    num_samples = len(relative_scene_ids)
                else:
                    num_scenes = root.compute_first_size() #if self.num_records is None else min(self.num_records, root.compute_first_size())
                    time_dim = root.time_dim
                    relative_scene_ids = np.arange(num_scenes).repeat(time_dim)  # .reshape([-1, 1])
                    relative_time_ids = np.tile(np.arange(time_dim), reps=num_scenes)  # .reshape([-1, 1])
                    tuples = (relative_scene_ids, relative_time_ids)
                    num_samples = len(relative_scene_ids)
                tuples = (relative_scene_ids, relative_time_ids)
            else:
                raise NotImplementedError
            extended_network_ids = np.full([num_samples], network_index)
            flatten_ids = np.arange(flatten_index, flatten_index + num_samples)

            network_index_map: dict[int, tuple[int | None, int | None]] = {}
            # fid_nid_map: dict[int, int] = {}
            lefts: np.ndarray | None = tuples[0] if tuples[0] is not None else None
            rights: np.ndarray | None = tuples[1] if tuples[1] is not None else None
            for i, fid in enumerate(flatten_ids):
                left = lefts[i] if lefts is not None else None
                right = rights[i] if rights is not None else None
                network_index_map[fid] = (left, right)
                # fid_nid_map[fid] = network_index
            # update the global map
            index_map.update(network_index_map)
            network_map.update(zip(flatten_ids, extended_network_ids))  # network_map.update(fid_nid_map)  #
            # update flatten index indicator and network index
            flatten_index += num_samples
            num_samples_per_network_list.append(num_samples)

        length = flatten_index
        return length, index_map, network_map, num_samples_per_network_list

    def is_node(self, component: str) -> bool:
        return component in ["junction", "tank", "reservoir", "node"]

    def get_component(self, attr: str) -> str:
        has_component = "_" in attr
        component = attr.split("_")[0] if has_component else "node" if is_node_simulation_output(attr) else "link"
        return component

    def get_component_by_attr(self, attr: str | tuple, axis=0) -> str:
        """if attr is str, return component| if attr is tuple, reuturn component of the axis-th item in the tuple"""

        if isinstance(attr, str):
            return self.get_component(attr)
        elif isinstance(attr, tuple):
            return self.get_component(attr[axis])
        else:
            raise TypeError(f"attr {attr} has type {type(attr)} which is illegal.")

    def get_available_node_components(self, okeys: OrderedDict, root_components: list[str]) -> list[str]:
        return [k for k in okeys if self.is_node(k) and k in root_components]

    def get_available_link_components(self, okeys: OrderedDict, root_components: list[str]) -> list[str]:
        return [k for k in okeys if not self.is_node(k) and k in root_components]

    def get_object_names_by_component(self, root: Root, component: str, wn: Optional[WaterNetworkModel] = None) -> list[str]:
        if wn is not None:
            return get_object_name_list_by_component(component, wn)
        else:
            return root.attrs["onames"][component] if component in root.attrs["onames"] else []

    def compute_selected_components(
        self,
        root: Root,
        sorted_attrs: list[str | tuple],
        has_asterisks: list[bool],
        skip_types: list[str] = [],
    ) -> list[str]:
        if len(sorted_attrs) <= 0:
            return []

        okeys: OrderedDict = root.attrs["okeys"]

        placeholder_components: list[str] = []

        root_components = [self.get_component(attr) for attr in root.array_keys]

        for attr, has_asterisk in zip(sorted_attrs, has_asterisks):
            if isinstance(attr, str):
                # if it is a pure str, we gather data w.r.t. str key
                assert attr in root.array_keys, f"Root has no attr {attr}! Check spelling"
                com = self.get_component(attr)
                if not has_asterisk:
                    if com == "node":
                        placeholder_components.extend(self.get_available_node_components(okeys=okeys, root_components=root_components))
                    elif com == "link":
                        placeholder_components.extend(self.get_available_link_components(okeys=okeys, root_components=root_components))
                    else:
                        placeholder_components.append(com)

            elif isinstance(attr, tuple):
                for sub_attr in attr:
                    assert sub_attr in root.array_keys, f"Root has no attr {attr}! Check spelling"
                    sub_component = self.get_component(sub_attr)
                    if sub_component not in placeholder_components:
                        placeholder_components.append(sub_component)

        # sort and deduplicate by okeys
        selected_placeholder_components = [k for k in okeys if k in placeholder_components and k not in skip_types]

        if len(selected_placeholder_components) <= 0:
            assert all(has_asterisks)
            selected_placeholder_components.extend([self.get_component(a) for a in sorted_attrs if isinstance(a, str)])

        return selected_placeholder_components

    def load_roots(
        self,
        zip_file_paths: list[str],
        input_paths: list[str] = [],
    ) -> None:
        for i, zip_file_path in enumerate(zip_file_paths):
            assert os.path.isfile(zip_file_path) and zip_file_path[-4:] == ".zip", f"{zip_file_path} is not a zip file"

            if self._is_init_root:
                root = self._roots[i]
            else:
                assert zip_file_path is not None
                root = GidaV6.Root(name=f"root_{i}", zip_file_path=zip_file_path, num_cpus=self.num_cpus)
                self._roots.append(root)
            if self.verbose:
                print(f"config keys = {root.attrs.keys()}")

            # sort user-define attrs w.r.t okeys
            sorted_node_attrs, node_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.node_attrs)
            sorted_edge_attrs, edge_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.edge_attrs)
            sorted_label_attrs, label_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.label_attrs)
            sorted_edge_label_attrs, edge_label_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.edge_label_attrs)

            assert not any(set(GidaV6.Pump_Curve).intersection(sorted_edge_attrs)) and not any(
                set(GidaV6.Pump_Curve).intersection(sorted_edge_label_attrs)
            ), "Curve-related parameters are not supported currently"

            # convert wdn to edge_index or load adj_list
            if len(input_paths) > 0:
                wn = WaterNetworkModel(input_paths[i])
                adj_list: list[tuple[str, str, str]] = get_adj_list(wn, [])
            else:
                wn = None
                adj_list: list[tuple[str, str, str]] = root.attrs["adj_list"]

            okeys = root.attrs["okeys"]
            root_components = [self.get_component(attr) for attr in root.array_keys]
            avail_total_node_components = self.get_available_node_components(okeys=okeys, root_components=root_components)
            avail_total_link_components = self.get_available_link_components(okeys=okeys, root_components=root_components)

            skip_nodes: list[str] = self.skip_nodes_list[i] if len(self.skip_nodes_list) > i else []  # type:ignore

            # skip nodes
            if self.bypass_skip_names_in_config:
                if self.verbose:
                    print(
                        "WARN! Flag bypass_skip_names_in_config is (True)! Consequently, GiDa includes invalid nodes that may cause negative pressures!"
                    )
            else:
                skip_nodes = list(set(skip_nodes).union(root.attrs["skip_names"]))

            # update adj according to skip_nodes
            skip_types: list[str] = self.skip_types_list[i] if len(self.skip_types_list) > i else []  # type:ignore

            # get node_components (in-order)
            node_components = self.compute_selected_components(
                root,
                sorted_attrs=sorted_node_attrs,
                has_asterisks=node_has_asterisks,
                skip_types=skip_types,
            )
            edge_components = self.compute_selected_components(
                root,
                sorted_attrs=sorted_edge_attrs,
                has_asterisks=edge_has_asterisks,
                skip_types=skip_types,
            )

            label_node_skip_types = list(set(avail_total_node_components).difference(node_components))
            label_edge_skip_types = list(set(avail_total_link_components).difference(edge_components))
            node_skip_types = list(set(skip_types).union(label_node_skip_types))
            edge_skip_types = list(set(skip_types).union(label_edge_skip_types))

            # update node mask
            node_masks = []
            node_names = []
            for node_com in node_components:
                object_names = self.get_object_names_by_component(root=root, component=node_com)
                if node_com not in node_skip_types:
                    mask_list = [name not in skip_nodes for name in object_names]
                    mask = np.asarray(mask_list, dtype=bool)
                    node_names.extend(list(compress(object_names, mask_list)))
                else:
                    mask = np.zeros(len(object_names), dtype=bool)
                    skip_nodes.extend(object_names)
                node_masks.append(mask)
            node_mask = np.concatenate(node_masks)

            if self.verbose:
                print(f"Skip nodes = {skip_nodes}")

            # update edge mask
            adj_masks = []
            edge_masks = []

            edge_names = []
            for edge_com in edge_components:
                if edge_com not in edge_skip_types:
                    edge_names.extend(self.get_object_names_by_component(root=root, component=edge_com))

            is_edge_names_empty = len(edge_names) <= 0
            for src, dst, link_name in adj_list:
                is_selected = src in node_names and dst in node_names
                is_in_edge_names = is_edge_names_empty or link_name in edge_names
                if is_in_edge_names:
                    edge_masks.append(is_selected)

                adj_masks.append(is_selected and is_in_edge_names)

            edge_names = list(compress(edge_names, edge_masks))
            edge_mask = np.asarray(edge_masks, dtype=bool)
            adj_mask = np.asarray(adj_masks, dtype=bool)

            root.node_mask = node_mask
            root.adj_mask = adj_mask
            root.edge_mask = edge_mask
            root.node_names = node_names
            root.edge_names = edge_names
            root.node_components = node_components
            root.edge_components = edge_components

            root.sorted_node_attrs = sorted_node_attrs
            root.node_has_asterisks = node_has_asterisks

            root.sorted_edge_attrs = sorted_edge_attrs
            root.edge_has_asterisks = edge_has_asterisks

            root.sorted_label_attrs = sorted_label_attrs
            root.label_has_asterisks = label_has_asterisks

            root.sorted_edge_label_attrs = sorted_edge_label_attrs
            root.edge_label_has_asterisks = edge_label_has_asterisks

            # perform skiping
            adj_list = list(compress(adj_list, adj_mask.tolist()))

            # after loading features and obtaining sorted attrs,
            # we update the adjacency list to ensure the nodal consistency between nodal features and adj

            num_nodes = len(node_names)
            num_edges = len(adj_list)
            if num_edges > 0:
                mapping = dict(zip(node_names, range(0, num_nodes)))
                edge_index = np.zeros((2, num_edges), dtype=np.int_)  # type:ignore
                for i, (src, dst, lkn) in enumerate(adj_list):
                    # if src in mapping and dst in mapping :
                    edge_index[0, i] = mapping[src]
                    edge_index[1, i] = mapping[dst]
            else:
                edge_index = np.zeros([2, 0], dtype=np.int_)  # type:ignore

            root.edge_index = edge_index

        self._is_init_root = True

    def strip_asterisk(self, attrs: list[str | tuple]) -> list[str | tuple]:
        new_attrs = []
        for attr in attrs:
            if isinstance(attr, str):
                new_attrs.append(attr.strip("*"))
            elif isinstance(attr, tuple):
                assert all("*" not in sub_attr for sub_attr in attr), f"tuple must have no asterisk, but get {attr}"
                new_attrs.append(attr)
        return new_attrs

    def sort_key_order_in_tuple(self, okeys: OrderedDict, compo_params: tuple) -> tuple[tuple, list]:
        if len(compo_params) <= 0:
            return (), []
        ordered_compoparams = []
        for k, vs in okeys.items():
            for v in vs:
                ordered_compoparams.append(f"{k}_{v}")

        indices = [ordered_compoparams.index(compo_param) if compo_param in ordered_compoparams else LARGE_NUMBER for compo_param in compo_params]
        arg_indices = np.argsort(np.array(indices))
        return tuple([compo_params[a] for a in arg_indices]), arg_indices.tolist()

    def sort_and_filter_key_order(self, root: Root, attrs: list[str | tuple]) -> tuple[list[str | tuple], list[bool]]:
        okeys: OrderedDict = root.attrs["okeys"]
        sorted_list = []
        has_asterisks: list[bool] = []
        for attr in attrs:
            if isinstance(attr, str):
                stripped_attr = self.strip_asterisk([attr])[0]
                if stripped_attr in root.array_keys:
                    sorted_list.append(stripped_attr)
                    has_asterisks.append("*" in attr)
            elif isinstance(attr, tuple):
                attr_mask = [sub_attr in root.array_keys for sub_attr in attr]
                attr = tuple(compress(attr, attr_mask))
                sorted_attr, sorted_idx = self.sort_key_order_in_tuple(okeys, compo_params=attr)
                sorted_list.append(sorted_attr)
                has_asterisks.append(False)

        return sorted_list, has_asterisks

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return self.length  # len(self._arrays)

    def get_set(
        self,
        ids: list[int],
        num_records: Optional[int] = None,
        sampling_strategy: Literal["sequential", "interval"] = "sequential",
        **kwargs: Any,
    ) -> Dataset:
        """get subset by computed ids

        Args:
            ids (list[int]): id list
            num_records (Optional[int], optional): pick only num_records samples. Set None to take all. Defaults to None.

        Returns:
            Dataset: a subset from GiDA
        """
        dataset = copy.copy(self)
        for k, v in kwargs.items():
            setattr(dataset, k, v)

        if sampling_strategy == "sequential":
            dataset._indices = ids[:num_records]
        elif sampling_strategy == "interval":
            if num_records is None or len(ids) <= num_records:
                dataset._indices = ids
            else:
                # step size  depends on by #neworks (len(self._roots)) or num_records
                # step_size = max(1, len(ids) // max(len(self._roots), num_records))
                step_size = max(1, len(ids) // num_records)
                tmp_ids = ids[::step_size]

                dataset._indices = tmp_ids[:num_records]
        dataset.length = len(dataset._indices)

        return dataset

    def get_and_cache_if_need(self, root: Root, key: str, indices: list[tuple[int | None, int | None]], do_cache: bool = False) -> np.ndarray:
        if do_cache:
            if not hasattr(self, "_array_cache"):
                setattr(self, "_array_cache", {})
            is_dynamic_key = key in GidaV6.DYNAMIC_PARAMS
            array_cache: dict[str, np.ndarray] = getattr(self, "_array_cache")
            time_or_one = root.time_dim if is_dynamic_key else 1

            cache_key = "_".join([root.name, key])
            if cache_key not in array_cache:
                my_zarr: zarr.Array = root.get_array_by_key(key)
                # arr has shape (num_scenes, time_or_one * num_components)
                arr: np.ndarray = my_zarr[:]  # OOM is acceptable
                # arr has shape (num_scenes, time_or_one, num_components)
                arr = arr.reshape([arr.shape[0], time_or_one, -1])
                # each has shape (num_scenes, num_objects, time_or_one)
                arr = arr.transpose(0, 2, 1)
                array_cache[cache_key] = arr

            optional_scene_ids, optional_time_ids = tuple(map(list, zip(*indices)))
            non_scene_dim = optional_scene_ids[0] is None
            non_time_dim = optional_time_ids[0] is None
            assert non_time_dim is not None or non_time_dim is not None

            if not non_scene_dim and non_time_dim:  # scene
                return array_cache[cache_key][optional_scene_ids]
            elif non_scene_dim and not non_time_dim:  # temporal
                arr = array_cache[cache_key]
                if arr.shape[-1] < len(optional_time_ids) and not is_dynamic_key:
                    return np.broadcast_to(arr, (arr.shape[0], arr.shape[1], len(optional_time_ids)))
                else:
                    return arr[..., optional_time_ids]
            else:  # snapshot
                arr = array_cache[cache_key]
                if arr.shape[-1] == 1 and not is_dynamic_key:
                    return arr[optional_scene_ids]
                else:
                    indexed_arr = arr[optional_scene_ids, :, optional_time_ids]
                    ret_arr = np.expand_dims(indexed_arr, axis=-1)
                    return ret_arr

        else:
            return self.get_numpy_array(root=root, key=key, indices=indices)

    def get_numpy_array(self, root: Root, key: str, indices: list[tuple[int | None, int | None]]) -> np.ndarray:
        # my_arr has shape (num_scenes,  time_or_one* num_objects)
        my_arr: zarr.Array = root.get_array_by_key(key)

        optional_scene_ids, optional_time_ids = tuple(map(list, zip(*indices)))

        non_scene_dim = optional_scene_ids[0] is None
        non_time_dim = optional_time_ids[0] is None
        assert non_time_dim is not None or non_time_dim is not None

        time_or_one = root.time_dim if key in GidaV6.DYNAMIC_PARAMS else 1

        if not non_scene_dim and non_time_dim:
            arr = my_arr[optional_scene_ids]
            arr = np.reshape(arr, [len(indices), time_or_one, -1]).transpose(0, 2, 1)
        else:  # not non_time_dim
            num_components = my_arr.shape[-1] // time_or_one
            if time_or_one > 1:
                cols = [np.arange(tid * num_components, tid * num_components + num_components) for tid in optional_time_ids]
                cols = np.concatenate(cols)
            else:
                cols = np.arange(num_components)
                cols = np.tile(cols, reps=len(optional_scene_ids))

            if non_scene_dim:  # temporal
                optional_scene_ids = np.arange(my_arr.shape[0])
                cols = np.tile(cols, my_arr.shape[0])
                rows = np.repeat(optional_scene_ids, num_components * len(optional_time_ids))
            else:  # snapshot
                optional_scene_ids = np.asarray(optional_scene_ids)
                rows = np.repeat(optional_scene_ids, num_components)
            arr = my_arr.vindex[rows, cols]
            if non_scene_dim:  # batch axis choice is temporal
                arr = arr.reshape([-1, len(indices), num_components]).transpose(1, 2, 0)
            else:  # batch axis choice is snapshot
                arr = arr.reshape([-1, num_components, 1])

        # arr must have shape of (batch_size, num_objects, time_dim)
        return arr

    def stack_features(
        self,
        root: Root,
        indices: list[tuple[int | None, int | None]],
        which_array: Literal["node", "edge", "label", "edge_label"] = "node",
    ) -> np.ndarray | None:
        if which_array == "node":
            sorted_attrs: list[str | tuple] = root.sorted_node_attrs
            has_asterisks: list[bool] = root.node_has_asterisks
            selected_placeholder_components = root.node_components
            is_node = True
        elif which_array == "edge":
            sorted_attrs: list[str | tuple] = root.sorted_edge_attrs
            has_asterisks: list[bool] = root.edge_has_asterisks
            selected_placeholder_components = root.edge_components
            is_node = False
        elif which_array == "label":
            sorted_attrs: list[str | tuple] = root.sorted_label_attrs
            has_asterisks: list[bool] = root.label_has_asterisks
            is_label_nodal = self.is_node(self.get_component_by_attr(sorted_attrs[0])) if len(sorted_attrs) > 0 else True
            selected_placeholder_components = root.node_components if is_label_nodal else root.edge_components
            is_node = is_label_nodal
        elif which_array == "edge_label":
            sorted_attrs: list[str | tuple] = root.sorted_edge_label_attrs
            has_asterisks: list[bool] = root.edge_label_has_asterisks
            selected_placeholder_components = root.edge_components
            is_node = False

        if len(sorted_attrs) <= 0:
            return None

        merging_arrs = []

        max_dim: int = -1
        for att, has_asterisk in zip(sorted_attrs, has_asterisks):
            if isinstance(att, str):
                # my_arr has shape (batch_size, num_objects, time_dim) if axis=0
                my_arr: np.ndarray = self.get_and_cache_if_need(root=root, key=att, indices=indices, do_cache=self.do_cache)
                if not has_asterisk:
                    max_dim = max(max_dim, my_arr.shape[1])
            elif isinstance(att, tuple):
                # each has shape (batch_size, DIFF num_objects, time_dim) if axis=0
                my_arrs: list[np.ndarray] = [
                    self.get_and_cache_if_need(root=root, key=sub_att, indices=indices, do_cache=self.do_cache) for sub_att in att
                ]
                my_arr = np.concatenate(my_arrs, axis=1)
                max_dim = max(max_dim, my_arr.shape[1])
            else:
                raise NotImplementedError

            merging_arrs.append(my_arr)

        if max_dim == -1:  # no non-asterisk attr
            max_dim = sum(a.shape[1] for a in merging_arrs)

        required_padding = any(has_asterisks)

        if required_padding:
            for i in range(len(merging_arrs)):
                feature: np.ndarray = merging_arrs[i]
                has_asterisk = has_asterisks[i]
                attr = sorted_attrs[i]
                if has_asterisk and feature.shape[1] < max_dim:
                    if self.verbose:
                        f"Warning! Attribute {attr} has feature.shape[1]: {feature.shape[1]} >= max dim: {max_dim}! It does not need asterisk sign!"
                    assert isinstance(attr, str)
                    component_i = self.get_component(attr)
                    placeholder_id = selected_placeholder_components.index(component_i)
                    prev_pad = 0
                    if placeholder_id > 0:
                        for j in range(0, placeholder_id):
                            component_j = selected_placeholder_components[j]
                            num_obj_dim_j = len(self.get_object_names_by_component(root, component_j, wn=None))
                            prev_pad += num_obj_dim_j

                    post_pad = 0
                    if placeholder_id < len(selected_placeholder_components) - 1:
                        for j in range(placeholder_id + 1, len(selected_placeholder_components)):
                            component_j = selected_placeholder_components[j]
                            num_obj_dim_j = len(self.get_object_names_by_component(root, component_j, wn=None))
                            post_pad += num_obj_dim_j

                    arr = np.pad(feature, ((0, 0), (prev_pad, post_pad), (0, 0)), constant_values=self.unstackable_pad_value)
                    assert arr.shape[1] == max_dim, (
                        "Error! Padding does not match! Check the consistency between sub attributes in tuple, and other attributes (e.g. curve attr)"
                    )

                    merging_arrs[i] = arr
                    if self.verbose:
                        print(f"vstack-[pad]- key {attr} shape: {feature.shape}(old) -> {arr.shape}(new)")
        cat_array = np.concatenate(merging_arrs, axis=-1)

        if is_node:
            cat_array = cat_array[:, root.node_mask]
        else:
            cat_array = cat_array[:, root.edge_mask]
        if self.verbose:
            print(f"vstack-prevskip:shape> all:{cat_array.shape}")

            print("*" * 40)
        return cat_array

    def get(self, idx: int | Sequence) -> Any:
        # return Data or list[Data]
        assert self._indices is not None
        if isinstance(idx, int):
            fids: list[int] = self._indices[idx]  # [idx]  #
        else:
            idx_list = np.asarray(idx).flatten().tolist()

            fids: list[int] = [self._indices[i] for i in idx_list]  # idx_list  #

        batch_size = len(fids)

        # group temporal ids by network id to save querying time
        root_vindices_dict = defaultdict(list)
        for fid in fids:
            network_id = self._network_map[fid]
            root_vindices_dict[network_id].append(self._index_map[fid])

        batch = []
        counter: int = 0
        for nid in root_vindices_dict.keys():
            root: GidaV6.Root = self._roots[nid]
            # print(f"Querying network nid=({nid})- zip_path=({os.path.basename(root.zip_file_path)}) ...")
            if counter >= batch_size:
                break
            index_tup_list: list[tuple[int | None, int | None]] = root_vindices_dict[nid]
            node_array = self.stack_features(root=root, indices=index_tup_list, which_array="node")
            assert node_array is not None

            edge_array = self.stack_features(root=root, indices=index_tup_list, which_array="edge")

            label_array = self.stack_features(root=root, indices=index_tup_list, which_array="label")

            edge_label_array = self.stack_features(root=root, indices=index_tup_list, which_array="edge_label")

            for i in range(node_array.shape[0]):
                if counter >= batch_size:
                    break
                x = node_array[i]
                edge_attr = edge_array[i] if edge_array is not None else None
                y = label_array[i] if label_array is not None else None
                edge_y = edge_label_array[i] if edge_label_array is not None else None
                edge_index: np.ndarray = root.edge_index  # self._roots[self._network_map[i]].edge_index  # type:ignore

                node_degrees = None
                # # if self.run_mode == "pretrain":
                # #     pass
                # #     # new_edges, added_edges = add_random_edge(edge_index=torch.tensor(edge_index), p=0.1)
                # #     #
                # #     # edge_index = to_undirected(new_edges, reduce="mean")
                # #     # edge_attr = None  # np.zeros_like(edge_index.T)
                # #     # node_degrees = compute_node_degree(edge_index=edge_index, num_nodes=x.shape[0])
                # #     # edge_index_undirected = edge_index  # <--------------------------DELETE THIS
                # #
                # #     # new_edges, added_edges = add_random_edge(edge_index=torch.tensor(edge_index), p=0.1)
                # #     # edge_index = new_edges
                # #     # edge_attr = None  # np.zeros_like(edge_index.T)
                # #     # node_degrees = compute_node_degree(edge_index=edge_index, num_nodes=x.shape[0])
                # # elif self.run_mode == "shared_enc" or self.run_mode == "pretrain-inference":
                # #     edge_index = to_undirected(torch.tensor(edge_index.tolist()), reduce="mean")
                # #     node_degrees = self._node_degree_map[nid]
                # # elif self.run_mode == "dual_encoder":
                # #     edge_index_undirected = to_undirected(torch.as_tensor(edge_index), reduce="mean")
                # #     node_degrees = self._node_degree_map[nid]
                # #     # node_degrees = compute_node_degree(edge_index=edge_index_undirected, num_nodes=x.shape[0])
                #
                #
                # # edge_index_undirected, edge_attr = to_undirected(edge_index=torch.as_tensor(edge_index),
                # #                                                  edge_attr=torch.as_tensor(edge_attr), reduce="mean")
                # # edge_index_undirected = edge_index
                if self.run_mode != "train_default":
                    # edge_index_undirected, edge_attr = to_undirected(edge_index=torch.as_tensor(edge_index), edge_attr=torch.as_tensor(edge_attr), reduce="mean")
                    node_degrees = self._node_degree_map[nid]

                # edge_index, edge_attr = to_undirected(torch.tensor(edge_index), torch.tensor(edge_attr))
                dat = parse2data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_y=edge_y, node_degrees=node_degrees)
                batch.append(dat)
                counter += 1
        assert counter == batch_size
        return batch  # type:ignore

    def indices(self) -> Sequence:
        assert self._indices is not None
        return self._indices

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union["Dataset", BaseData]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if isinstance(idx, (int, np.integer)) or (isinstance(idx, Tensor) and idx.dim() == 0) or (isinstance(idx, np.ndarray) and np.isscalar(idx)):
            # data = self.get(self.indices()[idx])[0]  # type:ignore
            data = self.get([idx])[0]  # type:ignore
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            batch = self.get(idx)  # type:ignore
            batch = batch if self.transform is None else [self.transform(data) for data in batch]  # type:ignore
            return batch  # type:ignore

    def __getitems__(self, idx: Union[int, np.integer, IndexType]) -> list[BaseData]:
        # return self.get(idx)  # type:ignore
        # batch: list[BaseData] = self.get(idx[0])  # type:ignore
        batch: list[BaseData] = self.get(idx)  # type:ignore

        batch = batch if self.transform is None else [self.transform(dat) for dat in batch]
        return batch

    def gather_statistic(
        self,
        which_array: Literal["node", "edge", "label", "edge_label"],
        norm_dim: Optional[int] = 0,
        to_tensor: bool = True,
        num_batches: int = 10,
    ) -> tuple[
        np.ndarray | Tensor,
        np.ndarray | Tensor,
        np.ndarray | Tensor,
        np.ndarray | Tensor,
    ]:
        """
        norm_dim only supports 0 or -1. Otherwise, we cannot concatenate arrays from different-in-size networks </br>
        if compute, stat tuple with be saved in `self.dataset_log_pt_path` (if non-empty), a .pt file. </br>
        return: (min_val, max_val, mean_val, std_val)
        """
        channel_dim = -1
        assert self._indices is not None
        which_array_attr_map = {
            "node": "x",
            "edge": "edge_attr",
            "label": "y",
            "edge_label": "edge_y",
        }
        which_array_attrs_map = {
            "node": getattr(self._roots[0], "sorted_node_attrs"),
            "edge": getattr(self._roots[0], "sorted_edge_attrs"),
            "label": getattr(self._roots[0], "sorted_label_attrs"),
            "edge_label": getattr(self._roots[0], "sorted_edge_label_attrs"),
        }

        time_dim = self._roots[0].time_dim  # self._roots[0].attrs["duration"] // self._roots[0].attrs["time_step"]
        param_attrs = which_array_attrs_map[which_array]
        assert param_attrs is not None and len(param_attrs) > 0, f"ERROR! No found paramattrs from which_array=({which_array}): ({param_attrs})"
        channel_splitters = [
            time_dim if (isinstance(pa, str) and pa in GidaV6.DYNAMIC_PARAMS) or (pa[0] in GidaV6.DYNAMIC_PARAMS) else 1 for pa in param_attrs
        ]
        sum_channels = sum(channel_splitters)

        are_all_same_as_sum_channels: bool = True

        cat_arrays: list[dac.Array] = []

        batch_indices = np.array_split(np.arange(len(self.indices())), num_batches)

        selected_attr_key = which_array_attr_map[which_array]
        for bids in batch_indices:
            bids_list = bids.tolist()
            if bids_list:
                data_list: list = self.get(bids_list)
                batch = Batch.from_data_list(data_list)
                batch_att_array: Tensor = getattr(batch, selected_attr_key)
                if batch_att_array.shape[-1] != sum_channels:
                    are_all_same_as_sum_channels = False
                dac_arr: dac.Array = dac.from_array(batch_att_array.numpy())
                cat_arrays.append(dac_arr)

        is_norm_dim_different_from_channel_dim: bool = norm_dim != -1 and norm_dim != len(cat_arrays[0].shape) - 1
        do_group_norm: bool = is_norm_dim_different_from_channel_dim and are_all_same_as_sum_channels and self.batch_axis_choice != "snapshot"
        if is_norm_dim_different_from_channel_dim and not are_all_same_as_sum_channels and self.batch_axis_choice != "snapshot":
            print(
                f"WARN! Networks' arrays do not match their channels. Group norm by channels is prohibited! Flatten norm along {norm_dim} instead!\
                This could change behavior and lead to unwanted statistic!"
            )

        for i in range(len(cat_arrays)):
            arr = cat_arrays[i]
            cat_arrays[i] = arr.reshape([-1, arr.shape[-1]], limit=self.chunk_limit)

        # flatten_array has shape (-1, t+ 1 + t + ...)  or (-1)
        flatten_array = dac.concatenate(cat_arrays, axis=0)

        if do_group_norm:
            std_vals, mean_vals, min_vals, max_vals = [], [], [], []
            current_idx = 0
            for i in range(len(channel_splitters)):
                num_channels = channel_splitters[i]
                t = flatten_array[:, current_idx : current_idx + num_channels]

                # t = t.flatten()
                current_idx += num_channels

                t_std_val, t_mean_val = t.std(axis=norm_dim), t.mean(axis=norm_dim)
                # torch.std_mean(t, dim=norm_dim)
                t_min_val, t_max_val = t.min(axis=norm_dim), t.max(axis=norm_dim)
                # torch.min(t, dim=norm_dim).values, torch.max(t, dim=norm_dim).values

                # std_vals.append(t_std_val.reshape([-1]).repeat(num_channels))
                # mean_vals.append(t_mean_val.reshape([-1]).repeat(num_channels))
                # min_vals.append(t_min_val.reshape([-1]).repeat(num_channels))
                # max_vals.append(t_max_val.reshape([-1]).repeat(num_channels))

                if norm_dim is not None:
                    t_std_val = np.expand_dims(t_std_val, axis=norm_dim)

                    t_mean_val = np.expand_dims(t_mean_val, axis=norm_dim)

                    t_min_val = np.expand_dims(t_min_val, axis=norm_dim)

                    t_max_val = np.expand_dims(t_max_val, axis=norm_dim)
                else:
                    t_std_val = t_std_val.reshape([-1]).repeat(num_channels)
                    t_mean_val = t_mean_val.reshape([-1]).repeat(num_channels)
                    t_min_val = t_min_val.reshape([-1]).repeat(num_channels)
                    t_max_val = t_max_val.reshape([-1]).repeat(num_channels)

                std_vals.append(t_std_val)
                mean_vals.append(t_mean_val)
                min_vals.append(t_min_val)
                max_vals.append(t_max_val)

            std_val = dac.concatenate(std_vals, axis=channel_dim)
            mean_val = dac.concatenate(mean_vals, axis=channel_dim)
            min_val = dac.concatenate(min_vals, axis=channel_dim)
            max_val = dac.concatenate(max_vals, axis=channel_dim)
        else:
            std_val, mean_val = flatten_array.std(axis=norm_dim), flatten_array.mean(axis=norm_dim)
            min_val, max_val = flatten_array.min(axis=norm_dim), flatten_array.max(axis=norm_dim)

        (min_val, max_val, mean_val, std_val) = compute(*(min_val, max_val, mean_val, std_val))

        if not isinstance(min_val, np.ndarray):
            min_val = np.asarray(min_val, dtype=np.float32).reshape([1, -1])
            max_val = np.asarray(max_val, dtype=np.float32).reshape([1, -1])
            mean_val = np.asarray(mean_val, dtype=np.float32).reshape([1, -1])
            std_val = np.asarray(std_val, dtype=np.float32).reshape([1, -1])
        else:
            min_val = min_val.astype(np.float32)
            max_val = max_val.astype(np.float32)
            mean_val = mean_val.astype(np.float32)
            std_val = std_val.astype(np.float32)

        if to_tensor:
            std_val = from_numpy(std_val)
            mean_val = from_numpy(mean_val)
            min_val = from_numpy(min_val)
            max_val = from_numpy(max_val)

        # save to reuse
        # my_dict = {
        #     "min_val": min_val,
        #     "max_val": max_val,
        #     "mean_val": mean_val,
        #     "std_val": std_val,
        #     f"{which_array}_min_val": min_val,
        #     f"{which_array}_max_val": max_val,
        #     f"{which_array}_mean_val": mean_val,
        #     f"{which_array}_std_val": std_val,
        # }

        # if self.dataset_log_pt_path != "" and do_save:
        #     self.save_dataset_checkpoint(**my_dict)
        # else:
        #     print(
        #         "WARN! Statistic cannot be saved as `dataset_log_pt_path` is empty or `do_save` is set False in Gida Interface! You cannot re-load these stats in the inference or next train!",  # noqa: E501
        #         flush=True,
        #     )

        return (min_val, max_val, mean_val, std_val)

    def save_dataset_checkpoint(self, **kwargs: Any) -> str:
        assert self.dataset_log_pt_path[-4:] == ".pth" or self.dataset_log_pt_path[-3:] == ".pt"

        if os.path.exists(self.dataset_log_pt_path):
            # prevent overriden
            cp_dict: dict = load(self.dataset_log_pt_path)
            cp_dict.update(kwargs)
            save(cp_dict, self.dataset_log_pt_path)
        else:
            save(kwargs, self.dataset_log_pt_path)

        return self.dataset_log_pt_path

