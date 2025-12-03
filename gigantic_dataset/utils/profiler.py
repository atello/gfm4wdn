#
# Created on Mon Mar 18 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: A profiler allows an online analysis
# ------------------------------
#
from typing import Any
import numpy as np
from collections import OrderedDict
import json
from time import time
import psutil

ONE_GB_IN_BYTES = 1024**3


class WDNProfiler:
    def __init__(self, profiler_path: str | None = None) -> None:
        self.profiler_path = profiler_path
        self.stat_dict = OrderedDict()

        self.global_records = OrderedDict()  # zarr.MemoryStore(root=profiler_path)

    def _collect_stats(self, param_values: np.ndarray, **kwargs) -> dict:
        min_val = np.min(param_values)
        max_val = np.max(param_values)
        q1_val = np.quantile(param_values, 0.25)
        q3_val = np.quantile(param_values, 0.75)
        mean_val = np.mean(param_values)
        std_val = np.std(param_values)

        ret_dict = {
            "min": min_val,
            "max": max_val,
            "q1": q1_val,
            "q3": q3_val,
            "mean": mean_val,
            "std": std_val,
            "len": len(param_values),
        }

        ret_dict.update(kwargs)

        return ret_dict

    def collect(
        self,
        param_name: str,
        component_name: str,
        wdn_name: str,
        param_values: np.ndarray,
        num_objects: int,
        track_global: bool = True,
    ):
        compo_param_key = f"{component_name}+{param_name}"
        if compo_param_key not in self.stat_dict:
            self.stat_dict[compo_param_key] = OrderedDict()
            self.stat_dict[compo_param_key][wdn_name] = OrderedDict()

        if wdn_name not in self.stat_dict[compo_param_key]:
            self.stat_dict[compo_param_key][wdn_name] = OrderedDict()

        tmp_dict = self._collect_stats(param_values, num_objects=num_objects)

        my_dict: OrderedDict = self.stat_dict[compo_param_key][wdn_name]
        my_dict.update(tmp_dict)

        if track_global:
            gr_keys = list(self.global_records.keys())
            if compo_param_key not in gr_keys:
                self.global_records[compo_param_key] = param_values.flatten()  #
            else:
                prev_array = self.global_records[compo_param_key]
                arr = np.concatenate([prev_array, param_values.flatten()])  # type:ignore
                self.global_records[compo_param_key] = arr

    def collect_globally(self, global_key: str = "global"):
        gr_keys = list(self.global_records.keys())
        for compo_param_name in self.stat_dict:
            if compo_param_name in gr_keys:
                records = self.global_records[compo_param_name]
                self.stat_dict[compo_param_name][global_key] = self._collect_stats(records)

    def export(self, json_path: str | None = None):
        def pretty(d, indent=0):
            # ref: https://stackoverflow.com/a/3229493/4229525
            for key, value in d.items():
                print("\t" * indent + str(key))
                if isinstance(value, dict):
                    pretty(value, indent + 1)
                else:
                    print("\t" * (indent + 1) + str(value))

        if json_path is None:
            pretty(self.stat_dict)
        else:
            with open(json_path, "w") as f:
                json.dump(self.stat_dict, f)


class Watcher:
    def __init__(self, overwatch: bool = False) -> None:
        self.start_time: float = 0
        self.elapsed_time: float = 0
        self.lap_times: list[float] = []
        self.auxil_dict: OrderedDict[str, Any] = OrderedDict()
        self.overwatch = overwatch
        self.do_valid_stop: bool = False

    def _dump(self) -> None:
        """Dump resource"""
        if self.overwatch:
            current_timestamp = time()
            self.auxil_dict[f"mem_avail_gb+{current_timestamp:.2f}"] = psutil.virtual_memory().available / ONE_GB_IN_BYTES
            self.auxil_dict[f"mem_used_gb+{current_timestamp:.2f}"] = psutil.virtual_memory().used / ONE_GB_IN_BYTES
            self.auxil_dict[f"mem_total_gb+{current_timestamp:.2f}"] = psutil.virtual_memory().total / ONE_GB_IN_BYTES

    def start(self) -> float:
        if self.start_time <= 0:
            self.start_time = time()

        self._dump()
        return self.start_time

    def lap(self) -> float:
        end_time = time()
        lap_time = end_time - self.start_time
        self.lap_times.append(lap_time)
        self.elapsed_time += lap_time
        self._dump()
        return lap_time

    def stop(self, do_valid_stop: bool = True) -> float:
        """Return elapsed time"""
        self.do_valid_stop = do_valid_stop
        if self.elapsed_time > 0:
            lap_time = self.watch()
            self._dump()
            return lap_time
        else:
            end_time = time()
            self.elapsed_time = end_time - self.start_time
            self._dump()
            return self.elapsed_time

    def watch(self) -> float:
        return self.elapsed_time

    def reset(self) -> None:
        self.start_time = 0
        self.elapsed_time = 0
        self.lap_times.clear()
        self.auxil_dict.clear()
        self.do_valid_stop = False

    def report_mem(self, key: str = "", indent: int = 50, precision: int = 2) -> str:
        ret = ""
        if len(self.auxil_dict) > 0:
            timestamp_dict = {}
            for k in self.auxil_dict:
                timestamp = k.split("+")[1]
                if timestamp not in timestamp_dict:
                    timestamp_dict[timestamp] = []
                timestamp_dict[timestamp].append(k)

            timestamp_keys = list(sorted(timestamp_dict.keys()))

            for timestamp in timestamp_keys:
                # ret += f"{key}-{timestamp:<{indent}}:"
                # ret += f"{k:<{indent}}:{elapsed:>{indent}.{precision}f} s | Valid: {w.do_valid_stop}\n"

                tmp = ""
                for v in timestamp_dict[timestamp]:
                    refined_prop = v.split("+")[0].split("_")[1]
                    tmp += f"{self.auxil_dict[v]:.{precision}f} ({refined_prop}) "

                left = f"{key}({timestamp}):"
                ret += f"{left:<{indent}} {tmp:>{indent}}\n"

        return ret


class WatcherManager:
    _watcher_dict: OrderedDict[str, Watcher] = OrderedDict()
    overwatch: bool = False

    @staticmethod
    def track(name: str, verbose: bool = False) -> Watcher | None:
        watcher = None
        if name not in WatcherManager._watcher_dict:
            watcher = Watcher(WatcherManager.overwatch)
            ts = watcher.start()
            WatcherManager._watcher_dict[name] = watcher

            if verbose:
                print(f"WatcherManager-Created a watcher ({name}) at ({ts:.4f})")

        return watcher

    @staticmethod
    def stop_all() -> None:
        for k, w in WatcherManager._watcher_dict.items():
            if not w.do_valid_stop:
                w.stop(do_valid_stop=False)

    @staticmethod
    def reset_all() -> None:
        for k, w in WatcherManager._watcher_dict.items():
            w.reset()

    @staticmethod
    def stop(name, do_valid_stop: bool = True, verbose: bool = False) -> float:
        watcher = WatcherManager._watcher_dict.get(name)
        if watcher is None:
            return -1
        else:
            ts = watcher.stop(do_valid_stop=do_valid_stop)
            if verbose:
                print(f"WatcherManager-Stopped a watcher ({name}) valid ({do_valid_stop}) passed ({ts:.4f}) sec")
            return ts

    @staticmethod
    def reset(name) -> None:
        watcher = WatcherManager._watcher_dict.get(name)
        if watcher is not None:
            watcher.reset()

    @staticmethod
    def report(indent: int = 30, precision: int = 4) -> str:
        ret: str = "#" * 50 + "Watcher report: " + "#" * 50 + "\n"
        for k, w in WatcherManager._watcher_dict.items():
            elapsed = w.watch()
            ret += f"{k:<{indent}}:{elapsed:>{indent}.{precision}f} s | Finished: {w.do_valid_stop}\n"

        if WatcherManager.overwatch:
            ret += "#" * 50 + "Auxiliary overwatch: " + "#" * 50 + "\n"
            for k, w in WatcherManager._watcher_dict.items():
                ret += w.report_mem(key=k)

        return ret
