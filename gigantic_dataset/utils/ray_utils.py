#
# Created on Thu Oct 17 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: generalized Ray support purpose
# Note: Do not merge with auxil.py since this need to import ray, while auxil doesn't need
# ------------------------------
#
from multiprocessing import cpu_count
from typing import Any, Callable, Protocol
import numpy as np
import ray
from ray._private.internal_api import memory_summary


class ChunkFnProto(Protocol):
    def __call__(self, chunk_size: int, fn: Callable, *args: Any, **kwds: Any) -> Any:
        pass


class DefaultChunkFn:
    def __call__(self, chunk_size: int, fn: Callable, *args: Any, **kwds: Any) -> Any:
        records = []
        for _ in range(chunk_size):
            record = fn(*args, **kwds)
            records.append(record)
        chunk = np.vstack(records)
        return chunk


def ray_mapping(fn: Callable, num_cpus: int, num_samples: int, chunk_fn: ChunkFnProto = DefaultChunkFn(), *args, **kwargs) -> Any:
    num_allocated_cpus = min(num_cpus, cpu_count())

    if num_allocated_cpus == 1:
        return chunk_fn(chunk_size=num_samples, fn=fn, *args, **kwargs)
    else:

        @ray.remote
        def raying_fn(chunk_size: int, *args, **kwargs) -> Any:
            return chunk_fn(chunk_size=chunk_size, fn=fn, *args, **kwargs)

        num_chunks = min(num_allocated_cpus, num_samples)
        chunk_size = num_samples // num_chunks
        # WARNING: we employ all memory at all cost to compensate for the bottleneck
        futures = [
            raying_fn.remote(  # type:ignore
                chunk_size=chunk_size if c < num_chunks - 1 else num_samples - (c * chunk_size),  # type:ignore
                *args,
                **kwargs,
            )
            for c in range(num_chunks)
        ]
        results = ray.get(futures)

        while len(futures) > 0:
            ray_ref = futures.pop()
            # ray.cancel(ray_ref, force=True)
            del ray_ref
        del futures

        # pattern_list has shape (num_samples, obj_names_len, time_dim)
        new_values = np.concatenate(results, axis=0) if len(results) > 1 else results[0]
        return new_values


def check_usage_if_verbose(title: str = "", verbose: bool = False) -> str:
    # print("Plasma memory usage 0 MiB" in memory_summary(stats_only=True))
    ret = ""
    if verbose and ray.is_initialized():
        if title != "":
            print("*" * 20 + title + "*" * 20)

        ret = memory_summary(stats_only=True)
        print(ret)

    return ret
