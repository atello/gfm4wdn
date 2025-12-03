from __future__ import annotations
# from typing import Callable
# from train2 import train, eval, TrainOneEpoch, TestOneEpoch, WandbStartProfiler, SemiSingleForward
# from gigantic_dataset.utils.train_protos import (
#     default_post_foward_transform,
#     default_load_criterion,
#     default_load_optimizers,
#     default_load_scheduler,
# )


# class FuncRef:
#     def __init__(self):
#         self.start_profiler_fn = WandbStartProfiler()
#         self.forward_fn = SemiSingleForward()
#         self.train_one_epoch_fn = TrainOneEpoch()
#         self.test_one_epoch_fn = TestOneEpoch()
#         self.train_fn = train
#         self.eval_fn = eval
#         self.post_forward_tf_fn = default_post_foward_transform
#         self.load_criterion = default_load_criterion
#         self.load_optimizers = default_load_optimizers
#         self.load_scheduler = default_load_scheduler

#     @staticmethod
#     def from_dict(my_dict: dict[str, Callable], verbose: bool = False) -> FuncRef:
#         func_ref = FuncRef()
#         for k, v in my_dict.items():
#             if hasattr(func_ref, k):
#                 setattr(func_ref, k, v)
#                 if verbose:
#                     print(f"SUCCESS! {k} is found and updated!")
#             else:
#                 if verbose:
#                     print(f"FAIL! {k} is unfound!")

#         return func_ref


from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigantic_dataset.utils.train_protos import (
        StartProfilerProto,
        ForwardProto,
        TrainOneEpochProto,
        TestOneEpochProto,
        TrainProto,
        EvalProto,
        PostForwardTransformProto,
        LoadCriterionProto,
        LoadDatasetsProto,
        LoadModelProto,
        LoadOptimizersProto,
        LoadSchedulerProto,
    )


@dataclass
class FuncRef:
    start_profiler_fn: StartProfilerProto  # start profiler fn
    forward_fn: ForwardProto  # forward fn
    train_one_epoch_fn: TrainOneEpochProto  # train one epoch fn
    test_one_epoch_fn: TestOneEpochProto  # test one epoch fn
    train_fn: TrainProto  # train fn
    eval_fn: EvalProto  # test fn
    post_forward_tf_fn: PostForwardTransformProto  # useful for converting ytrue, ypred. Default is an identity.
    load_criterion: LoadCriterionProto  # support simple loading loss function mse, mae, ce
    load_datasets: LoadDatasetsProto  # support loading gida datsets
    load_models: LoadModelProto  # casual gcn
    load_optimizers: LoadOptimizersProto  # support loading optims per model
    load_scheduler: LoadSchedulerProto  # load scheduler. Default is None.
