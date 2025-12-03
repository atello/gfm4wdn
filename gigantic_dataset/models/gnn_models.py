#
# Created on Mon Nov 25 2024
# Copyright (c) 2024 Huy Truong & Andrés Tello
# ------------------------------
# Purpose: a simple GATRes
# ------------------------------
#


from typing import Any, Literal, Tuple

from torch import clone, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, ReLU, Sequential
from torch_geometric.nn import GATConv, SimpleConv, Linear, BatchNorm, GraphNorm, StdAggregation, GCNConv, \
    LayerNorm, MLP, PositionalEncoding
from gigantic_dataset.utils.configs import ModelConfig
from gigantic_dataset.utils.train_protos import ConfigRef, LoadModelProto
from gigantic_dataset.utils.train_utils import load_weights
import os
import torch

from torch.nn import Module


class LoadStructuralEncoder(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = StructuralEncoder(
                in_dim=in_dim,
                out_dim=out_node_dim,
                hidden_channels=model_config.nc,
                **kwds,
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(
                    model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)
        return modules


class LoadStructuralEncoderFinetune(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = StructuralEncoder(
                in_dim=in_dim,
                out_dim=out_node_dim,
                hidden_channels=model_config.nc,
                **kwds,
            )
            setattr(model, "name", model_config.name)
            if model_config.weight_path != "" and os.path.exists(model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        # if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
        #         train_config.save_path):
        #     filter_word = "best" if do_load_best else "last"
        #     model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
        #                           "training_log" not in entry and filter_word in entry]
        #     if len(model_weight_paths) > 0:
        #         assert len(model_weight_paths) == len(modules)
        #         for i in range(len(modules)):
        #             matching_order_paths = [path for path in model_weight_paths if str(i) in path]
        #             matching_order_path = matching_order_paths[0]
        #             tmps, _ = load_weights(
        #                 path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
        #                 load_keys=[model_configs[i].name]
        #             )
        #             modules[i] = tmps[0].to(train_config.device)
        return modules


class LoadFunctionalEncoder(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = FunctionalEncoderModel(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_chanels=model_config.nc,
                num_blocks=model_config.num_layers
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(
                    model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)
        return modules


class LoadGATRes(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = GATResMeanConv(
                nc=model_config.nc,
                num_blocks=model_config.num_layers,
                in_dim=in_dim,
                out_dim=out_node_dim,
                **kwds,
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(
                    model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return modules


class LoadGraphWaterSE(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = GraphWaterSE(
                nc=model_config.nc,
                num_blocks=model_config.num_layers,
                in_dim=in_dim,
                out_dim=out_node_dim,
                **kwds,
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(
                    model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return modules


class LoadGraphWaterSEPretrainedEncoder(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_node_dim = out_dims[0]
        modules = []
        for model_config in model_configs:
            model = GraphWaterSE(
                nc=model_config.nc,
                num_blocks=model_config.num_layers,
                in_dim=in_dim,
                out_dim=out_node_dim,
                **kwds,
            )
            setattr(model, "name", model_config.name)
            if load_weights_from == "model_config" and model_config.weight_path != "" and os.path.exists(
                    model_config.weight_path):
                models, _ = load_weights(path=model_config.weight_path, models=[model], load_keys=[model_config.name])
                model = models[0]
            modules.append(model)
        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        encoder = GATResMeanConv(
            nc=model_config.nc,
            num_blocks=model_config.num_layers,
            in_dim=in_dim,
            out_dim=out_node_dim,
            **kwds,
        )

        if model_config.weight_path != "" and os.path.exists(
                model_config.weight_path):
            encoder_models, _ = load_weights(path=model_config.weight_path, models=[encoder], load_keys=["gatres"])
            encoder = encoder_models[0]

            modules[0].encoder = encoder

        return modules


class LoadGraphWaterDualEncoder(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = GraphWaterDualEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            nc=model_configs[0].nc,
            name=model_configs[0].name,
        )

        modules.append(model)

        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return [modules[0]]


class LoadDualEncoderGFM(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = DualEncoderGFM(
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            hidden_channels=model_configs[0].nc,
            name=model_configs[0].name,
        )
        setattr(model, "name", model_configs[0].name)

        modules.append(model)

        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return [modules[0]]


class LoadGraphWaterDualEncoderFinetune(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = GraphWaterDualEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            nc=model_configs[0].nc,
            name=model_configs[0].name,
        )

        if model_configs[0].weight_path != "" and os.path.exists(model_configs[0].weight_path):
            models, _ = load_weights(path=model_configs[0].weight_path,
                                     models=[model],
                                     load_keys=[model_configs[0].name])
            modules.append(models[0])
        else:
            raise Exception("PATH of model weights is not provided in config.")

        # for param in modules[0].structure_encoder.parameters():
        #     param.requires_grad = False
        # for param in modules[0].pressure_encoder.parameters():
        #     param.requires_grad = False
        #
        # modules[0].lin0 = Linear(modules[0].lin0.in_channels, modules[0].lin0.out_channels)
        # modules[0].pressure_encoder.lin0 = Linear(modules[0].pressure_encoder.lin0.in_channels, modules[0].pressure_encoder.lin0.out_channels)
        # modules[0].decoder.head = Linear(modules[0].decoder.head.in_channels, modules[0].decoder.head.out_channels)

        return [modules[0]]


class LoadGraphWaterDualEncoderPretrainedStructural(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = GraphWaterDualEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            nc=model_configs[0].nc,
            name=model_configs[0].name,
        )

        modules.append(model)

        model = StructuralEncoderModel(
            in_dim=2,
            hidden_chanels=model_configs[0].nc,
            name=model_configs[1].name,
        )

        if model_configs[1].weight_path != "" and os.path.exists(model_configs[1].weight_path):
            models, _ = load_weights(path=model_configs[1].weight_path,
                                     models=[model],
                                     load_keys=[model_configs[1].name])
            modules.append(models[0])

        modules[0].structure_encoder = modules[1].encoder

        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return [modules[0]]


class LoadGraphWaterDualEncoderPretrainedStructuralFunctional(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = GraphWaterDualEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            nc=model_configs[0].nc,
            name=model_configs[0].name,
        )

        modules.append(model)

        # load structural encoder weights
        model = StructuralEncoderModel(
            in_dim=2,
            hidden_chanels=model_configs[0].nc,
            name=model_configs[1].name,
        )

        if model_configs[1].weight_path != "" and os.path.exists(model_configs[1].weight_path):
            models, _ = load_weights(path=model_configs[1].weight_path,
                                     models=[model],
                                     load_keys=[model_configs[1].name])
            modules.append(models[0])

        # load functional encoder weights
        model = FunctionalEncoderModel(
            in_dim=3,
            out_dim=out_dim,
            hidden_chanels=model_configs[0].nc,
            num_blocks=model_configs[0].num_layers,
            name=model_configs[2].name,
        )

        if model_configs[2].weight_path != "" and os.path.exists(model_configs[2].weight_path):
            models, _ = load_weights(path=model_configs[2].weight_path,
                                     models=[model],
                                     load_keys=[model_configs[2].name])
            modules.append(models[0])

        modules[0].structure_encoder = modules[1].encoder
        modules[0].pressure_encoder = modules[2].encoder

        if load_weights_from == "train_config" and train_config.save_path != "" and os.path.exists(
                train_config.save_path):
            filter_word = "best" if do_load_best else "last"
            model_weight_paths = [entry for entry in os.listdir(train_config.save_path) if
                                  "training_log" not in entry and filter_word in entry]
            if len(model_weight_paths) > 0:
                assert len(model_weight_paths) == len(modules)
                for i in range(len(modules)):
                    matching_order_paths = [path for path in model_weight_paths if str(i) in path]
                    matching_order_path = matching_order_paths[0]
                    tmps, _ = load_weights(
                        path=os.path.join(train_config.save_path, matching_order_path), models=[modules[i]],
                        load_keys=[model_configs[i].name]
                    )
                    modules[i] = tmps[0].to(train_config.device)

        return [modules[0]]


class GraphWaterDualEncoder(Module):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, in_dim: int, out_dim: int, name: str = "GraphWater", num_blocks: int = 5, nc: int = 32,
                 **kwargs):
        super(GraphWaterDualEncoder, self).__init__()
        # self.in_dim = in_dim
        # self.out_dim = out_dim
        # self.num_blocks = num_blocks
        # self.nc = nc
        self.name = name
        # self.kwargs = kwargs,

        self.lin0 = Linear(in_channels=2, out_channels=nc)

        self.structure_encoder = StructuralEncoderGCN(
            in_dim=nc,
            out_dim=nc,
            hidden_chanels=nc
        )

        self.pressure_encoder = GATResMeanConv(
            in_dim=3,
            out_dim=nc,
            num_blocks=num_blocks,
            nc=nc
        )

        self.decoder = DownstreamTaskHead(in_dim=nc * 2, out_dim=out_dim)

    def forward(self, x_struct: Tensor, x_func: Tensor,
                edge_index: Tensor, edge_index_undirected: Tensor,
                batch: Tensor | None = None,
                edge_attr: Tensor | None = None) -> Tensor:
        x_struct = self.lin0(x_struct)
        h_structural = self.structure_encoder(x_struct, edge_index_undirected)
        h_functional = self.pressure_encoder(x_func, edge_index, edge_attr)
        h = torch.cat([h_structural, h_functional], dim=1)
        out = self.decoder(h)
        return out


class GraphWaterSE(Module):
    def __init__(self, in_dim: int, out_dim: int, name: str = "GraphWater", num_blocks: int = 5, nc: int = 32,
                 **kwargs):
        super(GraphWaterSE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.nc = nc

        self.encoder = GATResMeanConv(
            nc=self.nc,
            num_blocks=self.num_blocks,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            **kwargs,
        )

        self.pretrain_head = PretrainHead(self.nc)
        self.task_head = DownstreamTaskHead(self.nc)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None,
                edge_attr: Tensor | None = None) -> Tensor:
        h = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
        pre_output = self.pretrain_head(h[1])
        task_output = self.task_head(h[1])
        return pre_output, task_output


class PretrainHead(Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_deg_head = Linear(hidden_dim, 1)
        self.neigh_deg_head = Linear(hidden_dim, 1)

    def forward(self, x):
        pred_node_deg = self.node_deg_head(x).squeeze(-1)
        pred_neigh_deg = self.neigh_deg_head(x).squeeze(-1)
        return pred_node_deg, pred_neigh_deg


class DownstreamTaskHead(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = Linear(in_channels=in_dim, out_channels=out_dim)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class GResBlockMeanConv(Module):
    def __init__(self, in_dim, out_dim, hc, **kwargs):
        super(GResBlockMeanConv, self).__init__()

        add_self_loops = kwargs.get('add_self_loops', True)

        self.conv1 = GATConv(in_channels=in_dim, out_channels=hc, heads=2, edge_dim=2, concat=True,
                             add_self_loops=add_self_loops)
        self.conv2 = GATConv(in_channels=hc * 2, out_channels=out_dim, heads=1, edge_dim=2, concat=False,
                             add_self_loops=add_self_loops)
        self.mean_conv = SimpleConv(aggr="mean")

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
        x_0 = clone(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.mean_conv(x, edge_index) + x_0
        x = F.relu(x)
        return x


class GRBlockNorm(Module):
    def __init__(self, in_dim, out_dim, hc, **kwargs):
        super(GRBlockNorm, self).__init__()

        self.add_self_loops = kwargs.get('add_self_loops', True)
        self.edge_dim = kwargs.get('edge_dim', None)

        self.conv1 = GATConv(in_channels=in_dim, out_channels=hc, heads=2, edge_dim=self.edge_dim, concat=True, add_self_loops=self.add_self_loops)
        self.norm1 = LayerNorm(in_channels=hc * 2)
        self.conv2 = GATConv(in_channels=hc * 2, out_channels=out_dim, heads=1, edge_dim=self.edge_dim, concat=False, add_self_loops=self.add_self_loops)
        self.norm2 = LayerNorm(in_channels=out_dim)
        self.mean_conv = SimpleConv(aggr="mean")

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
        x_0 = clone(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.mean_conv(x, edge_index) + x_0
        x = F.relu(x)
        return x


class GATResMeanConv(Module):
    def __init__(self, in_dim: int, out_dim: int, name: str = "GATResMeanConv", num_blocks: int = 5, nc: int = 32,
                 **kwargs):
        super(GATResMeanConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.nc = nc

        self.lin0 = Linear(in_dim, nc)
        self.blocks = ModuleList()
        self.name = name

        add_self_loops = kwargs.get('add_self_loops', True)
        edge_dim = kwargs.get('edge_dim', None)

        for _ in range(self.num_blocks):
            # block = GResBlockMeanConv(in_dim=nc, out_dim=nc, hc=nc, add_self_loops=add_self_loops)
            block = GRBlockNorm(in_dim=nc, out_dim=nc, hc=nc, edge_dim=edge_dim, add_self_loops=add_self_loops)
            self.blocks.append(block)

        self.lin1 = Linear(nc, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None,
                edge_attr: Tensor | None = None) -> Tensor:
        x = self.lin0(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        embeddings = clone(x)
        x = self.lin1(x)
        return x, embeddings


# class MLP(Module):
#     def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None,
#                 edge_attr: Tensor | None = None) -> Tensor:
#         assert edge_attr is not None
#         # take node id pairs
#         src, dst = edge_index[0], edge_index[1]
#
#         # get their feature and concat with edge feature (assume shape is perfectly fit)
#         cat_x = torch.cat([x[src], x[dst], edge_attr], dim=-1)
#
#         # your mlp architecture
#         out = self.actual_mlp(cat_x)
#
#         return out


class StructuralEncoderGCN(Module):
    def __init__(self, in_dim, out_dim, hidden_chanels):
        super(StructuralEncoderGCN, self).__init__()

        self.conv1 = GCNConv(in_channels=in_dim, out_channels=hidden_chanels)
        self.norm1 = LayerNorm(in_channels=hidden_chanels)
        self.conv2 = GCNConv(in_channels=hidden_chanels, out_channels=out_dim)

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x


# class StructuralEncoderGCN(Module):
#     def __init__(self, in_dim, out_dim, hidden_chanels):
#         super(StructuralEncoderGCN, self).__init__()
#
#         self.conv1 = GCNConv(in_channels=in_dim, out_channels=hidden_chanels * 2)
#         self.conv2 = GCNConv(in_channels=hidden_chanels * 2, out_channels=out_dim)
#
#     def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
#         x = self.conv1(x, edge_index, edge_attr).relu()
#         x = self.conv2(x, edge_index, edge_attr)
#         return x


class StructuralEncoderGAT(Module):
    def __init__(self, in_dim, hidden_chanels, out_dim):
        super(StructuralEncoderGAT, self).__init__()

        self.conv1 = GATConv(in_channels=in_dim, out_channels=hidden_chanels, heads=2)
        self.norm1 = LayerNorm(hidden_chanels * 2)
        self.conv2 = GATConv(in_channels=hidden_chanels * 2, out_channels=out_dim, heads=1)

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> Tensor:
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x


class StructuralEncoder(Module):
    def __init__(self, in_dim, out_dim, hidden_channels, **kwargs):
        super(StructuralEncoder, self).__init__()

        # self.pe = PositionalEncoding(out_channels=hidden_channels)
        # self.stem = Linear(in_channels=in_dim, out_channels=hidden_channels)
        self.encoder = GATResMeanConv(in_dim=in_dim, out_dim=out_dim, nc=hidden_channels, edge_dim=3, add_self_loops=True)

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x_pe = self.pe(x)
        # x_stem = self.stem(x)
        # x = torch.cat([x_pe, x_stem], dim=1)
        x, emb = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        return x, emb


# class StructuralEncoderModel(Module):
#     def __new__(cls, *args, **kwargs):
#         return super().__new__(cls)
#
#     def __init__(self, in_dim, hidden_chanels, **kwargs):
#         super(StructuralEncoderModel, self).__init__()
#
#         self.lin0 = Linear(in_dim, hidden_chanels)
#         self.encoder = StructuralEncoderGCN(in_dim=hidden_chanels, out_dim=hidden_chanels,
#                                             hidden_chanels=hidden_chanels)
#         self.pretrain_head = PretrainHead(hidden_dim=hidden_chanels)
#
#     def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> tuple[Tensor, Tensor]:
#         x = self.lin0(x)
#         x = self.encoder(x, edge_index, edge_attr)
#         pred_node_deg, pred_neigh_deg = self.pretrain_head(x)
#         return pred_node_deg, pred_neigh_deg


class FunctionalEncoderModel(Module):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, in_dim, out_dim, hidden_chanels, **kwargs):
        super(FunctionalEncoderModel, self).__init__()

        num_blocks = kwargs.get("num_blocks", 15)

        self.encoder = GATResMeanConv(in_dim=in_dim, out_dim=hidden_chanels, nc=hidden_chanels, num_blocks=num_blocks)
        self.pressure_head = DownstreamTaskHead(in_dim=hidden_chanels, out_dim=out_dim)

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x, emb = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.pressure_head(emb)
        return x


#####################################################################################################################

class GraphStructureEncoder(Module):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, in_dim, hidden_chanels, **kwargs):
        super(GraphStructureEncoder, self).__init__()

        self.stem = Linear(in_dim, hidden_chanels)
        self.encoder = StructuralEncoderGAT(in_dim=hidden_chanels, hidden_chanels=hidden_chanels)
        self.pretrain_head = PretrainHead(hidden_dim=hidden_chanels)

    def forward(self, x, edge_index, edge_attr=None, batch: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        embeddings = self.encoder(x, edge_index, edge_attr)
        pred_node_deg, pred_neigh_deg = self.pretrain_head(x)
        return pred_node_deg, pred_neigh_deg, embeddings


class DualEncoderGFM(Module):
    def __init__(self, out_dim: int, hidden_channels: int, num_blocks: int, **kwargs):
        super(DualEncoderGFM, self).__init__()

        self.structure_encoder = GATResMeanConv(in_dim=1, out_dim=2, nc=hidden_channels, num_blocks=num_blocks, edge_dim=None, add_self_loops=False)
        self.pressure_encoder = GATResMeanConv(in_dim=3, out_dim=1, nc=hidden_channels, num_blocks=num_blocks, edge_dim=3, add_self_loops=True)
        # self.struct_bias = Linear(hidden_channels, hidden_channels)

        # self.task_head = Linear(in_channels=hidden_channels * 2, out_channels=out_dim)
        self.task_head = Linear(in_channels=hidden_channels, out_channels=out_dim)

    def forward(self, x_struct: Tensor, x_func: Tensor,
                edge_index: Tensor,
                batch: Tensor | None = None,
                edge_attr: Tensor | None = None) -> tuple[Any, Any, Any]:

        degrees, h_degree = self.structure_encoder(x=x_struct, edge_index=edge_index, edge_attr=None, batch=batch)
        intermediate_pressure, h_pressure = self.pressure_encoder(x=x_func, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        # h = torch.cat([h_degree, h_pressure], dim=1)
        # pressure = self.task_head(h)

        # struct_bias = self.struct_bias(h_degree).relu()
        h_pressure_biased = h_pressure + h_degree
        pressure = self.task_head(h_pressure_biased)
        return degrees, intermediate_pressure.squeeze(1), pressure.squeeze(1)


class LoadDualEncoderGFMFineTune(LoadModelProto):
    def __call__(
            self,
            in_dims: list[int],
            out_dims: list[int],
            load_weights_from: Literal["model_config", "train_config"] = "train_config",
            do_load_best: bool = True,
            **kwds: Any,
    ) -> list[Module]:
        train_config = ConfigRef.config
        model_configs: list[ModelConfig] = train_config.model_configs
        in_dim = in_dims[0]
        out_dim = out_dims[0]
        modules = []

        model = DualEncoderGFM(
            out_dim=out_dim,
            num_blocks=model_configs[0].num_layers,
            hidden_channels=model_configs[0].nc,
            name=model_configs[0].name,
        )
        setattr(model, "name", model_configs[0].name)
        if model_configs[0].weight_path != "" and os.path.exists(model_configs[0].weight_path):
            models, _ = load_weights(path=model_configs[0].weight_path, models=[model],load_keys=[model_configs[0].name])
            model = models[0]
        modules.append(model)

        return [modules[0]]
