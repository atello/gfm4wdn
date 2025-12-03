import argparse
import logging
import math
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import wntr
import torch
import wandb
from torch.ao.pruning.scheduler import lambda_scheduler
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from gigantic_dataset.models.gnn_models import DualEncoderGFM, GATResMeanConv
from gigantic_dataset.utils.gen_random_mask_v8 import generate_batch_mask
from gigantic_dataset.utils.pretrain_utils import compute_node_degree, LambdaScheduler
from gigantic_dataset.utils.test_components import collect_all_params
from gigantic_dataset.utils.configs import SimConfig
from gigantic_dataset.utils.train_utils import apply_masks, get_default_metric_fn_collection, print_single_metrics, \
    log_metrics_on_wandb, save_checkpoint, print_metrics


class Scaler:
    def __init__(self, norm_type="znorm", f="scale", mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.f = f
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

        assert norm_type in ['minmax', "znorm"]
        assert f in ['scale', 'unscale']

    def scale_data(self, data):
        if self.norm_type == "minmax":
            assert self.min_val is not None and self.max_val is not None, "min and max values missing"
            if self.f == "scale":
                return (data - self.min_val) / (self.max_val - self.min_val)
            elif self.f == "unscale":
                return self.min_val + (data * (self.max_val - self.min_val))
        elif self.norm_type == "znorm":
            assert self.mean is not None and self.std is not None, "mean and std values missing"
            if self.f == "scale":
                return (data - self.mean) / self.std
            elif self.f == "unscale":
                return self.mean[0] + (data * self.std[0])


def train_one_epoch(loader, model, criterion, optimizer, scaler, epoch, total_epochs, lambda_scheduler, device=torch.device("cpu")):
    model.eval()

    total_loss = 0
    metric_fn_dict = get_default_metric_fn_collection(prefix="val", task="semi")
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}

    lambda_weight = lambda_scheduler.get_lambda(epoch=epoch - 1, total_epochs=total_epochs - 1, variant="cos")

    for batch_idx, data in enumerate(loader):  # Iterate in batches over the training/test ds_name.
        optimizer.zero_grad()

        if args.scheduler:
            scheduler.step(epoch + batch_idx / len(loader))

        num_nodes = torch.unique(data.batch, return_counts=True)[1]
        batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=0.95,
                                         required_mask=None)

        data = data.to(device)
        data.y = deepcopy(data.x[:, [0, 3, 4]])

        cols_to_mask = torch.tensor([0, 1, 3])  # pressure, demand
        mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        data.x[mask_index[:, None], cols_to_mask] = 0

        # data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce="mean")

        deg_pred, intermediate_pressure, pressure = model(x_struct=data.x[:, [3]], x_func=data.x[:, :3],
                                                          edge_index=data.edge_index,
                                                          batch=data.batch, edge_attr=data.edge_attr)

        deg_pred = apply_masks(deg_pred, [batch_mask])
        deg_true = data.y[mask_index, -2:]
        y_pred_pressure = apply_masks(pressure, [batch_mask])  # out[batch_mask] #type:ignore
        y_true_pressure = data.y[mask_index, 0]  # type:ignore

        deg_loss = criterion(deg_pred, deg_true)
        pressure_loss = criterion(y_pred_pressure, y_true_pressure)
        # tr_loss = deg_loss + pressure_loss
        tr_loss = lambda_weight * deg_loss + (1 - lambda_weight) * pressure_loss

        tr_loss.backward()
        optimizer.step()

        # update metrics: re-scale
        y_pred_rescaled = scaler.scale_data(y_pred_pressure)
        y_true_rescaled = scaler.scale_data(y_true_pressure)

        with torch.no_grad():
            total_loss += float(tr_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                total_metric_dict[k] += computed_metric * data.num_graphs

    with torch.no_grad():
        dividend = max(1, len(loader.dataset))
        metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
        return total_loss / dividend, metric_dict


@torch.no_grad()
def test_one_epoch(loader, model, criterion, scaler, device=torch.device("cpu")):
    model.eval()

    total_loss = 0
    metric_fn_dict = get_default_metric_fn_collection(prefix="val", task="semi")

    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    for data in loader:  # Iterate in batches over the training/test ds_name.
        num_nodes = torch.unique(data.batch, return_counts=True)[1]
        batch_mask = generate_batch_mask(num_nodes=num_nodes, edge_index=data.edge_index, mask_rate=0.95, required_mask=None)

        data = data.to(device)
        data.y = deepcopy(data.x[:, [0, 3, 4]])

        cols_to_mask = torch.tensor([0, 1, 3])  # pressure, demand
        mask_index = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        data.x[mask_index[:, None], cols_to_mask] = 0

        # data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce="mean")

        deg_pred, intermediate_pressure, pressure = model(x_struct=data.x[:, [3]], x_func=data.x[:, :3],
                                                          edge_index=data.edge_index,
                                                          batch=data.batch, edge_attr=data.edge_attr)

        y_pred_pressure = apply_masks(pressure, [batch_mask])  # out[batch_mask] #type:ignore
        y_true_pressure = data.y[mask_index, 0]  # type:ignore

        val_loss = criterion(y_pred_pressure, y_true_pressure)

        # update metrics: re-scale
        y_pred_rescaled = scaler.scale_data(y_pred_pressure)
        y_true_rescaled = scaler.scale_data(y_true_pressure)

        with torch.no_grad():
            total_loss += float(val_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                total_metric_dict[k] += computed_metric * data.num_graphs

    with torch.no_grad():
        dividend = max(1, len(loader.dataset))
        metric_dict = {k: total_metric_dict[k] / dividend for k in total_metric_dict.keys()}
        return total_loss / dividend, metric_dict


def evaluate_final_model(loader, model, criterion, device, wandb_run, scaler):
    model.eval()
    # evaluate final model on hold out test set

    test_metrics_array = []
    for _ in range(5):
        test_loss, test_metric_dict = test_one_epoch(loader=loader, model=model, criterion=criterion, device=device, scaler=scaler)

        print_single_metrics(
            epoch=0,
            test_loss=test_loss,
            test_metric_dict=test_metric_dict,
        )

        if wandb_run:
            log_metrics_on_wandb(
                epoch=0,
                commit=True,
                test_loss=test_loss,
                best_epoch=0,
                test_metric_dict=test_metric_dict,
            )

        test_metrics_array.append([test_metric_dict[k].item() for k, v in test_metric_dict.items()])

    mean_val = np.mean(test_metrics_array, axis=0)
    std_val = np.std(test_metrics_array, axis=0)

    for idx, item in enumerate(test_metric_dict.items()):
        print(f"{item[0]}: {mean_val[idx]:.4f} ±{std_val[idx]:.3f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
    parser.add_argument('--test_only', help="indicates whether run only test without fine-tuning", action="store_true")
    parser.add_argument('--zero_shot', help="run zero-shot test", action="store_true")
    parser.add_argument('--log_wandb', help="save logs in wandb", action="store_true")
    parser.add_argument('--scheduler', help="lr_scheduler", action="store_true")

    args = parser.parse_args()

PRETRAINED_MODEL_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/GFM-struct-bias-weightedLoss+29wdns+DualEncoderGFM+20250828_185115_1748178/best_DualEncoderGFM_0.pt"
TRAIN_STATS_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/GFM-struct-bias-weightedLoss+29wdns+DualEncoderGFM+20250828_185115_1748178/gida_dataset_log.pt"
# FINETUNED_MODEL_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/DualEncoderGFM-FT-cos-wLoss+WA1+DualEncoderGFM+20250629_171818_1231466/best_DualEncoderGFM_0.pt"
FINETUNED_MODEL_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/Fine-tune_INP_WA1_20250708_173254/best_DualEncoderGFM_0.pt"
YAML_PATH = f"gigantic_dataset/arguments/{args.ds_name}_baseline.yaml"
EPOCHS = 50

config = SimConfig().parse_args(known_only=True)
config._from_yaml(yaml_path=YAML_PATH, unsafe_load=True)

skip_names = config.skip_names

wn = wntr.network.WaterNetworkModel(config.inp_paths[0])
junction_idx = np.array(
    [i for i, n in enumerate(wn.node_name_list) if n in wn.junction_name_list and n not in skip_names])
reservoir_idx = np.array(
    [i for i, n in enumerate(wn.node_name_list) if n in wn.reservoir_name_list])
tank_idx = np.array(
    [i for i, n in enumerate(wn.node_name_list) if n in wn.tank_name_list])

inp_value_dict = collect_all_params(frozen_wn=wn,
                                    time_from='wn',
                                    config=None,
                                    exclude_skip_nodes_from_config=False,
                                    output_only=False,
                                    sim_output_keys=['demand', 'pressure'])

demand = torch.as_tensor(inp_value_dict["demand"])[:, junction_idx]
pressure = torch.as_tensor(inp_value_dict["pressure"])[:, junction_idx]
junc_elevation = torch.as_tensor(inp_value_dict["junction_elevation"])[junction_idx]
junc_elevation = junc_elevation.unsqueeze(0).expand(pressure.shape[0], -1)
# tank_elevation = torch.as_tensor(inp_value_dict["tank_elevation"]).unsqueeze(0).expand(pressure.shape[0], -1)
# reservoir_base_head = torch.as_tensor(inp_value_dict["reservoir_base_head"]).unsqueeze(0).expand(pressure.shape[0], -1)
# elevations = torch.cat([junc_elevation, reservoir_base_head, tank_elevation], dim=1)

pipe_diameter = torch.as_tensor(inp_value_dict["pipe_diameter"]).reshape(-1, 1)
pipe_length = torch.as_tensor(inp_value_dict["pipe_length"]).reshape(-1, 1)
pipe_roughness = torch.as_tensor(inp_value_dict["pipe_roughness"]).reshape(-1, 1)

node_attrs = torch.stack([pressure, demand, junc_elevation], dim=2)
edge_attrs = torch.hstack([pipe_diameter, pipe_length, pipe_roughness])

node_dict = {name: i for i, name in enumerate(wn.node_name_list)}

start_nodes = []
end_nodes = []
pipe_names = []
for edge in wn.pipes():
    start_node = edge[1].start_node_name
    end_node = edge[1].end_node_name
    if start_node in wn.junction_name_list and start_node not in skip_names and end_node in wn.junction_name_list and end_node not in skip_names:
        start_nodes.append(node_dict[start_node])
        end_nodes.append(node_dict[end_node])
        pipe_names.append(edge[0])

node_idx_ordered = {v: i for i, v in enumerate(junction_idx)}

edge_index = torch.tensor([[node_idx_ordered[node] for node in start_nodes], [node_idx_ordered[node] for node in end_nodes]])
edge_index, edge_attrs = to_undirected(edge_index=edge_index, edge_attr=edge_attrs)

degrees = compute_node_degree(edge_index=torch.as_tensor(edge_index), num_nodes=node_attrs.shape[1])
node_attrs = torch.cat([node_attrs, degrees.expand(node_attrs.shape[0], -1, -1)], dim=2)

# valid_node_attrs = []
# for i in range(node_attrs.shape[0]):
#     if (node_attrs[i][:, 0] <= 0).sum() > 0:
#         continue
#     valid_node_attrs.append(node_attrs[i])

# node_attrs = torch.tensor(np.array(valid_node_attrs))

if not args.zero_shot and not args.test_only:
    num_samples_train = 1  # int(len(node_attrs) * 0.05)
    num_samples_val = 1  # int(len(node_attrs) * 0.05)
    num_samples_test = len(node_attrs) - num_samples_train - num_samples_val

    print(f"num_train: {num_samples_train}, num_val: {num_samples_val}, num_test: {num_samples_test}")

    node_mean_val = torch.mean(node_attrs[:num_samples_train], dim=[0, 1], keepdim=False)
    node_std_val = torch.std(node_attrs[:num_samples_train], dim=[0, 1], keepdim=False)
    edge_mean_val = torch.mean(edge_attrs, dim=0, keepdim=False)
    edge_std_val = torch.std(edge_attrs, dim=0, keepdim=False)

    scaler_nodes = Scaler(norm_type="znorm", f="scale", mean=node_mean_val, std=node_std_val)
    scaler_edges = Scaler(norm_type="znorm", f="scale", mean=edge_mean_val, std=edge_std_val)

else:
    train_stats = torch.load(TRAIN_STATS_PATH)
    scaler_nodes = Scaler(norm_type="znorm", f="scale", mean=train_stats["node_mean_val"], std=train_stats["node_std_val"])
    scaler_edges = Scaler(norm_type="znorm", f="scale", mean=train_stats["edge_mean_val"], std=train_stats["edge_std_val"])

node_attrs = scaler_nodes.scale_data(node_attrs)
edge_attrs = scaler_edges.scale_data(edge_attrs)

scaler_nodes.f = "unscale"

datalist = []
for i in range(node_attrs.shape[0]):
    x = node_attrs[i]
    datalist.append(Data(x=x.float(), edge_index=edge_index.long(), edge_attr=edge_attrs.float()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_model = DualEncoderGFM(out_dim=1, hidden_channels=32, num_blocks=5)
# model = GATResMeanConv(in_dim=3, out_dim=1, hidden_channels=32, num_blocks=5)

# load best model
checkpoint = torch.load(PRETRAINED_MODEL_PATH)

target_model.load_state_dict(checkpoint['DualEncoderGFM'])
setattr(target_model, "name", "DualEncoderGFM")
target_model = target_model.to(device)

optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0005, weight_decay=0.00001)
criterion = torch.nn.MSELoss().to(device)

if args.test_only:
    test_loader = DataLoader(datalist, batch_size=32, shuffle=False)
else:
    train_data = datalist[:num_samples_train]
    val_data = datalist[num_samples_train:num_samples_train + num_samples_val]
    test_data = datalist[-num_samples_test:]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    lambda_scheduler = LambdaScheduler()

    if args.scheduler:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of epochs before the first restart
            T_mult=2,  # Multiply T_i by this after a restart (optional)
            eta_min=1e-6  # Minimum learning rate
        )

    postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

    if args.log_wandb:
        if wandb.run is not None:  # check if wandb is inititized
            wandb.finish(0, quiet=True)
        wandb.init(
            # Set the project where this run will be logged
            project="gfm-wdn_train",
            save_code=True,
            group="inp_models_FT",
            # tags=[args.ds_name, args.ds_variant, args.model_name],
            name=f"Fine-tune_INP_{args.ds_name}_{postfix}",
            # Track hyperparameters and run metadata
            config=config,
        )

    logging.info(f"Fine-tuning INP model: {args.ds_name}")
    logging.info('-' * 89)

    # intial records
    best_loss = np.inf
    best_metric_dict = {}
    best_epoch = 0
    best_save_path = ""
    last_save_path = ""
    best_model = deepcopy(target_model.state_dict())

    SAVE_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/"
    RUN_NAME = f"Fine-tune_INP_{args.ds_name}_{postfix}"

    for epoch in range(1, EPOCHS + 1):

        # print(f"Training @epoch {epoch}...")
        tr_loss, tr_metric_dict = train_one_epoch(loader=train_loader, model=target_model, criterion=criterion, optimizer=optimizer, scaler=scaler_nodes, epoch=epoch, total_epochs=EPOCHS, lambda_scheduler=lambda_scheduler, device=device)

        val_loss, val_metric_dict = test_one_epoch(loader=val_loader, model=target_model, criterion=criterion, device=device, scaler=scaler_nodes)

        if val_loss < best_loss:
            best_loss = val_loss
            best_metric_dict = val_metric_dict
            best_epoch = epoch
            best_model = deepcopy(target_model.state_dict())
            # save training_checkpoint

            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict=optimizer.state_dict(),
                epoch=best_epoch,
                loss=best_loss,
                val_metric_dict=best_metric_dict,
                norm_type="",
            )

            best_save_path = save_checkpoint(path=f"{SAVE_PATH}/{RUN_NAME}", models=[target_model], prefix="best", **save_kwargs)  # type:ignore

        if (epoch == 1 or (epoch % 1) == 0) and not math.isnan(tr_loss):
            print_metrics(
                epoch=epoch,
                tr_loss=tr_loss,
                val_loss=val_loss,
                tr_metric_dict=tr_metric_dict,
                val_metric_dict=val_metric_dict,
            )
            save_kwargs = dict(  # noqa: C408
                optimizers_state_dict=optimizer.state_dict(),  # type:ignore
                epoch=epoch,  # type:ignore
                loss=tr_loss,  # type:ignore
                val_metric_dict=val_metric_dict,
                norm_type="",
            )
            last_save_path = save_checkpoint(path=f"{SAVE_PATH}/{RUN_NAME}", models=[target_model], prefix="last", **save_kwargs)  # type:ignore

        if args.log_wandb:
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

        if args.scheduler:
            scheduler.step(val_loss)

final_model = DualEncoderGFM(out_dim=1, hidden_channels=32, num_blocks=5)

if args.test_only:
    # load best model
    checkpoint = torch.load(FINETUNED_MODEL_PATH)
    final_model.load_state_dict(checkpoint['DualEncoderGFM'])
    setattr(target_model, "name", "DualEncoderGFM")
else:
    final_model.load_state_dict(best_model)

final_model = final_model.to(device)

if args.log_wandb:
    if wandb.run is not None:  # check if wandb is inititized
        wandb.finish(0, quiet=True)
    wandb.init(
        # Set the project where this run will be logged
        project="gfm-wdn_test",
        save_code=True,
        group="inp_models_evaluation",
        # tags=[args.ds_name, args.ds_variant, args.model_name],
        name=f"EVAL_INP_{args.ds_name}_{postfix}",
        # Track hyperparameters and run metadata
        config=config,
    )

logging.info(f"Testing INP model: {args.ds_name}")
logging.info('-' * 89)

evaluate_final_model(loader=test_loader, model=final_model, criterion=criterion, device=device, wandb_run=args.log_wandb, scaler=scaler_nodes)
print("baseline simulation complete")
