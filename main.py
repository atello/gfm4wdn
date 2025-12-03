#
# Created on Sat Jan 15 2025
# Copyright (c) 2025 Andrés Tello
# ------------------------------
# Purpose: main code
# ------------------------------
import argparse

from gigantic_dataset.core.run_dual_encoder import finetune_dual_encoder, dual_encoder_train, dual_encoder_inference, \
    dual_encoder_train_pretrained_encoders
from gigantic_dataset.core.run_shared_encoder import finetune_shared_encoder
# from gigantic_dataset.core.run_shared_encoder import train_shared_encoder, shared_enc_PE_inference, \
#     finetune_shared_encoder
from gigantic_dataset.utils.configs import *
from gigantic_dataset.core.run import pressure_estimation, pressure_estimation_inference
from gigantic_dataset.core.run_pretrain import pretraining_ndegree, pretrain_inference

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="", type=str)
    parser.add_argument('--dconf', default="", type=str)
    parser.add_argument('--mconf', default="", type=str)
    args = parser.parse_args()

    assert args.dconf is not None, "'--dconf' (data configuration file name) argument is required."
    assert args.mconf is not None, "''--mconf' (model configuration file name) argument is required."
    assert args.task in ['pretrain', "pretrain-inference", 'train', 'inference',
                         'shared-enc', "shared-enc-inference", "train_dual_enc", "dual_enc_inference",
                         "train_dual_pretrained_encoders", "finetune-shared-enc", "finetune-dual-enc"], "'--task' argument is required."

TASK = args.task

if TASK == "pretrain":
    pretraining_ndegree(
        f"gigantic_dataset/arguments/train/{args.dconf}",
        f"gigantic_dataset/arguments/train/{args.mconf}",
    )
elif TASK == "pretrain-inference":
    pretrain_inference(
        f"gigantic_dataset/arguments/train/{args.dconf}",
        f"gigantic_dataset/arguments/train/{args.mconf}",
        save_path=r"/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/PRETRAIN-MULTI-allfeat-edgeattr+6wdns+GraphWaterSE+20250609_081130_2844575_STRUCT_ENC",
        custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/PRETRAIN-MULTI-allfeat-edgeattr+6wdns+GraphWaterSE+20250609_081130_2844575_STRUCT_ENC/gida_dataset_log.pt",
    )
elif TASK == "train":
    pressure_estimation(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
        # save_path=r"/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295",
        # custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295/gida_dataset_log.pt",
        # custom_subset_shuffle_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295/gida_dataset_log.pt"
    )
elif TASK == "inference":
    pressure_estimation_inference(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
        save_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/BSLN-GATRes+ky8+GATRes+20250919_015950_146682",
        custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/BSLN-GATRes+ky8+GATRes+20250919_015950_146682/gida_dataset_log.pt",
    )
# elif TASK == "shared-enc":
#     train_shared_encoder(
#         gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
#         train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
#         # save_path=r"/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295",
#         # custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295/gida_dataset_log.pt",
#         # custom_subset_shuffle_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/shared-encoder_diameter-length-pre-norm-edges+CTOWN+GraphWaterSE+20250606_020235_114295/gida_dataset_log.pt"
#     )
# elif TASK == "shared-enc-inference":
#     shared_enc_PE_inference(
#         gida_yaml_path="gigantic_dataset/arguments/train/data.yaml",
#         train_yaml_path="gigantic_dataset/arguments/train/model.yaml",
#         save_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/fixed-shared-enc-default+CTOWN+GraphWaterSE+20250608_124003_296717",
#         custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/fixed-shared-enc-default+CTOWN+GraphWaterSE+20250608_124003_296717/gida_dataset_log.pt",
#         custom_subset_shuffle_pt_path=""
#         # "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/ndeg-dev-10k-pe-ltown+L-TOWN+gatres+20250525_152802_1094373/gida_dataset_log.pt"
#         # custom_subset_shuffle_pt_path=r"/scratch/p303753/GDS_OUTPUTS/experiments_logs/single-10k+ZJ+gatres+20250302_1646/gida_dataset_log.pt"
#     )
#
elif TASK == "finetune-shared-enc":
    finetune_shared_encoder(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
        # save_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/shared-enc-FINETUNE-GraphWaterSE+20250528_013728_900102_v4",
        # custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/shared-enc-FINETUNE-GraphWaterSE+20250528_013728_900102_v4/gida_dataset_log.pt",
    )
elif TASK == "train_dual_enc":
    dual_encoder_train(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
    )
elif TASK == "train_dual_pretrained_encoders":
    dual_encoder_train_pretrained_encoders(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
    )
elif TASK == "dual_enc_inference":
    dual_encoder_inference(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
        save_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/DualEncoderGFM-FT-cos-wLoss-10k-150epochs+ky8+DualEncoderGFM+20250919_020919_147775",
        custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/model_logs/29WDNs_experiments/DualEncoderGFM-FT-cos-wLoss-10k-150epochs+ky8+DualEncoderGFM+20250919_020919_147775/gida_dataset_log.pt",
        # custom_subset_shuffle_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/experiments_logs/FUNC_ONLY_GATRes+ky14+GATRes+20250616_060640_37682/gida_dataset_log.pt"
    )
elif TASK == "finetune-dual-enc":
    finetune_dual_encoder(
        gida_yaml_path=f"gigantic_dataset/arguments/train/{args.dconf}",
        train_yaml_path=f"gigantic_dataset/arguments/train/{args.mconf}",
        # save_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/shared-enc-FINETUNE-GraphWaterSE+20250528_013728_900102_v5",
        # custom_stats_tuple_pt_path="/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/gfm-wdn/gigantic_dataset/trained_models/model_logs/GFM-struct-bias-weightedLoss+6wdns+DualEncoderGFM+20250627_171620_856060/gida_dataset_log.pt",
    )
