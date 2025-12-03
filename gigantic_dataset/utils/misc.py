#
# Created on Tue Feb 06 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Collectiion of auxiliary functions
# ------------------------------
#
import warnings
from wntr.network import WaterNetworkModel
from wntr.epanet.util import FlowUnits  # , HydParam
import os
from glob import glob
# import numpy as np

# from gigantic_dataset.utils.configs import SenConfig, SenPlotConfig
# from gigantic_dataset.utils.oatvis import analyze_oat_sensitivity, plot_oat_sensisvity


def get_inp_paths(
    verbose: bool = False,
    filtered_wdn_names: list = [],
    shared_drive_path=r"G:\.shortcut-targets-by-id\1CxQ2YjkO9zW0p1Gb0ncXk-oisTYm_hJI\Paper Dataset - WDSA\input_files",
):
    inp_paths = glob(os.path.join(shared_drive_path, "*.inp"))
    if verbose:
        print(f"inp_paths = {inp_paths} - len = {len(inp_paths)}")
    filtered_inp_paths = []
    for inp_path in inp_paths:
        is_filtered = False
        if len(filtered_wdn_names) > 0:
            for exist_name in filtered_wdn_names:
                if exist_name in os.path.basename(inp_path):
                    is_filtered = True
                    if verbose:
                        print(f"{inp_path} is filterd out")
                    break
        if not is_filtered:
            filtered_inp_paths.append(inp_path)

    return filtered_inp_paths


def check_wdns(folder_path: str, verbose: bool = True) -> tuple[list[str], list[str]]:
    inp_paths = [
        os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name[-4:].lower() == ".inp"
    ]
    failed_list = []
    success_list = []
    for inp in inp_paths:
        wdn_name = os.path.basename(inp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                WaterNetworkModel(inp)
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


def get_flow_units(wn: WaterNetworkModel) -> FlowUnits:
    if isinstance(wn.options.hydraulic.inpfile_units, str):
        units = wn.options.hydraulic.inpfile_units.upper()
        flow_units = FlowUnits[units]
    else:
        flow_units = FlowUnits.GPM
    return flow_units


# def analyze_sensivity_and_plot():
#     #########################analyze_oat_sensitivity#############################
#     # unncessary
#     # inp_paths=  misc.get_inp_paths(filtered_wdn_names=[])
#     # misc.check_wdns(inp_paths)

#     config = SenConfig().parse_args()
#     config.inp_paths = [
#         r"gigantic-dataset\inputs\ctown.inp",
#     ]
#     analyze_oat_sensitivity(config=config)

#     #################################PLOT SENSITIVITY####################################
#     config = SenPlotConfig()
#     config.parse_args([])
#     config.trial_aggr = "max"
#     config.wdn_aggr = "max"
#     config.do_indinvidual_plot = True
#     # config.zarr_paths = [r'gigantic-dataset\outputs\sensitivity_Anytown_pressure_20240205_1524.zip']
#     config.zarr_paths = [
#         # r'gigantic-dataset\outputs\sensitivity_ctown_pressure_20240213_1519.zip',
#         r"G:\Other computers\My Laptop\PhD\Codebase\gigantic-dataset\gigantic-dataset\outputs\sensitivity_GEN-09 Oosterbeek_pressure_20240315_1243.zarr"
#     ]

#     # def cv(baseline, factors):
#     #     """
#     #     Low CV (CV < 0.1):If the CV is low, it indicates that the variability in the dataset is relatively small compared to the mean.
#     #     This suggests that the values in the dataset are close to the mean, and there is relatively little dispersion.

#     #     Moderate CV (0.1 < CV < 1): A moderate CV suggests a moderate level of variability relative to the mean.
#     #     The dataset has some dispersion, but it is not extremely spread out. This is a common range for many datasets.

#     #     High CV (CV > 1): A high CV indicates significant variability compared to the mean.
#     #     The dataset has a large spread of values, and there may be considerable differences between individual data points and the mean.

#     #     Very High CV (CV > 2): If the CV is very high, it suggests extreme variability compared to the mean.
#     #     The dataset may contain outliers or have a wide range of values.
#     #     """

#     #     x = baseline.reshape([-1,1])
#     #     y = factors.reshape([-1,1])
#     #     xy = np.concatenate([x,y],axis=0)
#     #     return np.std(xy,axis=0) / (np.mean(xy,axis=0) + 1e-6)

#     # metric_fn = cv#lambda x,y : np.std(np.concatenate([x,y],axis=0),axis=0) / np.mean(np.concatenate([x,y],axis=0),axis=0) #np.mean((x-y)**2,axis=0) #np.mean(np.abs(np.var(x,axis=0) - np.var(y,axis=0)),axis=0) #np.mean(np.abs(x-y),axis=0)
#     # plot_oat_sensisvity(metric_fn, config, annotation="""Low CV (CV < 0.1): variability is relatively small compared to the mean.
#     #     <br>Moderate CV (0.1 < CV < 1): This is a common range for many datasets.
#     #     <br>High CV (CV > 1): The dataset has a large spread of values.
#     #     <br>Very High CV (CV > 2): The dataset may contain outliers or have a wide range of values.""")

#     metric_fn = lambda baseline, factor: np.mean(np.abs(factor - baseline), axis=0)
#     plot_oat_sensisvity(metric_fn, config, annotation="MAE(baseline, factor)")
