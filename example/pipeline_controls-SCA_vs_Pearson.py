# STEP 1: Processing data
## 1. Import Python packages
import os
import numpy as np
import braindynamics_starprotocol as bd
import pandas as pd
import matplotlib.pyplot as plt
import json

## 2. Load the condition information for trials, separate trials (indices) based on the experimental conditions
with open("./PATHS.json", "r") as f:
    paths_dict = json.load(f)

INPUT_DIR = paths_dict["input_dir"]
OUTPUT_DIR = paths_dict["output_dir"]

info = pd.read_csv(os.path.join(INPUT_DIR, "Pipe02_trial_info.csv"))
control_trial_idx = info[info["condition"] == "Control"].index.values
patient_trial_idx = info[info["condition"] == "EtOH"].index.values

## 3. Load trials samples. Separate them based on experimental conditions
data = bd.TrialBrainData()
data.load_from_files(os.path.join(INPUT_DIR, "Pipe02_info.json"), os.path.join(INPUT_DIR, "Pipe02_samples.bin"))

control_trial_samps = [data.samp_mat_list[c] for c in control_trial_idx]
patient_trial_samps = [data.samp_mat_list[p] for p in patient_trial_idx]

## 4. Define correlogram for SCA & classical Pearson analysis
SCA_SCALE_SIZE = int(20*data.samp_freq) # scale size parameter for SCA
SCA_SHIFT_SIZE = int(20*data.samp_freq) # max shift size parameter for SCA
sca_corrgram = bd.CrossCorrelogram(SCA_SHIFT_SIZE, scale_size=SCA_SCALE_SIZE)

pearson_corrgram = bd.CrossCorrelogram(0) # Pearson's r at lag 0

## 5. Extract functional brain networks for the two groups (controls and patients); both analyses
### SCA
sca_control_networks = bd.NetworkData()
sca_control_networks.extract(control_trial_samps, sca_corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "sca_control_networks", "sca_control_networks.filelist"), os.path.join(OUTPUT_DIR, "control_lags", "control_lags.filelist")))
sca_control_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
#sca_control_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "sca_control_networks", "sca_control_networks.filelist"))

sca_patient_networks = bd.NetworkData()
sca_patient_networks.extract(patient_trial_samps, sca_corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "sca_patient_networks", "sca_patient_networks.filelist"), os.path.join(OUTPUT_DIR, "patient_lags", "patient_lags.filelist")))
sca_patient_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
#sca_patient_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "sca_patient_networks", "sca_patient_networks.filelist"))

### classical Pearson
pearson_control_networks = bd.NetworkData()
pearson_control_networks.extract(control_trial_samps, pearson_corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "pearson_control_networks", "pearson_control_networks.filelist"), os.path.join(OUTPUT_DIR, "control_lags", "control_lags.filelist")))
pearson_control_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
#pearson_control_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "pearson_control_networks","pearson_control_networks.filelist"))

pearson_patient_networks = bd.NetworkData()
pearson_patient_networks.extract(patient_trial_samps, pearson_corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "pearson_patient_networks", "pearson_patient_networks.filelist"), os.path.join(OUTPUT_DIR, "patient_lags", "patient_lags.filelist")))
pearson_patient_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
#pearson_patient_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "pearson_patient_networks/","pearson_patient_networks.filelist"))


# STEP 2: Calculate distributions
## 1. Compute and export edge weight distributions of the two groups
### SCA
sca_control_ewd = sca_control_networks.compute_ewd()
sca_patient_ewd = sca_patient_networks.compute_ewd()

np.savetxt(os.path.join(OUTPUT_DIR, "sca_control_networks-EWD.txt"), sca_control_ewd, fmt="%.8f")
np.savetxt(os.path.join(OUTPUT_DIR, "sca_patient_networks-EWD.txt"), sca_patient_ewd, fmt="%.8f")

### classical Pearson
pearson_control_ewd = pearson_control_networks.compute_ewd()
pearson_patient_ewd = pearson_patient_networks.compute_ewd()

np.savetxt(os.path.join(OUTPUT_DIR, "pearson_control_networks-EWD.txt"), pearson_control_ewd, fmt="%.8f")
np.savetxt(os.path.join(OUTPUT_DIR, "pearson_patient_networks-EWD.txt"), pearson_patient_ewd, fmt="%.8f")

# STEP 3: Visualizing distributions
## 1. Sort distributions
### NOTE: use same sorting for all groups & both analyses
_, sca_ewd_sorting = bd.sort_distribution(sca_control_ewd)
np.savetxt(os.path.join(OUTPUT_DIR, "sca_control_networks-EWD-sorting_ltof.txt"), sca_ewd_sorting, fmt="%d")

ewd_cmin, ewd_cmax = np.min([np.min(sca_control_ewd), np.min(sca_patient_ewd), np.min(pearson_control_ewd), np.min(pearson_patient_ewd)]), np.max([np.max(sca_control_ewd), np.max(sca_patient_ewd), np.max(pearson_control_ewd), np.max(pearson_patient_ewd)])

# _, pearson_ewd_sorting = bd.sort_distribution(pearson_control_ewd)
# np.savetxt(os.path.join(OUTPUT_DIR, "pearson_control_networks-EWD-sorting_ltof.txt"), pearson_ewd_sorting, fmt="%d")

## 2. Plot distributions
plt.rcParams["font.family"] = "arial"
plt.rcParams["font.size"] = 16

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,10), gridspec_kw={"width_ratios":[0.2,0.2,0.2,0.2]})
plt.subplots_adjust(wspace=0.6)

bd.plot_distribution(sca_control_ewd, sorting=sca_ewd_sorting, fig=fig, ax=ax[0], cmin=ewd_cmin, cmax=ewd_cmax, clabel="probability density", title="SCA", xlabel="edge weight", ylabel="Control", vmin=-1.0, vmax=1.0)
bd.plot_distribution(pearson_control_ewd, sorting=sca_ewd_sorting, fig=fig, ax=ax[1], cmin=ewd_cmin, cmax=ewd_cmax, clabel="probability density", show_cbar=False, title="Pearson", vmin=-1.0, vmax=1.0)

bd.plot_distribution(sca_patient_ewd, sorting=sca_ewd_sorting, fig=fig, ax=ax[2], cmin=ewd_cmin, cmax=ewd_cmax, clabel="probability density", ylabel="EtOH", title="SCA", vmin=-1.0, vmax=1.0)
bd.plot_distribution(pearson_patient_ewd, sorting=sca_ewd_sorting, fig=fig, ax=ax[3], cmin=ewd_cmin, cmax=ewd_cmax, clabel="probability density", show_cbar=True, title="Pearson", vmin=-1.0, vmax=1.0)

plt.savefig(os.path.join(OUTPUT_DIR, "EWD-sca_vs_pearson.pdf"), bbox_inches="tight", dpi=300)
