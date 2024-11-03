# STEP 1: Processing data
## 1. Import Python packages
import os
import numpy as np
import braindynamics_starprotocol as bd
import pandas as pd
import matplotlib.pyplot as plt
## 2. Load the condition information for trials, separate trials (indices) based on the experimental conditions
info = pd.read_csv("Pipe02_trial_info.csv")
control_trial_idx = info[info["condition"] == "Control"].index.values
patient_trial_idx = info[info["condition"] == "EtOH"].index.values
## 3. Load trials samples. Separate them based on experimental conditions
data = bd.TrialBrainData()
data.load_from_files("Pipe02_info.json", "Pipe02_samples.bin")

control_trial_samps = [data.samp_mat_list[c] for c in control_trial_idx]
patient_trial_samps = [data.samp_mat_list[p] for p in patient_trial_idx]
## 4. Define scaled cross-correlogram with the following two parameters
SCA_SCALE_SIZE = int(20*data.samp_freq) # scale size parameter for SCA
SCA_SHIFT_SIZE = int(20*data.samp_freq) # max shift size parameter for SCA
corrgram = bd.ScaledCrossCorrelogram(SCA_SCALE_SIZE, SCA_SHIFT_SIZE)
## 5. Extract functional brain networks for the two groups (controls and patients)
control_networks = bd.NetworkData()
control_networks.extract(control_trial_samps, corrgram, export_to_filelists=(os.path.join("control_networks", "control_networks.filelist"), os.path.join("control_lags", "control_lags.filelist")))
control_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
# control_networks.load_from_filelist("control_networks/control_networks.filelist")

patient_networks = bd.NetworkData()
patient_networks.extract(patient_trial_samps, corrgram, export_to_filelists=(os.path.join("patient_networks", "patient_networks.filelist"), os.path.join("patient_lags", "patient_lags.filelist")))
patient_networks.info_dict["node_label_list"] = data.info_dict["chan_name_list"]
# patient_networks.load_from_filelist("patient_networks/patient_networks.filelist")

# STEP 2: Calculate distributions
## 1. Compute and export edge weight distributions of the two groups
control_ewd = control_networks.compute_ewd()
patient_ewd = patient_networks.compute_ewd()

os.makedirs("results", exist_ok=True)
np.savetxt(os.path.join("results", "control_networks-EWD.txt"), control_ewd, fmt="%.8f")
np.savetxt(os.path.join("results", "patient_networks-EWD.txt"), patient_ewd, fmt="%.8f")
## 2. Compute and export node distance distributions (NDDs) for the two groups
### a. Determine first the maximal absolute edge weight and the minimum and maximum node distance for the two groups
max_weight = max(control_networks.get_maxabs_weights(), patient_networks.get_maxabs_weights())

control_min_distance, control_max_distance = control_networks.get_minmax_distances(norm_factor=max_weight)
patient_min_distance, patient_max_distance = patient_networks.get_minmax_distances(norm_factor=max_weight)
min_distance = min(control_min_distance, patient_min_distance)
max_distance = max(control_max_distance, patient_max_distance)
### b. Compute and export node distance distributions (NDDs) for the two groups
control_ndd = control_networks.compute_ndd(bin_nr=15, ndd_vmin=min_distance, ndd_vmax=max_distance, norm_factor=max_weight)
patient_ndd = patient_networks.compute_ndd(bin_nr=15, ndd_vmin=min_distance, ndd_vmax=max_distance, norm_factor=max_weight)

np.savetxt(os.path.join("results", "control_networks-NDD.txt"), control_ndd, fmt="%.8f")
np.savetxt(os.path.join("results", "patient_networks-NDD.txt"), patient_ndd, fmt="%.8f")
## 3. Compute and export node edge weight distributions (N-EWD) for the two groups
control_newd = control_networks.compute_newd()
patient_newd = patient_networks.compute_newd()

np.savetxt(os.path.join("results", "control_networks-NEWD.txt"), control_newd, fmt="%.8f")
np.savetxt(os.path.join("results", "patient_networks-NEWD.txt"), patient_newd, fmt="%.8f")


# STEP 3: Calculate statistics
## 1. Compare control and patient group NDD by calculating Cliff's delta statistic from the distribution samples
control_nds = control_networks.compute_nd_for_Cliffs_delta(norm_factor=max_weight)
patient_nds = patient_networks.compute_nd_for_Cliffs_delta(norm_factor=max_weight)
nds_Cliffs_deltas = bd.compare_nds_Cliffs_delta(patient_nds, control_nds)

np.savetxt(os.path.join("results", "patient_control_networks-NDD-Cliffs_deltas.txt"), nds_Cliffs_deltas, fmt="%.8f")
## 2. Compare control and patient group N-EWD by calculating separately Cliff's delta statistic for the negative and positive parts of the bimodal distribution
control_news = control_networks.compute_new_for_Cliffs_delta()
patient_news = patient_networks.compute_new_for_Cliffs_delta()
news_Cliffs_deltas = bd.compare_news_Cliffs_delta(patient_news, control_news)

np.savetxt(os.path.join("results", "patient_control_networks-NEWD-Cliffs_deltas.txt"), np.array(news_Cliffs_deltas).T, fmt="%.8f")

# STEP 4: Visualize results
## 1. Set plot parameters
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 16
## 2. Visualize edge weight distributions (EWDs)
### a. Sort and export EWD rows (edges) from last-to-first columns, for the control group
_, ewd_sorting = bd.sort_distribution(control_ewd) # we want second sorting indices of rows, from last-to-first columns for EWD <=> backbone on top!
np.savetxt(os.path.join("results", "control_networks-EWD-sorting_ltof.txt"), ewd_sorting, fmt="%d")
### b. Determine minimum and maximum EWD value (probability density) for the two groups
ewd_cmin, ewd_cmax = min(np.min(control_ewd), np.min(patient_ewd)), max(np.max(control_ewd), np.max(patient_ewd))
### c. Plot the two distributions next to each other
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 10), gridspec_kw={'width_ratios': [0.2, 0.2]})
plt.subplots_adjust(wspace=0.6)

bd.plot_distribution(control_ewd, sorting=ewd_sorting, fig=fig, ax=ax[0], cmin=ewd_cmin, cmax=ewd_cmax, show_cbar=True, clabel="probability density", title="Control", xlabel="edge weight", ylabel=f"rat {len(control_ewd)} edges", vmin=-1.0, vmax=1.0)
bd.plot_distribution(patient_ewd, sorting=ewd_sorting, fig=fig, ax=ax[1], cmin=ewd_cmin, cmax=ewd_cmax, clabel="probability density", title="EtOH", vmin=-1.0, vmax=1.0)
plt.savefig(os.path.join("results", "patient_control_networks-EWD.png"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join("results", "patient_control_networks-EWD.pdf"), bbox_inches='tight', dpi=300)
## 3. Visualize node distance distributions (NDDs)
### a. Sort and export NDD rows (nodes) from first-to-last columns, for the control group
ndd_sorting, _ = bd.sort_distribution(control_ndd) # sorting from first-to-last columns for NDD <=> more central nodes on the top!
np.savetxt(os.path.join("results", "control_networks-NDD-sorting_ftol.txt"), ndd_sorting, fmt="%d")
### b. Determine minimum and maximum NDD value (probability density) for the two groups
ndd_cmin, ndd_cmax = min(np.min(control_ndd), np.min(patient_ndd)), max(np.max(control_ndd), np.max(patient_ndd))
### c. Plot the two distributions and their comparison using Cliff's delta next to each other
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(7, 10), gridspec_kw={'width_ratios': [0.2, 0.05, 0.2, 0.03, 0.03]})
plt.subplots_adjust(wspace=0.6)
ax[1].axis('off')

bd.plot_distribution(control_ndd, sorting=ndd_sorting, fig=fig, ax=ax[0], title="Control", xlabel="node distance", ylabel=f"rat {len(control_ndd)} nodes", show_cbar=True, cmin=ndd_cmin, cmax=ndd_cmax, clabel="probability density", vmin=min_distance, vmax=max_distance, yticks=control_networks.info_dict["node_label_list"], mirror_yticks=True)

bd.plot_distribution(patient_ndd, sorting=ndd_sorting, fig=fig, ax=ax[2], title="EtOH",  cmin=ndd_cmin, cmax=ndd_cmax, vmin=min_distance, vmax=max_distance)

bd.plot_Cliffs_delta(np.expand_dims(nds_Cliffs_deltas, axis=1), sorting=ndd_sorting, fig=fig, axes=[ax[3], ax[4]], clabel=r'$\delta$', show_cbar=True, titles=[r'$\delta$', r'$|\delta|$'])

plt.savefig(os.path.join("results", "control_patient_networks-NDD.png"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join("results", "control_patient_networks-NDD.pdf"), bbox_inches='tight', dpi=300)
## 4. Visualize node edge weight distributions (N-EWDs)
### a. Sort and export N-EWD rows (edges) from last-to-first columns, for the control group
_, newd_sorting = bd.sort_distribution(control_newd) # sorting from first-to-last columns for NDD <=> more central nodes on the top!
np.savetxt(os.path.join("results", "control_networks-NEWD-sorting_ltof.txt"), newd_sorting, fmt="%d")
### b. Determine minimum and maximum N-EWD value (probability density) for the two groups
newd_cmin, newd_cmax = min(np.min(control_newd), np.min(patient_newd)), max(np.max(control_newd), np.max(patient_newd))
### c. Plot the two distributions and their comparison using Cliff's delta next to each other
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 10), gridspec_kw={'width_ratios': [0.2, 0.05, 0.2, 0.06, 0.06]})
plt.subplots_adjust(wspace=0.6)
ax[1].axis('off')

bd.plot_distribution(control_newd, sorting=newd_sorting, fig=fig, ax=ax[0], title="Control", xlabel="edge weight", ylabel=f"rat {len(control_ndd)} nodes", show_cbar=True, cmin=newd_cmin, cmax=newd_cmax, clabel="probability density", yticks=control_networks.info_dict["node_label_list"], mirror_yticks=True, vmin=-1.0, vmax=1.0)

bd.plot_distribution(patient_newd, sorting=newd_sorting, fig=fig, ax=ax[2], title="EtOH",  cmin=newd_cmin, cmax=newd_cmax, vmin=-1.0, vmax=1.0)

bd.plot_Cliffs_delta(np.column_stack((news_Cliffs_deltas[0], news_Cliffs_deltas[1])), sorting=newd_sorting, fig=fig, axes=[ax[3], ax[4]], clabel=r'$\delta$', show_cbar=True, titles=[r'$\delta_n \delta_p$', r'$|\delta_n| |\delta_p|$'])

plt.savefig(os.path.join("results", "control_patient_networks-NEWD.png"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join("results", "control_patient_networks-NEWD.pdf"), bbox_inches='tight', dpi=300)