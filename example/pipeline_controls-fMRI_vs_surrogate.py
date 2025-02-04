# STEP 1: Processing data
## 1. Import Python packages
import os
import json
import numpy as np
import braindynamics_starprotocol as bd
import pandas as pd
import matplotlib.pyplot as plt

def FFTPhaseScramble(vs, fAcceptedImaginary = 1e-6):
    #shuffling the phase spectrum keeping the simmetries of the real signal
    vsp = np.fft.fft(vs)
    nFFTSize = len(vsp)     #should be the same as the input signal length
    
    vphase =    np.angle(vsp)
    vamp =      np.abs(vsp)  

    #leave the first frequency bin alone - this is the DC component
    #shuffle the first half of remaining bins    
    nPhasesToShuffle =  (nFFTSize - 1) // 2 #math.floor((nFFTSize - 1) / 2)       #floor the phases to shuffle
    vIDxToShuffle =     np.arange(1, nPhasesToShuffle)
    vIDxShuffled =      np.arange(1, nPhasesToShuffle)
    np.random.shuffle(vIDxShuffled)

    vphasescrambled = vphase.copy()                                             #this copy ensures that anything that is left outside of shuffling is copyied (eg dc and exactly half of the sampling frequency)
    vphasescrambled[vIDxToShuffle] = vphase[vIDxShuffled]                       #shuffle first half of the phases
    vphasescrambled[nFFTSize - vIDxToShuffle] = - vphasescrambled[vIDxToShuffle]#antisymetric copy (mirrored around center, except DC) and with signs flipped
    vspscrambled = vamp * np.exp(1j*vphasescrambled)                            #going back into real,imaginary representation (element-wise multiplication)
    vscrambled =  np.fft.ifft(vspscrambled)
    
    if max(abs(vscrambled.imag)) > fAcceptedImaginary:
        return []
    else: 
       return vscrambled.real 

def generate_surrogate_signals(signals:np.ndarray):
    
    if len(signals.shape) != 3:
        raise ValueError("Input array expected to be 3D, with the dimensions being (trial/subject, ROI/channel, sample).")

    trial_nr, chan_nr, samp_nr = signals.shape
    
    new_signals = np.zeros_like(signals)
    for t in range(trial_nr):
        for c in range(chan_nr):
            new_signal =  FFTPhaseScramble(signals[t][c], fAcceptedImaginary=1.0e-5)
            if len(new_signal) != 0:
                new_signals[t][c] = new_signal

    return new_signals

def plotter(shifts:np.ndarray, maxabs:np.ndarray, hist:np.ndarray, bins:np.ndarray, output_path:str, color:str="tab:blue", title:str="")->None:

    if shifts.shape != maxabs.shape:
        raise ValueError("Shapes of input arrays do not match")    
    
    if hist.shape != bins.shape:
        raise ValueError("Shapes of histogram bins and values do not match")

    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios":[3,1]})
    ax[0].scatter(shifts, maxabs, s=1, color=color)
    ax[0].axhline(0, color="grey", alpha=0.25, linewidth=0.5)
    ax[0].set_ylim([-1.0, 1.0])
    ax[0].set_title(title)
    ax[0].set_xlabel("Time offset [s]")
    ax[0].set_ylabel("MaxAbs(SCA20s)")
    
    ax[1].plot(hist, bins, color=color)
    ax[1].axhline(0, color="grey", alpha=0.25, linewidth=0.5)
    ax[1].set_xlim([0, 1.25*np.max(hist)])
    ax[1].set_ylim([-1.0, 1.0])
    ax[1].xaxis.tick_top()
    ax[1].set_yticks([])
    ax[1].set_xlabel("Counts")
    ax[1].xaxis.set_label_position("top")

    plt.subplots_adjust(wspace=0.02)
    #plt.show()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
 
def main():
    
## 2. Load the condition information for trials, separate trials (indices) based on the experimental conditions
    with open("PATHS.json", "r") as file:
        paths_dict = json.load(file)

    INPUT_DIR = paths_dict["input_dir"]
    OUTPUT_DIR = paths_dict["output_dir"]

    info = pd.read_csv(os.path.join(INPUT_DIR, "Pipe02_trial_info.csv"))
    trial_idx = info[info["condition"] == "Control"].index.values

    data = bd.TrialBrainData()
    data.load_from_files(os.path.join(INPUT_DIR, "Pipe02_info.json"),
                         os.path.join(INPUT_DIR, "Pipe02_samples.bin"))

## 3. Load trials samples & generate surrograte data from them
    trial_samps = [data.samp_mat_list[t] for t in trial_idx] 
    surr_trial_samps = generate_surrogate_signals(np.array(trial_samps))

## 4. Define correlogram for SCA analysis
    SCA_SCALE_SIZE = int(20*data.samp_freq)
    SCA_SHIFT_SIZE = int(20*data.samp_freq)
    corrgram = bd.CrossCorrelogram(SCA_SHIFT_SIZE, scale_size=SCA_SCALE_SIZE)
    
## 5. Extract functional brain networks for the two groups (original and surrogate)
    networks = bd.NetworkData()
    networks.extract(trial_samps, corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "networks", "networks.filelist"), os.path.join(OUTPUT_DIR, "lags", "lags.filelist")))
    networks.info_dict["node_name_list"] = data.info_dict["chan_name_list"]
    #networks.load_from_filelist(os.path.join(OUTPUT_DIR, "networks", "networks.filelist"))
    #networks.load_from_filelist(os.path.join(OUTPUT_DIR, "lags", "lags.filelist"), load_lags=True)

    corrgram.clear()
    surr_networks = bd.NetworkData()
    surr_networks.extract(surr_trial_samps, corrgram, export_to_filelists=(os.path.join(OUTPUT_DIR, "surr_networks", "surr_networks.filelist"), os.path.join(OUTPUT_DIR, "surr_lags", "surr_lags.filelist")))
    surr_networks.info_dict["node_name_list"] = data.info_dict["chan_name_list"]
    #surr_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "surr_networks", "surr_networks.filelist"))
    #surr_networks.load_from_filelist(os.path.join(OUTPUT_DIR, "surr_lags", "surr_lags.filelist"), load_lags=True)

# STEP 2: Calculate distributions
## 1. Compute and export edge weight distributions of the two groups
    ewd = networks.compute_ewd()
    np.savetxt(os.path.join(OUTPUT_DIR, "networks", "ewd.txt"), ewd, fmt="%.8f")
    #ewd = np.loadtxt(os.path.join(OUTPUT_DIR, "networks", "ewd.txt"))

    surr_ewd = surr_networks.compute_ewd()
    np.savetxt(os.path.join(OUTPUT_DIR, "surr_networks", "surr_ewd.txt"), surr_ewd, fmt="%.8f")
    #surr_ewd = np.loadtxt(os.path.join(OUTPUT_DIR, "surr_networks", "surr_ewd.txt"))

## 2. Create centered bins of EWD (hardcoded, 31 bins from -1 to 1)
    BINS = np.linspace(-1.0, 1.0, 31)
    BINS = (BINS[1:] + BINS[:-1])/2

# STEP 3: Visualize
## Reproduce plots from Figure 5 (STARProtocols article)

### on left panel, show the edge weight FOR ONE TRIAL ONLY
### on right panel, show the EWD FOR ALL TRIALS & EDGES COMBINED (rotated & not normalized!!)
    plotter(networks.laglist_arr[0, :, -1], networks.edgelist_arr[0, :, -1],
            np.sum(np.round(ewd*networks.trial_nr, 1), axis=0), # sum for all edges, convert prob. density back to counts
            BINS,
            output_path=os.path.join(OUTPUT_DIR, "fmri_plot.pdf"),
            title="fMRI")

### same as before just for the surrogate data
    plotter(surr_networks.laglist_arr[0, :, -1], surr_networks.edgelist_arr[0, :, -1],
            np.sum(np.round(surr_ewd*surr_networks.trial_nr, 1), axis=0),
            BINS,
            output_path=os.path.join(OUTPUT_DIR, "surr_plot.pdf"),
            color="tab:red",
            title="Surrogate FFT phase shuffle")

    return 0

if __name__ == "__main__":
    main()
