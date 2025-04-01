import numpy as np
import os
import json

# CONSTANTS
SAMPLE_DTYPE = np.float32 # data type of samples
EVENT_DTYPE = np.int32 # data type of event timestamps & markers/codes

class TrialBrainData:
    """Abstract class for representing brain actvity recordings from multiple experimental trials.

    Attributes:
        trial_nr (int): number of trials.
        chan_nr (int): number of channels.
        trials_len (int or list[int]): length of trials.
        samp_freq (float): sampling frequency in Hz.
        samp_mat_list (list[np.ndarray]): list of 2D sample matrices (of shape chan_nr, trials_len).
        info_dict (dict): dictionary for storing meta data (e.g. channel labels).
    """

    def __init__(self):
        self.trial_nr = 0 # number of trials
        self.chan_nr = 0 # number of channels (variables, e.g. electrodes, ROIs)
        self.trials_len = None # length of the trials (in samples)
        self.samp_freq = 0.0 # sampling frequency (in Hz!!)
        self.samp_mat_list = [] # list of trial sample matris
        self.info_dict = dict() # dictionary of other infos (metadata)

    def __str__(self)->str:
        return f"Brain activity recording from {self.trial_nr} trials of length {self.trials_len}; {self.chan_nr} channels and a {self.samp_freq}Hz sampling frequency."

    def load(self, samp_mat_list:list[np.ndarray], samp_freq:float, info_dict:dict=None)->None:
        """Function that loads object from input parameters.

        Args:
            samp_mat_list (list[np.ndarray]): list of 2D sample matrices
            samp_freq (float): sampling frequency in Hz
            info_dict (dict, optional): dictionary for storing meta data. Defaults to None.

        Raises:
            ValueError: uneven number of channels in different trials
            ValueError: negative sampling frequency
        """
        self.trial_nr = len(samp_mat_list)
        self.chan_nr = samp_mat_list[0].shape[0]
        self.trials_len = [samp_mat_list[0].shape[1]]
        for t in range(1, self.trial_nr):
            chan_nr, trial_len = samp_mat_list[t].shape
            if chan_nr != self.chan_nr:
                raise ValueError(f"Number of channels across the trials must be constant. Expected {self.chan_nr} number of channels for trial {t}, got {chan_nr} instead.")
            self.trials_len.append(trial_len)
        if len(np.unique(self.trials_len)) == 1:
            trial_len = self.trials_len[0]
            self.trials_len = trial_len
        self.samp_mat_list = samp_mat_list
        if samp_freq <= 0.0:
            raise ValueError(f"Sampling frequency must be > 0! (value provided {samp_freq})")
        self.samp_freq = samp_freq
        if info_dict is not None: 
            self.info_dict = info_dict


    def load_from_files(self, info_json_path:str, samp_bin_path:str)->None:
        """Load brain activity recording data from two files.

        Args:
            info_json_path (str): path to json file containing information about the recording (i.e. number of channels, trials, trial lengths, channel names, etc.)
            samp_bin_path (str): path to binary file storing 3D array of shape (number of trials, number of channels, trial length) of 32-bit floating point values
        """
        with open(info_json_path, "r") as file:
            info_dict = json.load(file)
        
        self.trial_nr = info_dict["trial_nr"]
        self.chan_nr = info_dict["chan_nr"]
        self.trials_len = info_dict["trials_len"]
        self.samp_freq = info_dict["samp_freq_Hz"]
        self.info_dict["chan_name_list"] = info_dict["chan_name_list"]

        if type(self.trials_len) == list: # case of uneven trial lengths
            if len(self.trials_len) != self.trial_nr:
                raise ValueError("")
            max_trials_len = np.max(np.array(self.trials_len))
            trials_samp_mat = np.fromfile(samp_bin_path, dtype=SAMPLE_DTYPE).reshape(self.trial_nr, self.chan_nr, max_trials_len)
            for t in range(self.trial_nr):
                trial_len = self.trials_len[t]
                self.samp_mat_list.append(trials_samp_mat[t][:, :trial_len]) # remove zero padding
        
        elif type(self.trials_len) == int: # all trials have the same length
            trials_samp_mat = np.fromfile(samp_bin_path, dtype=SAMPLE_DTYPE).reshape(self.trial_nr, self.chan_nr, self.trials_len)
            for t in range(self.trial_nr):
                self.samp_mat_list.append(trials_samp_mat[t])

    def clear(self):
        """Function that clears all data.
        """
        self.trial_nr = 0
        self.chan_nr = 0
        self.trial_len = 0
        self.samp_freq = 0.0
        self.samp_mat_list = np.empty((self.trial_nr, self.chan_nr, self.trial_len), dtype=SAMPLE_DTYPE)
        self.info_dict.clear()
        
