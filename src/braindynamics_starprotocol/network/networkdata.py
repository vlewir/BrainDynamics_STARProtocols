import numpy as np
import os
from ..sca.crosscorrelogram import CrossCorrelogram
import igraph as ig


class NetworkData:
    """

    Attributes: 
        trial_nr (int): Number of experimental trials (2D time series) from which networks will be extracted. 
        edge_nr (int): Number of edges in the functional brain network. 
        edgelist_arr (np.ndarray): Array of edge lists (in format <source> <target> <weight>) for each trial.
        laglist_arr (np.ndarray): Array of lag lists (in format <source> <target> <lag>) for each trial.
        node_nr (int): Number of nodes in the functional brain network.
        info_dict (dict):  Dictionary storing metadata (e.g. node labels).
    """
    def __init__(self):
        self.trial_nr = 0 # total number of trials
        self.edge_nr = 0 # total number of edges in functional brain network
        self.edgelist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of edgelists (2D arrays, <source> <target> <weight (correlation)> format)
        self.laglist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of edgelists of lags/time-offsets from the MaxAbs of scaled cross-correlogram (2D arrays, <source> <target> <lag> format)
        self.node_nr = 0 # total number of nodes in functional brain network
        self.info_dict = {} # dictionary for other information (meta-data)

    def clear(self):
        """Function that clears NetworkData object.
        """
        self.trial_nr = 0
        self.edge_nr = 0
        self.edgelist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32)
        self.laglist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32)
        self.node_nr = 0
        self.info_dict.clear()
    
    def load_from_filelist(self, filelist_path:str, load_lags:bool=False)->None:
        """Function that loads edgelists from filelist.

        Args:
            filelist_path (str): Path to filelist (enumeration of edge list files)

        Raises:
            ValueError: uneven edge list lengths
        """
        
        filelist_root = os.path.abspath(os.path.dirname(filelist_path)) # root directory of filelist
        filelist = np.loadtxt(filelist_path, dtype=str) # filelist (list of edge list filenames)
        self.trial_nr = len(filelist) # number of trials (= of edge list files)
        self.edge_nr = np.loadtxt(os.path.join(filelist_root, filelist[0])).shape[0] # number of edges in an edge list
        
        tmp_edgelist = np.loadtxt(os.path.join(filelist_root, filelist[0]))
        self.node_nr = int(max(np.max(tmp_edgelist[:, 0]), np.max(tmp_edgelist[:, 1]))) + 1
        if not load_lags:
            # loading edge lists
            self.edgelist_arr = np.zeros((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of edge lists
            for i, filename in enumerate(filelist):
                edgelist = np.loadtxt(os.path.join(filelist_root, filename))
                if edgelist.shape[0] != self.edge_nr:
                    raise ValueError(f"Inconsistent edge list length in file '{filename}': {edgelist.shape[0]} (expected value: {self.edge_nr})")
                self.edgelist_arr[i] = edgelist
        else:
            # loading lag lists instead
            self.laglist_arr = np.zeros((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of lag lists
            for i, filename in enumerate(filelist):
                laglist = np.loadtxt(os.path.join(filelist_root, filename))
                if laglist.shape[0] != self.edge_nr:
                    raise ValueError(f"Inconsistent lag list length in file '{filename}': {laglist.shape[0]} (expected value: {self.edge_nr})")
                self.laglist_arr[i] = laglist
    
    def write_to_filelist(self, filelist_path:str, save_lags:bool=False)->None:
        """Function that exports edgelists into separate files and generates the filelist.

        Args:
            filelist_path (str): path to filelist
            save_lags (bool, optional): save edge weights (maxabs) or lags (time offsets from correlogram). Defaults to False.
        """
        if not save_lags:
            list_arr = self.edgelist_arr
        else:
            list_arr = self.laglist_arr
        filelist_root = os.path.abspath(os.path.dirname(filelist_path))
        os.makedirs(filelist_root, exist_ok=True)
        filelist = np.array([f"edge_list-{t}.txt" for t in range(1, self.trial_nr + 1)], dtype=str)
        for i, filename in enumerate(filelist):
            with open(os.path.join(filelist_root, filename), "w") as file:
                sources, targets, weights = list_arr[i].T
                sources = sources.astype(int)
                targets = targets.astype(int)
                for source, target, weight in zip(sources, targets, weights):
                    file.write(f"{source} {target} {weight}\n")
        np.savetxt(filelist_path, filelist, fmt="%s")

    def extract(self, samp_mat_list:list, corrgram:CrossCorrelogram, use_fisher:bool=True, cc_method:str="C", export_to_filelists:tuple=None)->None:
        """Function that extracts functional brain networks from experimental trial samples. Functional connectivity (edge weights) is defined as cross-correlation (either classical Pearson or SCA developed by Nikolic et al)

        Args:
            samp_mat_list (list): list of trial sample matrices of shape (number of trials, number of channels/nodes, length of a trial)
            corrgram (CrossCorrelogram): cross-correlogram object with pre-defined parameters
            use_fisher (bool, optional): use Fisher transform or not in scaled correlation computation (see Nikolic et al. 2012). Defaults to True.
            cc_method (str, optional): choose implementation (python or cython). Defaults to C.
        """
        self.clear()
        self.trial_nr = len(samp_mat_list)
        self.node_nr = samp_mat_list[0].shape[0]    
        self.edge_nr = int(self.node_nr*(self.node_nr - 1)/2)
        self.edgelist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of edgelists (weights)
        self.laglist_arr = np.empty((self.trial_nr, self.edge_nr, 3), dtype=np.float32) # array of lags (time offsets between channels)
        if export_to_filelists is None: # not exporting anything, just computing
            for t in range(self.trial_nr):
                e = 0
                for i in range(self.node_nr):
                    for j in range(i + 1, self.node_nr): # NOTE: scaled correlation is symmetric
                        corrgram.compute(samp_mat_list[t][i], samp_mat_list[t][j], use_fisher=use_fisher, cc_method=cc_method)
                        lag, maxabs = corrgram.get_maxabs()
                        self.edgelist_arr[t][e] = [i, j, maxabs]
                        self.laglist_arr[t][e] = [i, j, lag]
                        e += 1
        else: # exporting in real-time edge/laglists after they are computed for one trial
            if len(export_to_filelists) != 2:
                    raise ValueError("Please provide two separate filelist paths for edge weights (MaxAbs values) and lags (MaxAbs time offsets)")
            edge_filelist_path, lag_filelist_path = export_to_filelists
            edge_filelist_root, lag_filelist_root = os.path.dirname(edge_filelist_path), os.path.dirname(lag_filelist_path)
            os.makedirs(edge_filelist_root, exist_ok=True)
            os.makedirs(lag_filelist_root, exist_ok=True)
            filelist = np.array([f"edge_list-{t}.txt" for t in range(1, self.trial_nr + 1)], dtype=str)
            for t in range(self.trial_nr):
                e = 0
                with open(os.path.join(edge_filelist_root, f"edge_list-{t + 1}.txt"), "w") as edge_file:
                    with open(os.path.join(lag_filelist_root, f"edge_list-{t + 1}.txt"), "w") as lag_file:
                        for i in range(self.node_nr):
                            for j in range(i + 1, self.node_nr): # NOTE: scaled correlation is symmetric
                                corrgram.compute(samp_mat_list[t][i], samp_mat_list[t][j], use_fisher=use_fisher, cc_method=cc_method)
                                lag, maxabs = corrgram.get_maxabs()
                                self.edgelist_arr[t][e] = [i, j, maxabs]
                                self.laglist_arr[t][e] = [i, j, lag]
                                e += 1
                                edge_file.write(f"{i} {j} {maxabs}\n")
                                lag_file.write(f"{i} {j} {lag}\n")

            filelist = np.array([f"edge_list-{t}.txt" for t in range(1, self.trial_nr + 1)], dtype=str) 
            np.savetxt(edge_filelist_path, filelist, fmt="%s")
            np.savetxt(lag_filelist_path, filelist, fmt="%s")

    def get_maxabs_weights(self)->float:
        """Function that finds the maximum absolute edge weight across the trials. It can be used to normalize all edge weights.

        Returns:
            float: value of maximum absolute edge weight (if it's above 1.0, return 1.0 instead)
        """
        global_max = np.max(np.abs(self.edgelist_arr[:, :, -1]))
        if global_max < 1.0:
            return 1.0
        return global_max

    def get_minmax_distances(self, norm_factor:float=1.0)->tuple:
        """Function that finds the minimum and maximum distance (shortest path length between nodes).

        Args:
            norm_factor (float, optional): normalizion factor for edge weights (useful when there's a correlation value above 1.0 due to numerical error/interpolation in SCA). Defaults to 1.0.

        Returns:
            tuple: minimum and maximum distance value
        """
        global_min = np.inf
        global_max = 0.0
        for t in range(self.trial_nr):
            sources, targets, weights = self.edgelist_arr[t].T
            sources = sources.astype(int)
            targets = targets.astype(int)
            lengths = -np.log(np.abs(weights/norm_factor)) # link lengths: l_{ij} = -log(|w_{ij}|); normalize weights above +-1.0 to avoid negative lengths!
            graph = ig.Graph(n=self.node_nr, directed=True)
            graph.add_edges(list(zip(sources, targets)))
            graph.delete_vertices(graph.vs.select(_degree=0))
            graph.vs["name"] = np.arange(1, self.node_nr)
            graph.es["weight"] = lengths
            node_distances = graph.shortest_paths_dijkstra(weights=graph.es["weight"], mode=ig.ALL)
            local_min = np.amin(node_distances)
            local_max = np.amax(node_distances)
            if local_min < global_min: global_min = local_min
            if local_max > global_max: global_max = local_max
            
        return global_min, global_max

    def compute_ewd(self, bin_nr:int=30, ewd_vmin:float=-1.0, ewd_vmax:float=1.0)->np.ndarray:
        """Function that computes edge weight distributions from multiple trials, accordingly to Varga et al 2024.

        Args:
            bin_nr (int, optional): number of bins for the distribution. Defaults to 30.
            ewd_vmin (float, optional): minimum value for the distribution bins. Defaults to -1.0.
            ewd_vmax (float, optional): maximum value for the distribution bins. Defaults to 1.0.

        Returns:
            np.ndarray: array of shape (number of edges, number of bins) representing the edge weight distribution
        """
        bins = np.linspace(ewd_vmin, ewd_vmax, bin_nr+1)
        ewd = np.zeros((self.edge_nr, bin_nr))
        for t in range(self.trial_nr):
            weights = self.edgelist_arr[t][:, 2]
            for i in range(len(weights)):
                values, _ = np.histogram(weights[i], bins=bins)
                ewd[i] += values
        ewd /= self.trial_nr
        return ewd

    def compute_ndd(self, bin_nr:int=30, ndd_vmin:float=None, ndd_vmax:float=None, norm_factor:float=1.0)->np.ndarray:
        """Function that computes the node distance distribution accordingly to Varga et al. 2024

        Args:
            bin_nr (int, optional): _number of bins for the distribution. Defaults to 30.
            ndd_vmin (float, optional): minimum value for the distribution bins. Defaults to -1.
            ndd_vmax (float, optional): maximum value for the distribution bins. Defaults to -1.
            norm_factor (float, optional): normalizion factor for edge weights (useful when there's a correlation value above 1.0 due to numerical error/interpolation in SCA). Defaults to 1.0.

        Returns:
            np.ndarray: array of shape (number of nodes, number of bins) representing the node distance distribution 
        """
        if ndd_vmin is None and ndd_vmax is None:
            ndd_vmin, ndd_vmax = self.get_minmax_distances(norm_factor=norm_factor)
        bins = np.linspace(ndd_vmin, ndd_vmax, bin_nr + 1)
        ndd = np.zeros((self.node_nr, bin_nr))
        for t in range(self.trial_nr):
            sources, targets, weights = self.edgelist_arr[t].T
            sources = sources.astype(int)
            targets = targets.astype(int)
            lengths = -np.log(np.abs(weights/norm_factor)) # link lengths: l_{ij} = -log(|w_{ij}|); normalize weights above +-1.0 to avoid negative lengths!
            graph = ig.Graph(n=self.node_nr, directed=True)
            graph.add_edges(list(zip(sources, targets)))
            graph.delete_vertices(graph.vs.select(_degree=0))
            graph.vs["name"] = np.arange(1, self.node_nr)
            graph.es["weight"] = lengths
            node_distances = graph.shortest_paths_dijkstra(weights=graph.es["weight"], mode=ig.ALL)
            values_row = []
            for row in node_distances:
                values, _ = np.histogram(row, bins=bins)
                values = values/self.node_nr # /= doesnt work here due to values being array of ints initially
                values_row += [values]
            
            ndd += values_row
        ndd /= self.trial_nr
        return ndd

    def compute_newd(self, bin_nr:int=30, newd_vmin:float=-1.0, newd_vmax:float=1.0)->np.ndarray:
        """Function that computes the node edge weight distribution (aggregated EWD for the nodes), accordingly to Varga et al. 2024

        Args:
            bin_nr (int, optional): number of bins for the distribution. Defaults to 30.
            newd_vmin (float, optional): mimimum value for the distribution bins. Defaults to -1.0.
            newd_vmax (float, optional): maximum value for the distribution bins. Defaults to 1.0.

        Returns:
            np.ndarray: array of shape (number of nodes, number of bins) representing the node edge weight distribution
        """
        bins = np.linspace(newd_vmin, newd_vmax, bin_nr+1)
        newd = np.zeros((self.node_nr, bin_nr))
        for t in range(self.trial_nr):
            sources, targets, weights = self.edgelist_arr[t].T
            sources = sources.astype(int)
            targets = targets.astype(int)
            graph = ig.Graph(n=self.node_nr, directed=True)
            graph.add_edges(list(zip(sources, targets)))
            graph.delete_vertices(graph.vs.select(_degree=0))
            graph.vs["name"] = np.arange(1, self.node_nr)
            graph.es["weight"] = weights
            res = []
            for v in graph.vs:
                edges = graph.incident(v, mode=ig.ALL) # find incident edges/links to node
                res_row = []
                for e in edges:
                    res_row += [graph.es[e]["weight"]]
                res += [res_row]
            
            res = np.array(res)
            values_row = []
            for row in res:
                values, _ = np.histogram(row, bins=bins)
                values = values/(self.node_nr - 1) # - 1 because there are no loops in this network!
                values_row += [values]
            
            newd += values_row
        newd /= self.trial_nr
        return newd

    def compute_ew_for_Cliffs_delta(self)->np.ndarray:
        """Function that prepares the edge weights for Cliff's delta computation, accordingly to Varga et al. 2024

        Returns:
            np.ndarray: array of shape (number of edges, number of trials) representing the edge weights for all trials
        """
        for t in range(self.trial_nr):
            weights = self.edgelist_arr[t][:, 2]
            weights = np.expand_dims(weights, axis=1)
            if t == 0:
                ewd_Cliffs = weights
            else:
                ewd_Cliffs = np.append(ewd_Cliffs, weights, 1)
        
        return ewd_Cliffs


    def compute_nd_for_Cliffs_delta(self, norm_factor:float=1.0)->np.ndarray:
        """Function that computes the node distances for Cliff's delta accordingly to Varga et al. 2024

        Args:
            norm_factor (float, optional): normalizion factor for edge weights (useful when there's a correlation value above 1.0 due to numerical error/interpolation in SCA). Defaults to 1.0.

        Returns:
            np.ndarray: array of shape (number of nodes, number of nodes * number of trials) representing the node distances for all trials
        """
        for t in range(self.trial_nr):
            sources, targets, weights = self.edgelist_arr[t].T
            sources = sources.astype(int)
            targets = targets.astype(int)
            lengths = -np.log(np.abs(weights/norm_factor)) # link lengths: l_{ij} = -log(|w_{ij}|); normalize weights above +-1.0 to avoid negative lengths!
            graph = ig.Graph(n=self.node_nr, directed=True)
            graph.add_edges(list(zip(sources, targets)))
            graph.delete_vertices(graph.vs.select(_degree=0))
            graph.vs["name"] = np.arange(1, self.node_nr)
            graph.es["weight"] = lengths
            node_distances = graph.shortest_paths_dijkstra(weights=graph.es["weight"], mode=ig.ALL)
            
            if t == 0:
                ndd_Cliffs = node_distances
            else:
                ndd_Cliffs = np.append(ndd_Cliffs, node_distances, 1)

        return ndd_Cliffs

    def compute_new_for_Cliffs_delta(self)->np.ndarray:
        """Function that computes the node edge weights for Cliff's delta (aggregated EWD for the nodes), accordingly to Varga et al. 2024

        Args:
            -
        Returns:
            np.ndarray: array of shape (number of nodes, number of nodes * number of trials) representing the node edge weights for all trials
        """
        for t in range(self.trial_nr):
            sources, targets, weights = self.edgelist_arr[t].T
            sources = sources.astype(int)
            targets = targets.astype(int)
            graph = ig.Graph(n=self.node_nr, directed=True)
            graph.add_edges(list(zip(sources, targets)))
            graph.delete_vertices(graph.vs.select(_degree=0))
            graph.vs["name"] = np.arange(1, self.node_nr)
            graph.es["weight"] = weights
            res = []
            for v in graph.vs:
                edges = graph.incident(v, mode=ig.ALL) # find incident edges/links to node
                res_row = []
                for e in edges:
                    res_row += [graph.es[e]["weight"]]
                res += [res_row]
            
            res = np.array(res)
            
            if t == 0:
                newd_Cliffs = res
            else:
                newd_Cliffs = np.append(newd_Cliffs, res, 1)

        return newd_Cliffs
