import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

def plot_distribution(data:np.ndarray, sorting:np.ndarray=None, fig:matplotlib.figure.Figure=None, ax:matplotlib.axes._axes.Axes=None, cmap:matplotlib.colors.LinearSegmentedColormap=cm.coolwarm, cmin:float=None, cmax:float=None, clabel:str="", show_cbar:bool=False, xlabel:str=None, ylabel:str=None, title:str=None, vmin:float=None, vmax:float=None, yticks:np.ndarray=None, mirror_yticks:bool=False)->None:
    """Function that plots a heatmap based on a 2D array. It can be used to visualize distributions (EWD, NDD, N-EWD) or to plot the results of their comparison (Cliff's delta values).

    Args:
        data (np.ndarray): Array of 2D.
        sorting (np.ndarray, optional): Sorting indices of the first dimension (e.g. nodes). Defaults to None.
        fig (matplotlib.figure.Figure, optional): Plot on given matplotlib figure if given. Defaults to None.
        ax (matplotlib.axes._axes.Axes, optional): Plot on given matplotlib axis if given. Defaults to None.
        cmap (matplotlib.colors.LinearSegmentedColormap, optional): Colormap of the heatmap. Defaults to cm.coolwarm.
        cmin (float, optional): Minimum value of the colorbar if given. Defaults to None.
        cmax (float, optional): Maximum value of the colorbar if given. Defaults to None.
        clabel (str, optional): Label of the colorbar if given. Defaults to "".
        show_cbar (bool, optional): Show colorbar or not. Defaults to False.
        xlabel (str, optional): Label of the x axis of the heatmap if given. Defaults to None.
        ylabel (str, optional): Label of the y axis of the heatmap if given. Defaults to None.
        title (str, optional): Title of the plot if given. Defaults to None.
        vmin (float, optional): First tick label of the x axis if given. Defaults to None.
        vmax (float, optional): Last tick label of the x axis  if given. Defaults to None.
        yticks (np.ndarray, optional): Array of tick labels of the y axis of the heatmap (e.g. node names). Defaults to None.
        mirror_yticks (bool, optional): Mirror tick labels of the y axis or not (useful for readability, labels with odd indices appear on the left, even ones on the right). Defaults to False.

    """

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 10))
    
    if sorting is None: sorting = np.arange(data.shape[0]) # default sorting indices of rows

    if cmin is None or cmax is None:
        act_cmin = np.min(data)
        act_cmax = np.max(data)
    else:
        act_cmin = cmin
        act_cmax = cmax

    act_cmin, act_cmax = np.round(act_cmin, 2), np.round(act_cmax, 2) # round to 2 decimals

    im = ax.imshow(data[sorting], interpolation="nearest", cmap=cmap, aspect="auto", vmin=act_cmin, vmax=act_cmax) # NOTE: act_max mar legyen felkerektive
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    if vmin is not None and vmax is not None:
        xticks = [np.round(vmin, 2), np.round(vmax, 2)]
        ax.set_xticks([-0.5, data.shape[1] - 1 + 0.5], xticks) 
    else:
        ax.set_xticks([], [])
    if yticks is not None:
        yticks = np.array(yticks)
        if mirror_yticks: # put yticks of even indices on the left, odd indices on the right (for readability)
            ax.set_yticks(np.arange(0, data.shape[0], 2), yticks[sorting][::2])
            rhs_ax = ax.twinx()
            rhs_ax.set_yticks(np.arange(1, data.shape[0], 2), yticks[sorting][1::2])
            rhs_ax.set_ylim(ax.get_ylim()) # this ensures that the limits are same on left and rhs
        else:
            ax.set_yticks(np.arange(0, data.shape[0], 1), yticks[sorting])
    else:
        ax.set_yticks([], [])

    if show_cbar:
        ax_pos = ax.get_position()
        cbar_ticks = [act_cmin, act_cmax]
        cbar = fig.colorbar(im, cax=fig.add_axes([ax_pos.x0, ax_pos.y0 - 0.1, ax_pos.width, 0.02]), orientation="horizontal", ticks=cbar_ticks)
        if np.abs(cbar_ticks[0]) < 0.01:
            cbar.ax.set_xticklabels(["0", str(cbar_ticks[1])])
        else:
            cbar.ax.set_xticklabels([str(cbar_ticks[0]), str(cbar_ticks[1])])
        cbar.set_label(clabel)

def plot_Cliffs_delta(deltas:np.ndarray, sorting:np.ndarray=None, fig:matplotlib.figure.Figure=None, axes:list[matplotlib.axes._axes.Axes]=None, clabel:str="", show_cbar:bool=False, titles:list[str]=None)->None:
    """Plot Cliff's delta statistic resulting from the comparison of two 2D distributions. Two separate axes are plotted, one with the delta values and the other showing the absolute thresholds representing significance based on Vargha et al. 2000

    Args:
        deltas (np.ndarray): Array of Cliff deltas
        sorting (np.ndarray, optional): Sorting indices of the first dimension (e.g. nodes). Defaults to None.
        fig (matplotlib.figure.Figure, optional): Plot on given matplotlib figure if given. Defaults to None.
        ax (matplotlib.axes._axes.Axes, optional): Plot on given matplotlib axis if given. Defaults to None.
        clabel (str, optional): Label of the colorbar if given. Defaults to "".
        show_cbar (bool, optional): Show colorbar or not. Defaults to False.
        titles (list[str], optional): Title of the plots if given. Defaults to None.

    Raises:
        ValueError: The number of provided axes is not two (one for deltas, another one for the thresholded values)
    """
    if len(axes) != 2:
        raise ValueError("Two axes must be provided, one for the Cliff's delta values and the other for their thresholded versions.")

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 10))

    if sorting is None: sorting = np.arange(data.shape[0]) # default sorting indices of rows

    cmin, cmax = -1.0, 1.0
    
    if len(deltas.shape) == 1:
        deltas = np.expand_dims(deltas, axis=1)

    # colormaps of deltas
    c1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#4A2659', '#927D9B'])
    colors1 = c1(np.linspace(0, 1, 5700))
    c2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#3836CA', '#8886DF'])
    colors2 = c2(np.linspace(0, 1, 1500))
    c3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#6BC6D6', '#A6DDE6'])
    colors3 = c3(np.linspace(0, 1, 1700))
    c3mid = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#E6E6E6', '#FFFFFF'])
    colors3mid = c3mid(np.linspace(0, 1, 1100))
    c4mid = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#FFFFFF', '#E6E6E6'])
    colors4mid = c4mid(np.linspace(0, 1, 1100))
    c4 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#80C680', '#2CA02C'])
    colors4 = c4(np.linspace(0, 1, 1700))
    c5 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#FFB26E', '#FF7F0E'])
    colors5 = c5(np.linspace(0, 1, 1500))
    c6 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#E67D7E', '#D62728'])
    colors6 = c6(np.linspace(0, 1, 5700))
    cmap = np.concatenate((colors1, colors2), axis=0)
    cmap = np.concatenate((cmap, colors3), axis=0)
    cmap = np.concatenate((cmap, colors3mid), axis=0)
    cmap = np.concatenate((cmap, colors4mid), axis=0)
    cmap = np.concatenate((cmap, colors4), axis=0)
    cmap = np.concatenate((cmap, colors5), axis=0)
    cmap = np.concatenate((cmap, colors6), axis=0)
    cmap = (matplotlib.colors.LinearSegmentedColormap.from_list("", cmap, N=1000000).with_extremes(over='#000000'))

    im0 = axes[0].imshow(deltas[sorting], interpolation="nearest", cmap=cmap, aspect="auto", vmin=cmin, vmax=cmax)
    axes[0].set_xticks([], [])
    axes[0].set_yticks([], [])
    if titles is not None: axes[0].set_title(titles[0])
    if show_cbar:
        ax_pos0 = axes[0].get_position()
        ax_pos1 = axes[1].get_position()
        # decide rounding based on sign
        if cmin < 0:
            round_cmin = np.ceil
        else:
            round_cmin = np.floor
        if cmax < 0:
            round_cmax = np.ceil
        else:
            round_cmax = np.floor
        cbar_ticks = [round_cmin(100*cmin)/100, round_cmax(100*cmax)/100] # round to 2 decimals
        cbar = fig.colorbar(im0, cax=fig.add_axes([ax_pos0.x0, ax_pos0.y0 - 0.1, ax_pos1.x0 - ax_pos0.x0 + ax_pos1.width, 0.02]), orientation="horizontal", ticks=cbar_ticks)
        if np.abs(cbar_ticks[0]) < 0.01:
            cbar.ax.set_xticklabels(["0", str(cbar_ticks[1])])
        else:
            cbar.ax.set_xticklabels([str(cbar_ticks[0]), str(cbar_ticks[1])])
        cbar.set_label(clabel)

    cmap_thresholded = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff', '#2ca02c', '#ff7f0e', '#d62728', '#000000'])
    # defining thresholds based on Vargha et al. 2000
    SMALL_DELTA_THRESHOLD = 0.11
    MEDIUM_DELTA_THRESHOLD = 0.28
    LARGE_DELTA_THRESHOLD = 0.43
    thresholded_deltas = np.zeros_like(deltas)
    if deltas.shape[1] == 1:
        thresholded_deltas[np.abs(deltas) <= SMALL_DELTA_THRESHOLD] = 0
        small_deltas = np.abs(deltas) > SMALL_DELTA_THRESHOLD
        thresholded_deltas[small_deltas] = 1
        medium_deltas = np.abs(deltas) > MEDIUM_DELTA_THRESHOLD
        thresholded_deltas[medium_deltas] = 2
        large_deltas = np.abs(deltas) > LARGE_DELTA_THRESHOLD
        thresholded_deltas[large_deltas] = 3
        nan_deltas = np.isnan(deltas)
        thresholded_deltas[nan_deltas] = 4
        ## display text of thresholded values only if there are such in the data
        if (nan_deltas == True).any():
            axes[1].text(2.0, 0.3, "N/A", color='#000000', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (small_deltas == True).any():
            axes[1].text(2.0, 0.4, "small", color='#2ca02c', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (medium_deltas == True).any():
            axes[1].text(2.0, 0.5, "medium", color='#ff7f0e', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (large_deltas == True).any():
            axes[1].text(2.0, 0.6, "large", color='#d62728', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
    
    elif deltas.shape[1] == 2: # if there are two sets of Cliff's delta (case of bimodal distributions)
        # negative part
        thresholded_deltas[np.abs(deltas[:, 0]) <= SMALL_DELTA_THRESHOLD, 0] = 0
        small_deltas_neg = np.abs(deltas[:, 0]) > SMALL_DELTA_THRESHOLD
        thresholded_deltas[small_deltas_neg, 0] = 1
        medium_deltas_neg = np.abs(deltas[:, 0]) > MEDIUM_DELTA_THRESHOLD
        thresholded_deltas[medium_deltas_neg, 0] = 2
        large_deltas_neg = np.abs(deltas[:, 0]) > LARGE_DELTA_THRESHOLD
        thresholded_deltas[large_deltas_neg, 0] = 3
        nan_deltas_neg = np.isnan(deltas[:, 0])
        thresholded_deltas[nan_deltas_neg, 0] = 4
        # positive part
        thresholded_deltas[np.abs(deltas[:, 1]) <= SMALL_DELTA_THRESHOLD, 1] = 0
        small_deltas_pos = np.abs(deltas[:, 1]) > SMALL_DELTA_THRESHOLD
        thresholded_deltas[small_deltas_pos, 1] = 1
        medium_deltas_pos = np.abs(deltas[:, 1]) > MEDIUM_DELTA_THRESHOLD
        thresholded_deltas[medium_deltas_pos, 1] = 2
        large_deltas_pos = np.abs(deltas[:, 1]) > LARGE_DELTA_THRESHOLD
        thresholded_deltas[large_deltas_pos, 1] = 3
        nan_deltas_pos = np.isnan(deltas[:, 1])
        thresholded_deltas[nan_deltas_pos, 1] = 4
        ## display text of thresholded values only if there are such in the data
        if (nan_deltas_neg == True).any() or (nan_deltas_pos == True).any():
            axes[1].text(1.5, 0.3, "N/A", color='#000000', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (small_deltas_neg == True).any() or (small_deltas_pos == True).any():
            axes[1].text(1.5, 0.4, "small", color='#2ca02c', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (medium_deltas_neg == True).any() or (medium_deltas_pos == True).any():
            axes[1].text(1.5, 0.5, "medium", color='#ff7f0e', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)
        if (large_deltas_neg == True).any() or (large_deltas_pos == True).any():
            axes[1].text(1.5, 0.6, "large", color='#d62728', rotation=90, weight='bold', ha='center', va='center', transform=axes[1].transAxes)

    # plotting
    im1 = axes[1].imshow(thresholded_deltas[sorting], interpolation="nearest", cmap=cmap_thresholded, aspect="auto", vmin=0, vmax=4)
    axes[1].set_xticks([], [])
    axes[1].set_yticks([], [])
    if titles is not None: axes[1].set_title(titles[1])



