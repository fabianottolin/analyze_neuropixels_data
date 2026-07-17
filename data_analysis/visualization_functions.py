### visualization functions ###
## code so that u can reuse the functions for future projects

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spikeinterface as si
import pynapple as nap
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from typing import Literal
from basic_analyses import average_PSTH


def depth_rasterplot(data, alpha = 0.2):
    """
    ------ Parameters ------
    data: TsGroup object, SortingAnalyzer object, or Path/string
        TsGroup object should have depth information in metadata (named "depth")
        Path/string should point to spikeinterface SortingAnalyzer
    """
    if isinstance(data, (str, Path)):
        data = si.load(data)
    
    fig, ax = plt.subplots()

    if isinstance(data, nap.TsGroup):
        plot_data = data.to_tsd("depth")
        ax.plot(plot_data, marker='|', color='black', linestyle="None", markersize=0.1, alpha=alpha)
    elif isinstance(data, si.SortingAnalyzer):
        df_plot = pd.DataFrame({"depth": data.get_extension("unit_locations").get_data()[:, 1], # 1 -> y coordinate
                    "spike_times": [data.sorting.get_unit_spike_train(unit, return_times=True, segment_index=0) for unit in data.unit_ids]}).explode("spike_times")
        ax.plot(df_plot["spike_times"], df_plot["depth"], marker='|', color='black', linestyle="None", markersize=0.1, alpha=alpha)

    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Depth (µm)", fontsize=18)
    ax.set_title("Spike times of all units", fontsize=20)
    
    return fig, ax


def overlay_stimuli(ax, stimulus_epochs, stimulus_column = "stimulus", alpha = 0.1, default_color = "#53b874", colors_per_stimulus = None): # rename overlay_stimulus_epochs
    """
    ------ Parameters ------
    ax: matplotlib axis object
        stimulus_epochs are overlayed on this axis
    stimulus_epochs: nap.IntervalSet
        IntervalSet containing start & end times of stimuli
    stimulus_column: str (default: "stimulus")
        Name of metadata column in stimulus_epochs IntervalSet containing stimulus information
        Only needed if colors_per_stimulus != None
    alpha: float (default: 0.1)
    default_color: str (default: "#53b874")
    colors_per_stimulus: None or dict or matplotlib colormap (default: None)
        Defines colors to use for stimulus overlays
        None -> same color (default_color) is used for all stimuli
        dict -> mapping of stimulus labels to colors (e.g.: {stimulus_label1: color1, stimulus_label2: color2, ...})
        matplotlib colormap -> continuous stimulus values are mapped to colors using provided colormap
    """
    if colors_per_stimulus is None: # one color for all stimuli
        for start, end in zip(stimulus_epochs["start"], stimulus_epochs["end"]):
            ax.axvspan(start, end, alpha = alpha, color = default_color)
        return ax
    
    if isinstance(colors_per_stimulus, dict): # dict color mapping
        for start, end, stimulus_type in zip(stimulus_epochs["start"], stimulus_epochs["end"], stimulus_epochs[stimulus_column]):
            color = colors_per_stimulus.get(stimulus_type, default_color)
            ax.axvspan(start, end, alpha = alpha, color = color)
    elif isinstance(colors_per_stimulus, mcolors.Colormap):
        normalization = mcolors.Normalize(vmin=np.min(stimulus_epochs[stimulus_column]), vmax=np.max(stimulus_epochs[stimulus_column]))
        for start, end, stimulus_type in zip(stimulus_epochs["start"], stimulus_epochs["end"], stimulus_epochs[stimulus_column]):
            color = colors_per_stimulus(normalization(stimulus_type))
            ax.axvspan(start, end, alpha=alpha, color=color)
    else:
        raise ValueError("colors_per_stimulus must be None, dict, or matplotlib colormap.")

    return ax


def _plot_all_stimulus_onsets(axes, SOA, time_range, stimulus_duration = None, color = "#4D575C", shading = True, alpha_shading = 0.25, **kwargs):
    """
    ------ Parameters ------
    time_range: tuple (min, max)
    """
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    if stimulus_duration is None:
        if shading: print("Warning: To plot shaded areas for stimuli, stimulus_duration must be provided.")
        stimulus_duration = 0
    
    onsets = np.arange(np.floor(time_range[0] / SOA) * SOA, np.ceil(time_range[1] / SOA) * SOA + SOA, SOA)
    onsets = onsets[(onsets >= time_range[0]) & (~np.isclose(onsets, 0)) & (onsets <= time_range[1]-stimulus_duration)] # stimulus at 0 already plotted
    for onset_time in onsets:
        for ax in axes:
            if shading: ax.axvspan(onset_time, onset_time + stimulus_duration, alpha=alpha_shading, color=color)
            ax.axvline(onset_time, color=color, linestyle="--", **kwargs)
    
    return axes


def plot_unit_psth(ps_times_unit, mean_rates_unit, stimulus_duration, soa = None, color = "#4D575C", stimulus_color = "#53b874", other_stimulus_color = "#53b874", axes = None):
    # TODO: split into plot_unit_psth() and unit_rasterplot(), then def unit_PSTH_and_rasterplot() calling both 
    """
    ------ Parameters ------
    ps_times_unit
    mean_rates_unit
    stimulus_duration: float
    soa: float or None (default: None)
        If provided, other stimuli in window are plotted at multiples of stimulus onset asynchrony (SOA)
    color: str (default: "#4D575C")
    stimulus_color: str (default: "#53b874")
        Decides color stimulus at t=0 is plotted in
    other_stimulus_color: str (default: "#53b874")
        Decides color other stimuli (those not at t=0) are plotted in
    axes: tuple (default: None)
    """
    if axes is None:
        fig, (ax_mean, ax_spikes_raster) = plt.subplots(2, 1, sharex = True, height_ratios = [0.3, 1], figsize=(6, 6), gridspec_kw={"hspace": 0.3})
    else:
        ax_mean, ax_spikes_raster = axes
        fig = ax_mean.get_figure()
    sns.despine()

    ax_mean.plot(mean_rates_unit, color=color)
    ax_mean.set_ylabel("Spikes/s", fontsize=18)
    ax_mean.axvspan(0, stimulus_duration, alpha=0.25, color=stimulus_color)
    ax_mean.axvline(0, color=stimulus_color, linestyle="--")
    ax_mean.tick_params(labelsize=18)

    ax_spikes_raster.plot(ps_times_unit.to_tsd(), "|", markersize=5, color=color)
    ax_spikes_raster.set_ylabel("Event", fontsize=18)
    ax_spikes_raster.axvspan(0, stimulus_duration, alpha=0.25, color=stimulus_color)
    ax_spikes_raster.axvline(0, color=stimulus_color, linestyle="--")
    if soa is not None: # plot other stimuli as well
        bin_size = round(mean_rates_unit.index[1] - mean_rates_unit.index[0], 4)
        time_range = ((min(mean_rates_unit.index) - bin_size/2), (max(mean_rates_unit.index) + bin_size/2))
        _plot_all_stimulus_onsets([ax_mean, ax_spikes_raster], soa, time_range, stimulus_duration = stimulus_duration, color = other_stimulus_color)
    ax_spikes_raster.set_xlabel("Time from event (s)", fontsize=18)
    ax_spikes_raster.tick_params(labelsize=18)

    return fig, ax_mean, ax_spikes_raster


def plot_fq_heatmap(plot_matrix_unit, edges, stimulus_duration, smooth = True, sigma_smoothing = (1,1), ax = None):
    """
    Plots time (relative to stimulus onset) on x, frequency on y, firing rate (spikes/s) as color.
    Frequencies plotted on log scale.

    ------ Parameters ------
    plot_matrix_unit
    edges
    stimulus_duration
    smooth
    sigma_smoothing
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if smooth:
        plot_matrix_unit = gaussian_filter(plot_matrix_unit, sigma = sigma_smoothing) # adjust sigma for more/less smoothing
        ax.set_title(f"Frequency heatmap (smoothed)", fontsize=20)
    else:
        ax.set_title(f"Frequency heatmap", fontsize=20)

    img = ax.pcolormesh(edges["time_edges"], edges["fq_edges_hz"], plot_matrix_unit, shading="auto")
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    ax.axvline(0, color="white", lw=1, linestyle="--")
    ax.axvline(stimulus_duration, color="white", lw=1, linestyle="--")
    ax.set_xlabel("Time relative to stimulus onset (s)", fontsize=18)
    ax.set_ylabel("Frequency (kHz)", fontsize=18)
    ax.tick_params(labelsize = 18)
    legend_bar = plt.colorbar(img, ax = ax)
    legend_bar.set_label("Average firing rate (spikes/s)", fontsize=18)
    legend_bar.ax.tick_params(labelsize=18)
    
    return fig, ax


def plot_fq_population_heatmap(plot_matrix, fq_edges_hz, smooth = True, sigma_smoothing = (0, 1), ax=None, vmin=None, vmax=None, estimated_best_frequency = None):
    """
    Plots units on y, frequency (log-scaled) on x and response as color.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if smooth:
        plot_matrix = gaussian_filter(plot_matrix, sigma=sigma_smoothing)
        ax.set_title("Best frequency population heatmap (smoothed)", fontsize=20)
    else:
        ax.set_title("Best frequency population heatmap", fontsize=20)

    img = ax.pcolormesh(fq_edges_hz, np.arange(len(plot_matrix) + 1), plot_matrix, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Frequency (kHz)", fontsize=18)
    ax.set_ylabel("Units", fontsize=18)
    ax.tick_params(left=False, labelleft=False, labelsize=18)

    legend_bar = plt.colorbar(img, ax=ax)
    legend_bar.set_label("Normalized firing rate", fontsize=18)
    legend_bar.ax.tick_params(labelsize=18)

    if estimated_best_frequency is not None:
        ax.axvline(estimated_best_frequency, color="#f83687", linestyle='--', label="BF estimate")
        ax.legend(fontsize=18, bbox_to_anchor=(1.3, 1.05), frameon=False, loc='upper left')

    return fig, ax


def plot_mean_psth(data, stimulus_duration, units = None, soa = None, ax = None, color = "#4D575C", plot_stimulus = True, stimulus_color = "#53b874",
                   other_stimulus_color = "#53b874", shading = True, alpha_shading = 0.2, kwargs_main_plot = {}, sigma_smoothing_s = None, verbose = True):
    """
    ------ Parameters ------
    data: dict or pd.DataFrame
        Dict from get_peristimulus_data(), unit ids as keys, contains mean firing rates (spikes/s) per bin
        pd.DataFrame from  get_grand_mean_PSTH_df(), columns are unit/recording ids, index is time bins, values are mean firing rates (spikes/s) per bin
    stimulus_duration: float
        Stimulus duration in seconds
    units: list or None (default: None)
        Units to include in averaged PSTH
        If None all units in data are included
    soa: float or None (default: None)
        If provided, other stimuli in window are plotted at multiples of stimulus onset asynchrony (SOA)
    kwargs_main_plot: dict
        Additional kwargs passed as dict will be applied to matplotlib.ax.plot() plotting mean time per bin
    sigma_smoothing_s: float or None (default: None)
        If provided, applies Gaussian smoothing with specified sigma (in seconds) to the mean and standard error of the mean
        If None no smoothing is applied
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    sns.despine()

    if units is None and verbose:
        print("No units specified, using all units in data.")
    
    means_per_time_bin, SE_per_time_bin = average_PSTH(data, units)

    if sigma_smoothing_s is not None:
        bin_size = means_per_time_bin.index[1] - means_per_time_bin.index[0]
        sigma_smoothing_bins = sigma_smoothing_s / bin_size
        means_per_time_bin[:] = gaussian_filter(means_per_time_bin.values, sigma=sigma_smoothing_bins)
        SE_per_time_bin[:] = gaussian_filter(SE_per_time_bin.values, sigma=sigma_smoothing_bins)

    # plot mean & SE
    ax.plot(means_per_time_bin, color=color, **kwargs_main_plot)
    ax.set_title(f"Grand average PSTH", fontsize=20)
    ax.set_xlabel("Time from event (s)", fontsize=18)
    ax.set_ylabel("Spikes/s", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.fill_between(means_per_time_bin.index, means_per_time_bin - SE_per_time_bin, means_per_time_bin + SE_per_time_bin, color=color, alpha=0.2)

    if plot_stimulus:
        ax.axvspan(0, stimulus_duration, alpha=0.25, color=stimulus_color)
        ax.axvline(0, color=stimulus_color, linestyle="--")

    if soa is not None:
        bin_size = round(means_per_time_bin.index[1] - means_per_time_bin.index[0], 4)
        ax, = _plot_all_stimulus_onsets([ax], soa, (means_per_time_bin.index[0]-bin_size/2, means_per_time_bin.index[-1]+bin_size/2), stimulus_duration = stimulus_duration,
                                        color = other_stimulus_color, shading = shading, alpha_shading = alpha_shading)
   
    return fig, ax


def overlay_protocol_type(ax, protocol_names, stimuli_per_protocol: list, color_map = plt.cm.gist_rainbow):
    """
    ------ Parameters ------
    protocol_names
        Names of protocols (in order of presentation)
    stimuli_per_protocol: list
        Number of stimuli of given type for each protocol (in order of presentation)
    color_map: dict or matplotlib colormap (default: plt.cm.gist_rainbow)
        If dict, keys are protocol names (or unique identifiers in protocol names) and values are colors
    """
    if isinstance(color_map, dict): # dict color mapping
        def get_color(i, protocol_names):
            section_color = next((value for key, value in color_map.items() if key.lower() in protocol_names[i].lower()), None)
            if section_color is None:
                print(f"Warning: No color found for protocol {protocol_names[i]} in color_map. Using default color.")
                section_color = "white" # default color
            return section_color
    elif isinstance(color_map, mcolors.Colormap):
        def get_color(i, protocol_names):
            return color_map(i/len(protocol_names))
    else:
        raise ValueError("color_map must be dict or matplotlib colormap.")

    previous_stimulus = 0
    for i in range(len(protocol_names)):
        section_color = get_color(i, protocol_names)
        ax.axhspan(previous_stimulus, previous_stimulus + stimuli_per_protocol[i], color = section_color, alpha = 0.1, label = protocol_names[i])
        previous_stimulus += stimuli_per_protocol[i]
    
    handles, labels = ax.get_legend_handles_labels() # make sure legend order matches plot
    ax.legend(handles[::-1], labels[::-1], loc = "upper left", bbox_to_anchor = (1, 1), frameon = False, fontsize = 18)

    return ax


def plot_population_heatmap(plot_matrix, time_edges, stimulus_duration = None, soa= None, smooth = True, sigma_smoothing = (1,1), ax = None,
                            vmin = None, vmax = None, t0_color = "white", shading = True, alpha_shading = 0.1, label_legend_bar = "Firing rate"):
    """
    Plots time (relative to stimulus onset) on x, unit ids (sorted as specified in get_population_heatmap_data) on y, firing rate as color.

    ------ Parameters ------
    plot_matrix
    time_edges
    stimulus_duration
    smooth: bool (default: True)
    sigma_smoothing: tuple (default: (1,1))
    ax
    vmin
        Clips colormap, values below vmin are shown in same color as vmin
    vmax
        Clips colormap, values above vmax are shown in same color as vmax
    label_legend_bar: str
        Label for the colorbar
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if smooth:
        plot_matrix = gaussian_filter(plot_matrix, sigma = sigma_smoothing) # adjust sigma for more/less smoothing
        ax.set_title(f"Population heatmap (smoothed)", fontsize=20)
    else:
        ax.set_title(f"Population heatmap", fontsize=20)

    img = ax.pcolormesh(time_edges, np.arange(len(plot_matrix) + 1), plot_matrix, vmin=vmin, vmax=vmax, shading="auto") # y-axis is unit ids
    ax.axvline(0, color=t0_color, lw=1, linestyle="--")
    if (stimulus_duration is not None) and shading:
        ax.axvspan(0, stimulus_duration, alpha=alpha_shading, color="white")
    if soa is not None:
        ax, = _plot_all_stimulus_onsets([ax], soa, (time_edges[0], time_edges[-1]), stimulus_duration = stimulus_duration, color = "#FFFFFF", linewidth = 1, shading = shading, alpha_shading = alpha_shading)
    ax.set_xlabel("Time relative to stimulus onset (s)", fontsize=18)
    ax.set_ylabel("Units", fontsize=18)
    ax.tick_params(left = False, labelleft=False, labelsize = 18)
    legend_bar = plt.colorbar(img, ax = ax)
    legend_bar.set_label(label_legend_bar, fontsize=18)
    legend_bar.ax.tick_params(labelsize=18)
    
    return fig, ax


def highlight_units_in_heatmap(heatmap_sorted_ids, highlighted_ids: list, heatmap_time_edges, heatmap_ax, color_highlight="#f83687", line_width=1, side_line_offset = 0.0035, verbose = False):
    """
    highlighted_ids: list
        Ids that will be highlighted in heatmap
    """
    id_to_row = {uid: row_i for row_i, uid in enumerate(heatmap_sorted_ids)}
    highlighted_rows = sorted(id_to_row[uid] for uid in highlighted_ids if uid in id_to_row)

    if len(highlighted_rows) == 0:
        raise ValueError("Units to be highlighted and units shown in heatmap don't overlap.")
    
    if verbose: print(f"{(len(highlighted_rows) / len(highlighted_ids))*100:.2f}% of units to highlight are included in the heatmap.")

    blocks = [] # get list of starts and ends of consecutively plotted units in heatmap that should be highlighted
    start = highlighted_rows[0]
    end = highlighted_rows[0]
    for row in highlighted_rows[1:]:
        if row == end + 1:
            end = row
        else:
            blocks.append((start, end + 1))
            start = end = row
    blocks.append((start, end + 1))

    x_min, x_max = heatmap_time_edges[0] + side_line_offset, heatmap_time_edges[-1] - side_line_offset
    for start, end in blocks:
        heatmap_ax.axhline(start, color=color_highlight, lw=line_width, linestyle="-")
        heatmap_ax.axhline(end, color=color_highlight, lw=line_width, linestyle="-")        
        heatmap_ax.vlines(x_min, ymin=start, ymax=end, color=color_highlight, lw=line_width)
        heatmap_ax.vlines(x_max, ymin=start, ymax=end, color=color_highlight, lw=line_width)

    return heatmap_ax


def response_scatterplot(scatterplot_data, ax = None, identity_line: Literal["rising", "falling", None] = "rising", color = "#585858", color_by = None,
                         color_map = None, highlight_units = None, highlight_color = "#FC1494", identity_line_color = "gray", label_unit_ids = False,
                         **scatterplot_kwargs): # add highlight kwargs, then .update()?
    """
    Scatter plot of unit in two conditions.

    ------ Parameters ------
    scatterplot_data: pd.DataFrame
        Indices are unit_ids, columns "x" & "y" -> data plotted on x and y axis
        Output of get_response_scatterplot_data().
    ax: matplotlib.Axes (default: None)
    identity_line: Literal["rising", "falling", None] (default: "rising")
        Whether to plot rising, falling, or no identity line
    color:
        Color to use for points if color_by is None
    color_by: column name in scatterplot_data (default: None)
        If provided points will be colored according to values in this column, color argument will be ignored
        If you want to use some unit information that is not already in TsGroup.metadata add it to TsGroup.metadata before using get_response_scatterplot_data()
    color_map: str or dict
        You can pass a dictionary mapping categorical values in color_by to colors
    highlight_ids: list or None (default: None)
        Unit ids to highlight
    label_unit_ids: bool or list (default: False)
        Prints unit ids in plot. If True all unit ids are added, if list only the unit ids specified in the list
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sns.despine()

    if color_by is None:
        ax.scatter(scatterplot_data["x"], scatterplot_data["y"], color=color, **scatterplot_kwargs)
    elif isinstance(color_map, dict): # for categorical color values
        if (missing:=set(scatterplot_data[color_by]) - set(color_map)):
            print(f"Warning: Missing colors for categories: {missing} in color_map. Assigning default color {color} to these categories.")
            color_map.update({missing_category: color for missing_category in missing})
        category_colors = scatterplot_data[color_by].map(color_map)
        scatterplot = ax.scatter(scatterplot_data["x"], scatterplot_data["y"], c=category_colors, **scatterplot_kwargs)
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[category]) for category in scatterplot_data[color_by].unique()]
        legend_labels = [str(label) for label in scatterplot_data[color_by].unique()]
        fig.legend(legend_handles, legend_labels, loc="upper right", fontsize=18, frameon=False, bbox_to_anchor=(1.3, 0.9))
    else: # continuous color values
        scatterplot = ax.scatter(scatterplot_data["x"], scatterplot_data["y"], c=scatterplot_data[color_by].values, cmap=color_map, **scatterplot_kwargs)
        legend_bar = plt.colorbar(scatterplot, ax=ax)
        legend_bar.set_label(color_by, fontsize=18)
        legend_bar.ax.tick_params(labelsize=18)
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set(xlim = limits, ylim = limits) # ensure square axes ratios

    if identity_line == "rising":
        ax.plot(limits, limits, color=identity_line_color, linestyle="-", lw=1, zorder=0)
    elif identity_line == "falling":
        ax.plot(limits, limits[::-1], color=identity_line_color, linestyle="-", lw=1, zorder=0)

    if highlight_units is not None:
        if missing:= set(highlight_units) - set(scatterplot_data.index):
            print(f"Warning: {len(missing)} unit id(s) to highlight not found in scatterplot data unit ids: {missing}")
        highlighted = scatterplot_data[scatterplot_data.index.isin(highlight_units)]
        ax.scatter(highlighted["x"], highlighted["y"], facecolor = "none", edgecolor = highlight_color, linewidth = 1.5)

    if label_unit_ids:
        ids_to_label = scatterplot_data.index if isinstance(label_unit_ids, bool) else label_unit_ids
        if missing:= set(ids_to_label) - set(scatterplot_data.index):
            print(f"Warning: {len(missing)} unit id(s) to label not found in scatterplot data unit ids: {missing}")
        for unit_id, row in scatterplot_data[scatterplot_data.index.isin(ids_to_label)].iterrows():
            ax.annotate(str(unit_id), (row["x"], row["y"]), textcoords="offset points", xytext=(3.5, 3.5), fontsize=8, alpha=0.7)            

    ax.tick_params(labelsize=18)
    ax.set_title("Scatter plot units", fontsize=20)

    fig.tight_layout()

    return fig, ax