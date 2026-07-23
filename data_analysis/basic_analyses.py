### basic analyses functions ###
## code so that u can reuse the functions for future projects given sorting_analyzer

import numpy as np
import spikeinterface.extractors as si_extractors
import spikeinterface as si
import pandas as pd
import pynapple as nap
from pathlib import Path
from typing import Literal
from scipy.interpolate import interp1d
from scipy.stats import wilcoxon, chi2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from dataclasses import dataclass

from utilities import *


def extract_triggers_from_analogue(raw_recording: Path, trigger_channels: dict, thresholds: Literal["median"] = "median", threshold_std = 2):
    """
    Returns triggers in binary format (0 off, 1 on) for each trigger type specified in trigger_channels

    ------ Parameters ------
    raw_recording: Path
    trigger_channels: dict
        dictionary containg trigger types as keys and corresponding NIDQ channel_ids as values, defined in parameters.py
    thresholds: Literal["median"] (default: "median")
    threshold_std: float (default: 2)

    ------ Returns ------
    triggers_dict: dict
        Keys trigger types, values binary arrays where 1 if trigger signal > threshold
    nidq_sampling_rate: float
    """

    nidq_stream = si_extractors.read_spikeglx(raw_recording, stream_id="nidq")
    triggers_dict = {}

    for trigger_type, channel_id in trigger_channels.items():

        trigger_channel = nidq_stream.get_traces(channel_ids = [channel_id], return_in_uV = True)
        trigger_channel = ((trigger_channel + abs(trigger_channel.min())) / 1e6).flatten() # set minimum to 0 and transform to volts
        
        match thresholds:
            case "median":
                threshold = np.median(trigger_channel) + threshold_std*np.std(trigger_channel) # threshold threshold_std stds above median
            case _:
                raise ValueError(f"Threshold method {thresholds} not known, please choose from: 'median' or add other method.")
        
        triggers_binary = (trigger_channel > threshold).astype(int) # 1 if > threshold, 0 if <= threshold
        triggers_dict[trigger_type] = triggers_binary

    return triggers_dict, nidq_stream.get_sampling_frequency()


def build_sync_channel_dict(recording: Path, recording_software: str = "spikeGLX"): # contains stream names as keys and channel_ids as values
    sync_channel_dict = {}
    match recording_software:
        case "spikeGLX":
            for probe_i in range(len(check_streams(recording, ".ap-SYNC"))):
                sync_channel_dict[f"imec{probe_i}.ap-SYNC"] = f"imec{probe_i}.ap#SY0"
            sync_channel_dict["nidq"] = "nidq#XD0"
        case _:
            raise ValueError(f"Recording software {recording_software} is unknown, please choose from: 'spikeGLX' or add support.")
    return sync_channel_dict
 

def get_trigger_times(binary_channel, sampling_frequency, mode: Literal["onset", "offset"]): # returns array of trigger onset/offset times
    match mode:
        case "onset":
            binary_channel = np.concatenate([[0], binary_channel]) # this way np.diff returns correct index, also detects if trigger signal already high at t=0
            transitions = np.where(np.diff(binary_channel) > 0)[0]
        case "offset":
            binary_channel = np.concatenate([binary_channel, [0]]) # this way np.diff returns correct index, also detects if trigger signal already high at t=0
            transitions = np.where(np.diff(binary_channel) < 0)[0]
    trigger_timestamps = transitions / sampling_frequency
    return trigger_timestamps


def align_triggers(trigger_times: dict, interpolant):
    """
    ------ Returns ------
    triggers_aligned: dict
        Keys are trigger types, values are arrays of trigger times in seconds aligned to probe sync channel (based on interpolant passed to function).
    """
    triggers_aligned = {}
    for trigger_type, trigger_timestamps in trigger_times.items():
        triggers_aligned[trigger_type] = interpolant(trigger_timestamps)
    
    return triggers_aligned


def sync_triggers_and_recording(recording: Path, trigger_times: dict, recording_software: str = "spikeGLX"):
    """
    Synchronizes trigger times to repective probe sync channels.

    ------ Parameters ------
    recording: Path
    trigger_times: dict or list of dicts
        Keys are trigger types, values array of trigger times in seconds
        Also allows passing list of dicts (e.g.: [onset_times, offset_times])
    recording_software: str (default: "spikeGLX")

    ------ Returns ------
    triggers_synced: dict
        Probes as keys and trigger types (as specified in trigger_times dict) as sub-keys with the transformed synced trigger times in seconds as values.
    """
    # make trigger times list
    if isinstance(trigger_times, dict):
        trigger_times_list = [trigger_times]
        return_single = True
    elif isinstance(trigger_times, list):
        trigger_times_list = trigger_times
        return_single = False
    else:
        raise TypeError("trigger_times must be dict or list of dicts")

    sync_streams = check_streams(recording, ".ap-SYNC", verbose = True) # checks how many probes
    sync_channel_dict = build_sync_channel_dict(recording, recording_software)
    
    sync_pulse_times = {}
    
    print("Extracting sync pulse times from recording...")
    for stream_id, channel_id in sync_channel_dict.items():
        stream = si_extractors.read_spikeglx(recording, stream_id=stream_id)
        channel = stream.get_traces(channel_ids = [channel_id]).squeeze()
        sync_pulse_times[stream_id] = get_trigger_times(channel, stream.get_sampling_frequency(), mode="onset")

    match recording_software: # get NIDQ sync channel
        case "spikeGLX":
            NIDQ_sync_channel_time = sync_pulse_times["nidq"]
        case _:
            raise ValueError(f"Recording software {recording_software} is unknown, please choose from: 'spikeGLX' or add support.")

    triggers_synced = [{} for _ in trigger_times_list]

    for probe_i, stream_id in enumerate(sync_streams): # loop over probes
        print(f"Syncing trigger times to probe {probe_i+1}/{len(sync_streams)}...")

        sync_onsets_probe = sync_pulse_times[stream_id]
        interpolant = interp1d(NIDQ_sync_channel_time, sync_onsets_probe, kind='linear', fill_value="extrapolate") # piecewise linear interpolation

        for i, trigger_times in enumerate(trigger_times_list):
            triggers_synced[i][stream_id] = align_triggers(trigger_times, interpolant)

    return triggers_synced[0] if return_single else triggers_synced


def get_synced_trigger_times(raw_recording: Path, trigger_channels: dict, expected_triggers: dict): # RENAME
    """
    ------ Parameters ------
    raw_recording: Path
    trigger_channels: dict
    expected_triggers: dict
        Dict with expected numbers of triggers for each trigger type
        Key names indicate trigger types and should match key names in trigger_channels dict (which is defined in configurations.py)

    ------ Returns ------
    triggers_final: dict
    Keys are probes that trigger times were synced to, values are dicts with trigger types as keys that contain dataframes with onset & offset times
    -> {"probe_1": {"stimulus_A_triggers": pd.DataFrame with ["onset", "offset"], "stimulus_B_triggers": pd.DataFrame with ["onset", "offset"], ...},
        "probe_2": {"stimulus_A_triggers": pd.DataFrame with ["onset", "offset"], "stimulus_B_triggers": pd.DataFrame with ["onset", "offset"], ...}, ...}
    """
    trigger_onset_times, trigger_offset_times = {}, {}
    triggers_binary, nidq_fs  = extract_triggers_from_analogue(raw_recording, trigger_channels)

    for trigger_type, binary_stream in triggers_binary.items():
        trigger_onset_times[trigger_type] = get_trigger_times(binary_stream, nidq_fs, mode="onset")
        trigger_offset_times[trigger_type] = get_trigger_times(binary_stream, nidq_fs, mode="offset")

        if len(trigger_onset_times[trigger_type]) != len(trigger_offset_times[trigger_type]):
            raise ValueError(f"Amount of trigger onset & offset times do not match for {trigger_type} triggers. Please check extracted trigger times.")
   
    check_trigger_n(trigger_onset_times, expected_triggers)

    # sync trigger times to probes
    trigger_onsets_synced, trigger_offsets_synced = sync_triggers_and_recording(raw_recording, [trigger_onset_times, trigger_offset_times], recording_software="spikeGLX")

    triggers_final = reshape_trigger_times(trigger_onsets_synced, trigger_offsets_synced)

    return triggers_final


def get_curation_labels(analyzer, curation_method: Literal["bombcell", "simple_thresholds", None] = None):
    match curation_method:
        case "bombcell" | "simple_thresholds":
            return analyzer.sorting.get_property(f"{curation_method}_label")
        case None:
            return [None] * len(analyzer.unit_ids)
        case _:
            raise ValueError(f"Curation method {curation_method} not recognized.\nPlease choose from: None, bombcell, simple_thresholds")


def get_spikes_tsgroup(analyzer, curation_labels: Literal["bombcell", "simple_thresholds", None] = None):
    """
    Returns pynapple TsGroup object with unit ids as index, spike times and metadata: depth.

    ------ Parameters ------
    analyzer: spikeinterface SortingAnalyzer object or path to spikeinterface sorting analyzer object
    curation_labels: Literal["bombcell", "simple_thresholds", None] (default: None)
        Curation method whichs labels are included as metadata in the returned TsGroup
        Select None if you don't want to include curation labels in the metadata
    """
    if not isinstance(analyzer, si.SortingAnalyzer):
        analyzer = si.load(analyzer)

    unit_labels = get_curation_labels(analyzer, curation_labels)

    unit_df = pd.DataFrame({"unit_id": analyzer.unit_ids,
                    "depth": analyzer.get_extension("unit_locations").get_data()[:, 1], # 1 -> y coordinate
                    "label": unit_labels,
                    "spike_times": [analyzer.sorting.get_unit_spike_train(unit, return_times=True, segment_index=0) for unit in analyzer.unit_ids]}) # in seconds

    timestamp_dict = {row["unit_id"]: nap.Ts(row["spike_times"]) for _, row in unit_df.iterrows()} # convert data to dict of Ts objects, keys are unit ids (integer convertible)
    spikes_group = nap.TsGroup(data = timestamp_dict, metadata=unit_df.set_index("unit_id")[["depth", "label"]])
    return spikes_group


def normalize_unit(normalization_method: Literal["baseline_z-score", "global_z-score", "baseline_subtraction", "max_rate"], mean_rate, baseline_window = None,
                     reference_rate = None):
    """
    ------ Parameters ------
    normalization_method: str (options: "baseline_z-score", "global_z-score", "baseline_subtraction", "max_rate")
        "baseline_subtraction" -> suptracts average firing rate in baseline window from mean_rate in each bin
        "baseline_z-score" -> uses provided baseline_window to compute mean and std for z-score normalization
        "global_z-score" -> uses entire psth trace to z-score normalize
        "max_rate" -> divides mean rate in each bin by maximum mean rate across all bins
    mean_rate: nap.Tsd
    baseline_window: tuple (start, end)
    """
    normalization_rate = mean_rate if reference_rate is None else reference_rate

    match normalization_method:
        case "baseline_subtraction":
            if baseline_window is None:
                raise ValueError("If you want to normalize using baseline subtraction, normalization_baseline must be provided.")
            
            baseline = normalization_rate[(normalization_rate.index >= baseline_window[0]) & (normalization_rate.index < baseline_window[1])]
            normalized = mean_rate - baseline.mean()
            return nap.Tsd(t=mean_rate.index, d = normalized.values)
        case "baseline_z-score":
            if baseline_window is None:
                raise ValueError("If you want to normalize using baseline z-score, normalization_baseline must be provided.")
            baseline = normalization_rate[(normalization_rate.index >= baseline_window[0]) & (normalization_rate.index < baseline_window[1])]
            baseline_std = baseline.std()
            if baseline_std != 0:
                normalized = (mean_rate - baseline.mean())/baseline_std # z-score normalization
                return nap.Tsd(t=mean_rate.index, d = normalized.values)
            else:
                return None
        case "global_z-score":
            if baseline_window is not None:
                raise ValueError("If you want to normalize using baseline z-score, use normalization_method = 'baseline_z-score'.\n If you want to use the entire trace, set normalization_baseline to None.")
            global_std = normalization_rate.std()
            if global_std != 0:
                normalized = (mean_rate - normalization_rate.mean())/global_std # z-score normalization
                return nap.Tsd(t=mean_rate.index, d = normalized.values)
            else:
                return None
        case "max_rate":
            maximum_rate = normalization_rate.max()
            if maximum_rate != 0:
                normalized = mean_rate / maximum_rate
                return nap.Tsd(t=mean_rate.index, d = normalized.values)
            else:
                return None
        case _:
            raise ValueError(f"Invalid normalization method: {normalization_method}. Choose 'baseline_subtraction', 'baseline_z-score', 'global_z-score', or 'max_rate'.")
 

def get_normalized(data_to_normalize, normalization_method, baseline_window = None, normalization_reference = None):
    """
    Normalizes peristimulus data

    ------ Parameters ------
    data_to_normalize: dict
        Results of get_peristimulus_data()
        Unit ids as keys, contains mean firing rates (spikes/s) per bin (get_peristimulus_data -> output["mean_rates"])
    normalization_method: str
        See normalize_unit() for options
    normalization_reference: dict
        Results of get_peristimulus_data
        Unit ids as keys, contains reference mean firing rates (spikes/s) per bin (get_peristimulus_data -> output["mean_rates"])
    method: str
        See normalize_unit() for options
    """
    if normalization_reference is not None and data_to_normalize["mean_rates"].keys() != normalization_reference["mean_rates"].keys():
        raise ValueError("data_to_normalize and reference_data don't contain the same unit ids.")
        
    normalized_mean_rates = {}
    for unit_id, mean_rate in data_to_normalize["mean_rates"].items():
        reference_rate = None if normalization_reference is None else normalization_reference["mean_rates"][unit_id]
        normalized_rates = normalize_unit(normalization_method, mean_rate, baseline_window, reference_rate)

        if normalized_rates is not None:
            normalized_mean_rates[unit_id] = normalized_rates
        else:
            print(f"Unit {unit_id} baseline standard deviation or maximum firing rate is 0, skipping normalization") # return this message in normalize_unit (depending on normalization_method) and do isinstance check isntead?
            continue

    return normalized_mean_rates


def get_peristimulus_data(spikes, stimulus_onsets, window, bin_size = 0.01, verbose = True,
                          normalization_method: Literal["baseline_z-score", "baseline_subtraction", "max_rate", "global_z-score"] = None,
                          normalization_baseline = None, normalization_reference = None):
    # TODO: switch input order, first stimulus_onsets, then spikes, -> better for groupby_apply
    """
    Calculates trial-relative firing times in window and mean firing rates per bin for given stimulus onsets.
    If a normalization_baseline is provided, normalized mean firing rates (z-scored) are returned as well.
    
    ------ Parameters ------
    spikes: nap.TsGroup
    stimulus_onsets: nap.IntervalSet or nap.Ts
        Onset times of stimulus events to align spikes to
    window: tuple with (min, max)
        Window around stimulus onset to consider for peri-stimulus data
    bin_size: float (default: 0.01)
        Size of time bins for spike counts in seconds
    normalization_baseline: tuple with (min, max) or None (default: None)
        Window used to compute baseline firing rate for normalization, if provided normalized mean rates are returned
    normalization_reference: (default: None)
        If provided, mean rates are normalized to reference data (e.g.: across conditions, from another condition), useful for groupby_apply calls
        Unit ids in normalization_reference must match unit ids in spikes.
        If None, normalization is done within the stimulus_onsets provided.

    ------ Returns ------
    peristimulus_data: dict
        Peri-stimulus data with keys: "times", "mean_rates"
        "times": dict with unit ids as keys, contains spike times in window relative to stimulus onsets
        "mean_rates": dict with unit ids as keys, contains mean firing rates (spikes/s) per bin
        "normalized_mean_rates": dict with unit ids as keys, contains normalized mean firing rates (z-scored) per bin, only returned if normalization_baseline is provided
    """
    if isinstance(stimulus_onsets, nap.IntervalSet):
        stimulus_onsets = nap.Ts(stimulus_onsets.start)

    if verbose:
        print("Computing peristimulus data...")
    perievent_times = nap.compute_perievent(spikes, events=stimulus_onsets, window=window) # -> returns spike times of spikes in window in trial-relative time, {unit_id1: tsgroup of trials with relative onsets, unit_id2: ...}
    perievent_mean_rates = {}
    standard_deviations = {}
    
    for unit_id in spikes.index: # loop over unit ids
        perievent_counts = perievent_times[unit_id].count(bin_size)
        mean_rate = np.mean(perievent_counts / bin_size, axis=1) # divide by bin size to get rate in spikes/s, average across trials
        perievent_mean_rates[unit_id] = mean_rate
        standard_deviations[unit_id] = np.std(perievent_counts / bin_size, axis=1)

    peristimulus_data = {"times": perievent_times, "mean_rates": perievent_mean_rates, "standard_deviations": standard_deviations}

    if normalization_method is not None:
        peristimulus_data["normalized_mean_rates"] = get_normalized(peristimulus_data, normalization_method, normalization_baseline, normalization_reference)

    return peristimulus_data


def average_PSTH(data, units = None):
    # TODO: also change this how data is transformed shouldn't depend on input format
    if isinstance(data, pd.DataFrame): # grand mean data
        combined_df = data
    else: # recording level data
        if units is None:
            units = data.keys()
        combined_df = pd.DataFrame({unit: data[unit].as_series() for unit in units})
    means_per_time_bin = combined_df.mean(axis=1)
    SE_per_time_bin = combined_df.sem(axis=1)
    return means_per_time_bin, SE_per_time_bin


def get_grand_mean_PSTH_df(all_psth_data: dict, average_by = Literal["unit", "recording"], units: dict = None):
    # TODO: this is a bit confusing, perform more computations in this so loop to populate all psth data not necessary
    """
    ------ Parameters ------
    all_psth_data: dict
        Keys are recordings, values are dicts with units as keys and PSTH data (e.g.: mean rates per time bin) as values
    average_by: str ("unit" or "recording")
        "unit" -> average across all units from all recordings (grand mean)
        "recording" -> first average across units within each recording, then average these means across recordings (mean of means, animal is basis of analysis)
    units: dict or None
        Keys are recordings, values are lists of unit ids to include in average
            Key names must match key names in all psth_data dict
        If None all units in psth data are included
    """
    if units is not None and (missing := (all_psth_data.keys() - units.keys())):
        print(f"\033[93mWarning: Recordings in all_psth_data but not in units:\n{"\n".join(missing)}\nAll units of these recordings will be included in the average!\033[0m")

    all_means = []
    for recording, values in all_psth_data.items():
        if units is None or recording not in units:
            included_units = values.keys()
        else:
            included_units = units[recording]

        plot_df = pd.DataFrame({f"{recording}_{unit}": values[unit].as_series() for unit in included_units})

        match average_by:
            case "unit":
                all_means.append(plot_df)
            case "recording":
                all_means.append(plot_df.mean(axis=1).rename(f"{recording}_mean")) # average across units within each recording first
            case _:
                raise ValueError(f"Invalid value for average_by: {average_by}. Must be 'unit' or 'recording'.")
    
    combined_df = pd.concat(all_means, axis=1)

    return combined_df


def get_fq_heatmap_data(spikes, stimulus_epochs, window, bin_size, fq_metadata_column = "stimulus"):
    """
    spikes: nap.TsGroup
    stimulus_epochs: nap.IntervalSet
        Should contain stimulus onsets and metadata column (default: "stimulus") containing frequency information
    window: tuple (min, max)
    bin_size: float
    fq_metadata_column: str (default: "stimulus")
        Name of metadata column in stimulus_epochs containing frequency information (in Hz)
    """
    fqs_played = sorted(stimulus_epochs[fq_metadata_column].unique(), key=float) # sort by float value of fq
    plot_matrix = {unit: [] for unit in spikes.index}
    print("Computing peristimulus data per frequency...")
    for stimulus_fq in fqs_played:
        stimulus_onsets = nap.Ts(stimulus_epochs[stimulus_epochs[fq_metadata_column] == stimulus_fq]["start"])
        peristimulus_data = get_peristimulus_data(spikes, stimulus_onsets, window=window, bin_size=bin_size, verbose = False)

        for unit in spikes.index:
            plot_matrix[unit].append(peristimulus_data["mean_rates"][unit].values)

    frequencies_log = np.log2(sorted([float(fq) for fq in fqs_played]))
    difference_log = np.diff(frequencies_log)
    fq_edges_log = np.concatenate([[frequencies_log[0] - difference_log[0]/2], frequencies_log[:-1] + difference_log/2, [frequencies_log[-1] + difference_log[-1]/2]])
    fq_edges_hz = 2 ** fq_edges_log
    time_edges = np.arange(window[0], window[1] + bin_size, bin_size)

    return plot_matrix, {"fq_edges_hz": fq_edges_hz, "time_edges": time_edges}


def get_fq_population_heatmap_data(spikes, stimulus_epochs, response_window, fq_metadata_column = "stimulus",
                                   sort_by: Literal["best_frequency", "unit_id", "custom_sorting"] = "best_frequency", custom_unit_sorting = None):
    """
    Returns average responses (normalized to maximum rate within each unit) per frequency for each unit.
    ------ Parameters ------
    spikes: nap.TsGroup
    stimulus_epochs: nap.IntervalSet
        Should contain stimulus onsets and metadata column (default: "stimulus") containing frequency information
    window: tuple (min, max)
    bin_size: float
    fq_metadata_column: str (default: "stimulus")
        Name of metadata column in stimulus_epochs containing frequency information (in Hz)
    """
    fqs_played = sorted(stimulus_epochs[fq_metadata_column].unique(), key=float) # sort by float value of fq
    response_per_fq = {uid: [] for uid in spikes.index}

    for fq in fqs_played:
        trial_onsets = nap.Ts(stimulus_epochs[stimulus_epochs[fq_metadata_column] == fq]["start"])
        average_rate = firing_rate_per_trial(spikes, trial_onsets, response_window).mean(axis=0)
        for i, unit_id in enumerate(spikes.index):
            response_per_fq[unit_id].append(average_rate[i])

    for uid in response_per_fq: # normalize via maximum rate
        maximum_rate = np.max(response_per_fq[uid])
        if maximum_rate != 0:
            response_per_fq[uid] = np.array(response_per_fq[uid]) / maximum_rate
        else:
            response_per_fq[uid] = np.zeros(len(response_per_fq[uid]))

    # log-spaced frequency bin edges
    frequencies_log = np.log2([float(fq) for fq in fqs_played])
    diff_log = np.diff(frequencies_log)
    fq_edges_log = np.concatenate([[frequencies_log[0] - diff_log[0] / 2], frequencies_log[:-1] + diff_log / 2, [frequencies_log[-1] + diff_log[-1] / 2]])
    fq_edges_hz = 2 ** fq_edges_log

    match sort_by:
        case "unit_id":
            sorted_ids = spikes.index
        case "best_frequency":
            sorted_ids = sorted(spikes.index, key=lambda uid: np.argmax(response_per_fq[uid]))
        case "custom_sorting":
            if custom_unit_sorting is not None:
                sorted_ids = custom_unit_sorting
            else:
                raise ValueError("custom_unit_sorting must be provided if sort_by='custom_sorting'")
        case _:
            raise ValueError(f"Invalid sort_by: {sort_by}. Allowed: 'best_frequency', 'unit_id', 'custom_sorting'")

    plot_matrix = [np.array(response_per_fq[uid]) if uid in response_per_fq else np.full(len(fqs_played), np.nan) for uid in sorted_ids]

    return plot_matrix, fq_edges_hz, sorted_ids


def firing_rate_per_trial(spikes, trial_onsets, window):
    """
    Get spike rate in fixed window relative to trial onsets.
    
    ------ Parameters ------
    window: tuple (start, end)
        (min, max) of analysis window in seconds (relative to stimulus onset)

    ------ Returns ------
    Firing rates per trial: nap.Tsd
        Rows are trials (from stimulus onsets), columns are unit ids
    """
    if isinstance(trial_onsets, nap.IntervalSet):
        trial_onsets = nap.Ts(trial_onsets.start)

    window_size = window[1] - window[0] # in seconds
    window_epochs = nap.IntervalSet(start = trial_onsets.t + window[0], end = trial_onsets.t + window[1])
    
    counts = spikes.count(ep = window_epochs)
    return counts/window_size


def spike_count_per_trial(spikes, trial_onsets, window):
    """
    Get spike counts in fixed window relative to trial onsets.
    """
    if isinstance(trial_onsets, nap.IntervalSet):
        trial_onsets = nap.Ts(trial_onsets.start)
    
    window_epochs = nap.IntervalSet(start = trial_onsets.t + window[0], end = trial_onsets.t + window[1])
    counts = spikes.count(ep = window_epochs)

    return counts


def get_baseline_trials(trials_of_interest, all_trials, n_preceding): # rename get_relative_trials, then n_relative -> negative or positive? 
    """
    Returns nap.IntervalSet of trials of interest and n_preceding trials
    """
    # TODO: add option to check if some variable matches trial of interest & all preceding trials, add input check_column = None or list or str
        # maybe add metadata column for block and use that in get mean_baseline_firing_rate
    idx = np.searchsorted(all_trials.start, trials_of_interest.start)
    
    if not np.allclose(all_trials.start[idx], trials_of_interest.start):
        raise ValueError("Some trials_of_interest were not found in all_trials.")
    idx = idx[:, None] - np.arange(n_preceding + 1)   # (n_interest, n_preceding+1)

    if np.any(idx < 0):
        raise ValueError(f"Negative indices found, please adjust inputs! n_preceding may be too large.")
    
    idx = np.sort(idx.flatten())

    if len(idx) != len(np.unique(idx)):
        raise ValueError(f"Duplicate indices found.") # TODO: check if this is necessary or if there are cases where this is wanted
    
    return all_trials[idx]


def get_mean_baseline_firing_rate(spikes, stimuli, baseline_window, n_preceding_trials):
    """
    Calculates mean firing rate from each n_preceding_trials consecutive trials in stimuli.
    """
    # TODO: add option to average across all provided baseline trials instead of last n
    # TODO: add option to check some variable first (e.g.: noise level, probably have to do this in get_baseline_trials())
            # however, then in this function there should be way to average all up to & including trial of interest, without fixed reliance on n_preceding_trials
            # (e.g.: if only 2 trials available at fixed background dB, average those 2 only, after continnue with 3, ...)
    
    if missing := [name for name, val in (("stimuli", stimuli), ("n_preceding_trials", n_preceding_trials)) if val is None]:
        raise ValueError(f"{', '.join(missing)} must be provided to compute mean baseline firing rate.")

    if isinstance(stimuli, nap.IntervalSet):
        stimuli = nap.Ts(stimuli.start)

    if len(stimuli) % (n_preceding_trials + 1) != 0:
        raise ValueError(f"Number of stimuli ({len(stimuli)}) is no multiple of n_preceding_trials ({n_preceding_trials}).")

    baseline_rates = [firing_rate_per_trial(spikes, stimuli[j::n_preceding_trials + 1], baseline_window) for j in range(n_preceding_trials)] # one arrray for j-th member of each block
    baseline_mean = np.mean([baseline_rate.values for baseline_rate in baseline_rates], axis=0)

    return nap.TsdFrame(t = baseline_rates[n_preceding_trials - 1].index,
                        d = baseline_mean,
                        columns = baseline_rates[n_preceding_trials - 1].columns) # n-1 -> last member of each block is trial of interest


def wilcoxon_test_responsive(stimulus_onsets, spikes, baseline_window, response_window, alpha = 0.05, alternative="two-sided",
                             fdr_method = "fdr_bh", n_preceding_trials = None, all_trials = None, **kwargs_wilcoxon): # maybe create spikes_group class and add to that?
    """
    Uses Wilcoxon signed-rank test to compare firing rates at baseline (window) and response window

    ------ Parameters ------
    spikes: pynapple.TsGroup
    stimulus_onsets: pynapple.Ts or nap.IntervalSet
    baseline_window: tuple (min, max)
    response_window: tuple (min, max) or dict {label1: (min, max), label2: (min, max), ...}
        Using dict enables testing multiple response windows (e.g.: onset, offset, late) in one function call
        Results will be corrected for multiple comparisons if enabled
    alpha: float (default: 0.05)
    alternative: str (default: "two-sided")
        Significance level for multiple comparisons correction and classification
    fdr_method: str (default: "fdr_bh")
        Method for FDR correction, default Benjamini-Hochberg ("fdr_bh"), for other options see statsmodels.stats.multitest.multipletests
        None to skip FDR correction
    baseline_preceding_n: int (default: None)
        Number of preceding trials to use for baseline firing rate calculation.
        If n>1, the baseline window is the mean of the last n trials before stimulus onset
    all_trials: pynapple.Ts or nap.IntervalSet (default: None)
        These trials will be used to get baseline trials and calculate the average baseline firing rate.

    ------ Returns ------
    results: pandas.DataFrame
        Indexed by unit_id
        Columns: p_raw (float, uncorrected p-value), p_corrected (float, only returned if fdr_method != None), mean_difference, responsive (bool), direction
    """    
    if n_preceding_trials is None and baseline_trials is None:
        baseline_rates = firing_rate_per_trial(spikes, stimulus_onsets, baseline_window)
    else:
        baseline_trials = get_baseline_trials(stimulus_onsets, all_trials, n_preceding_trials)
        if not np.allclose(baseline_trials[n_preceding_trials::n_preceding_trials + 1].start, stimulus_onsets.start): # is this still needed?
            raise ValueError("Baseline and response onsets are not aligned trial-for-trial.")
        baseline_rates = get_mean_baseline_firing_rate(spikes, baseline_trials, baseline_window, n_preceding_trials)
    
    response_rates = firing_rate_per_trial(spikes, stimulus_onsets, response_window)

    p_raw = []
    mean_difference = []

    for unit_id in spikes.index:
        baseline_rate = baseline_rates.loc[unit_id].values
        response_rate = response_rates.loc[unit_id].values

        difference = response_rate - baseline_rate
        mean_difference.append(np.mean(difference))

        if np.all(difference == 0):
            p_raw.append(1.0)
        else:     
            _, p_value = wilcoxon(difference, alternative = alternative, **kwargs_wilcoxon) # returns statisitc, p value
            p_raw.append(p_value)

    results = pd.DataFrame({"p_raw": p_raw, "mean_diff": mean_difference}, index=spikes.index)

    if fdr_method is not None: # correct for multiple comparisons
        _, results["p_corrected"], _, _ = multipletests(results["p_raw"], alpha=alpha, method=fdr_method)
        results["significant"] = results["p_corrected"] < alpha
    else:
        results["significant"] = results["p_raw"] < alpha

    # assess direction of significant responses
    results["direction"] = np.where(results["significant"], np.where(results["mean_diff"] > 0, "excitation", "suppression"), None)
        
    return results


def _compute_spike_density_and_baseline_corrected_count(mean_rate, smoothing_seconds, baseline_window, response_window, bin_size):
    """
    Applies Gaussian smoothing to mean firing-rate vector (returned by get_peristimulus_data) and computes baseline-corrected spike count

    ------ Parameters ------
    mean_rate
        Mean firing rate in spikes/s
    smoothing_seconds
        Standard deviation for Gaussian smoothing kernel in s
    baseline_window: tuple (start, end)

    ------ Returns ------
    spike_density_function:
        Smoothed spike-density function (Hz)
    baseline_corrected_spike_count:
        positive-area BCSC (spikes)
    baseline_rate:
        mean Hz over baseline window
    """
    spike_density_function = nap.Tsd(mean_rate.index.values, d=gaussian_filter1d(mean_rate.values, sigma=smoothing_seconds/bin_size))
    baseline_rate = spike_density_function.get(start = baseline_window[0], end=baseline_window[1]).mean()
    response_rate = spike_density_function.get(start = response_window[0], end=response_window[1]) - baseline_rate
    baseline_corrected_spike_count = np.sum(np.maximum(response_rate, 0)) * bin_size # only consider area above baseline (min 0), multiply with bin size to get spike count from rate
    return spike_density_function, baseline_rate, baseline_corrected_spike_count


def _montecarlo(mean_rates, baseline_FR, observed_spike_count, baseline_window, response_window, smoothing_seconds, n_samples, bin_size):
    """
    ------ Parameters ------
    observed_spike_count
        Baseline corrected spike count observed in response window
    """
    n_bins = len(mean_rates)
    baseline_slice = mean_rates.get_slice(start=baseline_window[0], end=baseline_window[1])
    response_slice = mean_rates.get_slice(start=response_window[0], end=response_window[1])

    lambda_bin = baseline_FR * bin_size # expected spike count
    simulated_counts = np.random.poisson(lambda_bin, size=(n_samples, n_bins)) # simulate n_samples PSTHs with baseline spike count
    simulated_spike_density_fct = gaussian_filter1d(simulated_counts / bin_size, sigma=smoothing_seconds/bin_size, axis=1)

    simulated_baseline = simulated_spike_density_fct[:, baseline_slice].mean(axis=1, keepdims=True)
    simulated_response = simulated_spike_density_fct[:, response_slice] - simulated_baseline
    null_spike_count = np.sum(np.maximum(simulated_response, 0.0), axis=1) * bin_size

    null_greater = int((null_spike_count >= observed_spike_count).sum())
    return (null_greater + 1) / (n_samples + 1)


def montecarlo_test_responsive(stimulus_times, spikes, baseline_window, response_window, PSTH_window, smoothing_seconds, PSTH_bin_size = 0.001, n_samples = 1000, alpha = 0.05):
    """
    ------ Parameters ------
    spikes: nap.TsGroup
    stimuli: nap.IntervalSet or nap.Ts
        Stimulus onsets to use as reference points
    PSTH_window: tuple (start, end)
        Time window for PSTH in seconds
    smoothing_seconds: float
        Standard deviation for Gaussian smoothing kernel in seconds
    PSTH_bin_size: float (default 0.001)
        bin_size for PSTH in seconds
    n_samples: int (default 1000)
        Number of Monte Carlo iterations to perform
    alpha: float (default 0.05)
        Significance threshold for responsiveness

    ------ Returns ------
    pd.DataFrame with unit ids as index and columns:
        "p_value": p_value from _montecarlo()
        "significant": boolean flag for responsiveness at given alpha
    """
    if isinstance(stimulus_times, nap.IntervalSet):
        stimulus_times = nap.Ts(stimulus_times.start)
    
    ps_data = get_peristimulus_data(spikes, stimulus_times, window=PSTH_window, bin_size=PSTH_bin_size, verbose=False)

    p_values = []
    for unit_id in spikes.index:
        mean_rate = ps_data["mean_rates"][unit_id] # spikes/s
        _, baseline_FR, corrected_spike_count = _compute_spike_density_and_baseline_corrected_count(mean_rate, smoothing_seconds, baseline_window, response_window, PSTH_bin_size)
        p = _montecarlo(mean_rate, baseline_FR, corrected_spike_count, baseline_window, response_window, smoothing_seconds, n_samples, PSTH_bin_size)
        p_values.append(p)
    
    return pd.DataFrame({"p_value": p_values, "significant": np.array(p_values) < alpha}, index=spikes.index)


def peak_detection_responsive(stimulus_times, spikes, baseline_window, response_window, peak_height_threshold, PSTH_bin_size = 0.01):
    """
    Finds units with peaks > peak_height_threshold * baseline standard deviation + baseline mean in response window PSTH.

    ------ Parameters ------
    spikes: nap.TsGroup
    baseline_window: tuple (start, end)
    response_window: tuple (start, end)
    peak_height_threshold: float
        Number of standard deviations above baseline mean for peak to be considered significant
    PSTH_bin_size: float (default 0.01)
        Bin size for PSTH in seconds
    """
    if isinstance(stimulus_times, nap.IntervalSet):
        stimulus_times = nap.Ts(stimulus_times.start)
    
    baseline_rate = np.mean(firing_rate_per_trial(spikes, stimulus_times, baseline_window), axis = 0)
    baseline_sd = np.std(firing_rate_per_trial(spikes, stimulus_times, baseline_window), axis = 0)
    unit_baselines = pd.DataFrame({"baseline_mean": baseline_rate, "baseline_sd": baseline_sd}, index = pd.Index(spikes.index))

    psth_response_window = get_peristimulus_data(spikes, stimulus_times, response_window, bin_size = PSTH_bin_size, verbose = False, normalization_method = None)
    
    responsive_units = []
    for unit_id in spikes.index:
        peaks, _ = find_peaks(psth_response_window["mean_rates"][unit_id], height = unit_baselines.loc[unit_id, "baseline_mean"] + peak_height_threshold * unit_baselines.loc[unit_id, "baseline_sd"])
        # TODO: add trough detection?
        
        if len(peaks) != 0:
            responsive_units.append(unit_id)
    
    return responsive_units


def classify_responsive_units(stimulus_times, spikes, baseline_window, response_window, method: Literal["wilcoxon", "montecarlo", "peak_detection"], **kwargs):
    """
    ------ Parameters ------
    stimulus_times: nap.IntervalSet or nap.Ts
        Stimulus onsets to base classification on
    spikes: nap.TsGroup
    baseline_window: tuple (start, end)
    response_window: tuple (start, end)

    ------ Returns ------
    unit_ids_responsive: list
    """
    match method:
        case "wilcoxon":
            responsiveness_df = wilcoxon_test_responsive(stimulus_times, spikes, baseline_window=baseline_window, response_window=response_window, **kwargs)
            unit_ids_responsive = responsiveness_df[responsiveness_df["significant"] == True].index.to_list()
        case "montecarlo":
            responsiveness_df = montecarlo_test_responsive(stimulus_times, spikes, baseline_window=baseline_window, response_window=response_window, **kwargs)
            unit_ids_responsive = responsiveness_df[responsiveness_df["significant"] == True].index.to_list()
        case "peak_detection":
            unit_ids_responsive = peak_detection_responsive(stimulus_times, spikes, baseline_window=baseline_window, response_window=response_window, **kwargs)
        case _:
            raise ValueError(f"Method {method} not recognized, please choose from: 'wilcoxon', 'montecarlo' or add new method.")
    
    return unit_ids_responsive


def get_population_heatmap_data(spikes, stimuli, configs, normalization_method, normalization_baseline = None, sort_by: Literal["mean_response", "peak_response", "unit_id", "custom_sorting"] = "unit_id",
                                sort_window = None, verbose = True, custom_unit_sorting = None, normalization_reference = None):
    """
    ------ Parameters ------
    spikes: nap.TsGroup
    stimuli: nap.IntervalSet or nap.Ts
    configs: AnalysisConfigurations
    normalization_baseline: tuple (start, end)
    normalization_reference: PeristimulusData
        Results of get_peristimulus_data for reference condition
    
    ------ Returns ------
    plot_matrix
        Rows are units (sorted by sort_by), values are mean firing rates in each bin
    time_edges: np.ndarray 
    sorted_ids: list
    """
    if (sort_window is not None) and (len(sort_window) != 2):
        raise ValueError("sort_window must be a tuple with (start, end)")
    if isinstance(stimuli, nap.IntervalSet):
        stimuli = nap.Ts(stimuli.start)

    if verbose:
        print("Computing heatmap data...")
    peristimulus_data = get_peristimulus_data(spikes, stimuli, window=configs.window, bin_size=configs.bin_size,
                                              normalization_method = normalization_method, normalization_baseline = normalization_baseline,
                                              normalization_reference = normalization_reference, verbose = False)

    detection_start, detection_end = (0, configs.window[1]) if sort_window is None else sort_window
    post_window = nap.IntervalSet(start = detection_start, end = detection_end)

    match sort_by:
        case "unit_id":
            sorted_ids = spikes.index
        case "mean_response":
            sorted_ids = sorted(peristimulus_data["normalized_mean_rates"], key=lambda uid: peristimulus_data["normalized_mean_rates"][uid].restrict(post_window).values.mean(), reverse=True)
        case "peak_response":
            sorted_ids = sorted(peristimulus_data["normalized_mean_rates"], key=lambda uid: peristimulus_data["normalized_mean_rates"][uid].restrict(post_window).values.max(), reverse=True)
        case "custom_sorting":
            if custom_unit_sorting is not None:
                sorted_ids = custom_unit_sorting
            else:
                raise ValueError("custom_unit_sorting containing sorted list of unit ids must be provided if sort_by = 'custom_sorting'")
        case _:
            raise ValueError(f"Invalid sort_by value: {sort_by}. Allowed values: 'mean_response', 'peak_response', 'unit_id', 'custom_sorting'.")

    time_edges = np.arange(configs.window[0], configs.window[1] + configs.bin_size, configs.bin_size)
    plot_matrix = [peristimulus_data["normalized_mean_rates"][uid].values if uid in peristimulus_data["normalized_mean_rates"] else np.full((len(time_edges) - 1), np.nan) 
                   for uid in sorted_ids] # all units in sorted_ids but not in "normalized_mean_rates" -> np.nan, white in plot (for case "unit_id", "custom_sorting")
 
    return plot_matrix, time_edges, sorted_ids


def calculate_firing_rate_index(spikes, response_window, stimulus_a, stimulus_b, formula = lambda a, b: (a - b) / (a + b), result_column_name = "result"):
    """
    Takes 2 sets of stimulus onsets, calculates mean firing rate in response window for each stimulus type, then applies formula to get index for each unit
        Default formula to calulate index: index = (FR_a - FR_b) / (FR_a + FR_b)
    
    ------ Parameters ------
    spikes: TsGroup
    response_window: tuple (start, end)
        In seconds relative to stimulus onset
    stimulus_a, stimulus_b: nap.Ts or IntervalSet
        Times of stimulus presentations for stimulus_a, stimulus_b
    formula: function (default: lambda a, b: (a - b) / (a + b))
        Function takes 2 arrays as inputs (mean firing rate stimulus_a, mean firing rate stimulus_b)

    ------ Returns ------
    pd.DataFrame with index unit_id and column result_column_name containing calculated index for each unit
    """
    if isinstance(stimulus_a, nap.IntervalSet): stimulus_a = nap.Ts(stimulus_a.start)
    if isinstance(stimulus_b, nap.IntervalSet): stimulus_b = nap.Ts(stimulus_b.start)
    
    a_mean = firing_rate_per_trial(spikes, stimulus_a, response_window).mean(axis=0)
    b_mean = firing_rate_per_trial(spikes, stimulus_b, response_window).mean(axis=0)

    return pd.DataFrame({result_column_name: formula(a_mean, b_mean)}, index=pd.Index(spikes.index, name="unit_id"))


def compare_nominal_data(all_unit_ids, units_a, units_b, method: Literal["mcnemar"], **kwargs):
    """
    Compare two sets of nominal data (e.g.: responsive vs. non-responsive units) using specified method.
    
    ------ Parameters ------
    all_unit_ids
    units_a
        Units in condition a
    units_b
        Units in condition b
    method: str
        "mcnemar" -> for paired nominal data
            If (counts in a + counts in b) < 25, exact test will be used, otherwise chi-squared approximation. Can be overridden by providing "exact" in kwargs.
        "chi2" -> for unpaired nominal data & large enough sample sizes
        "fishers_exact_test" -> for unpaired nominal data & small sample sizes (n<5 per cell)

    ------ Returns ------
    test_statistic
    p_value
    """

    units_a_bool = np.isin(all_unit_ids, units_a)
    units_b_bool = np.isin(all_unit_ids, units_b)

    match method:
        case "mcnemar":
            counts_a = np.sum(units_a_bool & units_b_bool)  # counts in both
            counts_b = np.sum(units_a_bool & ~units_b_bool) # counts in a only
            counts_c = np.sum(~units_a_bool & units_b_bool) # counts in b only
            counts_d = np.sum(~units_a_bool & ~units_b_bool) # counts in neither
            mc_nemar_table = np.array([[counts_a, counts_b], [counts_c, counts_d]]) # square contingency table
            exact = kwargs.pop("exact", (counts_b + counts_c) < 25) # if exact not provided in kwargs and b + c < 25 use exact test (assumptions mc_nemar)
            result = mcnemar(mc_nemar_table, exact = exact, **kwargs)
            return result.statistic, result.pvalue
        
        case "chi2":
            print("Chi-squared test not implemented yet.")
        
        case "fishers_exact_test":
            print("Fisher's exact test not implemented yet.")

        case _:
            raise ValueError(f"Unsupported method: {method}. Available mehtods are: 'mcnemar'")
        

def GLM_likelihood_ratio_test(full_result, reduced_result):
    """
    Likelihood ratio test (Wilks, 1938) for generalized linear models
    D = -2*(l_reduced - l_full) ~ chi2k
    """
    lr_df = (reduced_result.df_resid - full_result.df_resid)
    lr_stat = -2*(reduced_result.llf - full_result.llf)
    lr_p_value = chi2.sf(lr_stat, lr_df)

    return lr_stat, lr_p_value, lr_df


def get_response_scatterplot_data(spikes, x_events, y_events, response_metric: Literal["mean_rate", "peak_rate"], response_window, bin_size = None):
    """
    """
    if isinstance(x_events, nap.IntervalSet):
        x_events = nap.Ts(x_events.start)
    if isinstance(y_events, nap.IntervalSet):
        y_events = nap.Ts(y_events.start)

    match response_metric:
        case "mean_rate":
            x_vals = firing_rate_per_trial(spikes, x_events, response_window).mean(axis=0)
            y_vals = firing_rate_per_trial(spikes, y_events, response_window).mean(axis=0)
        case "peak_rate":
            x_ps_data = get_peristimulus_data(spikes, x_events, window = response_window, bin_size = bin_size, verbose = False)
            y_ps_data = get_peristimulus_data(spikes, y_events, window = response_window, bin_size = bin_size, verbose = False)
            x_vals = [mean_rate.max() for mean_rate in x_ps_data["mean_rates"].values()]
            y_vals = [mean_rate.max() for mean_rate in y_ps_data["mean_rates"].values()]
        case _:
            raise ValueError(f"Response metric {response_metric} not implemented. Please choose from 'mean_rate' or 'peak_rate'.")
        
    results_df = pd.DataFrame({"x": x_vals, "y": y_vals}, index = spikes.index) # unit_ids as index
    return results_df.join(spikes.metadata)



## classes ##
@dataclass
class AnalysisConfigurations: #TODO: add other configurations that are reusable and not too function-specific, maybe enable adding optional ones
    window: tuple
    bin_size: float

    def __post_init__(self):
        if len(self.window) != 2:
            raise ValueError("window must be a tuple with (min, max)")
          

@dataclass
class PeristimulusData: # TODO: use class instead of dict, replace everywhere in code
    times: dict
    mean_rates: dict[int, nap.Tsd] # unit ids as keys, nap.Tsd with mean_rates per bin as values
    standard_deviation: dict
    normalized_mean_rates = None

# TODO: add normalization configs class? (similar to configs in project specific functions?)