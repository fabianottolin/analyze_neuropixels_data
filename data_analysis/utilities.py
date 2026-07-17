### utilities ###

import spikeinterface.extractors as si_extractors
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
from typing import Literal
from scipy.io import loadmat
from pprint import pformat
from collections import defaultdict
import re
import pickle
import json


def check_streams(recording_path, stream_name = Literal[".ap", ".ap-SYNC"], verbose = False):
    stream_names, _ = si_extractors.get_neo_streams("spikeglx", recording_path)
    streams = [stream for stream in stream_names if stream.endswith(stream_name)]
    if verbose:
        print(f"{len(streams)} stream(s) found")
    return streams


def load_matlab_data(file: Path, mat_variable_name: str, column_names: list = None):
    matlab_data = loadmat(file, struct_as_record=False, squeeze_me=True)
    matlab_data = matlab_data.get(mat_variable_name, {})

    mat_dict = mat_to_dict(matlab_data)
    mat_df = pd.DataFrame(mat_dict)
    
    if column_names is not None:
        mat_df.columns = column_names

    return mat_df


def mat_to_dict(matlab_data): ## convert .mat to nested dictionary
    if isinstance(matlab_data, np.ndarray):
        if matlab_data.dtype != object:  # numeric array
            return matlab_data
        if matlab_data.size == 1:
            return mat_to_dict(matlab_data.item())  # Extract single-value arrays
        else:
            return [mat_to_dict(item) for item in matlab_data]  # Convert lists of structs
    elif isinstance(matlab_data, dict):  # Base case for regular dictionaries
        return {key: mat_to_dict(val) for key, val in matlab_data.items() if not key.startswith("__")}
    elif hasattr(matlab_data, "_fieldnames"):  # MATLAB struct case
        return {field: mat_to_dict(getattr(matlab_data, field)) for field in matlab_data._fieldnames}
    else:
        return matlab_data  # Base case (numeric, string, etc.)


def get_expected_triggers(parameters, parameter_names, ignore_key_errors = False, triggers_if_not_found = 0, print_information = True):
    if isinstance(parameters, dict):
        key = next((key_name for key_name in parameter_names if key_name in parameters), None)
        if key is None:
            if not ignore_key_errors:
                print(pformat(parameters))
                raise KeyError(f"None of {parameter_names} were found in parameters.\n  Available parameters shown above.")
            elif print_information:
                print(f"Key(s) {parameter_names} not found, assigned {triggers_if_not_found} triggers. Available keys: {parameters.keys()}")
            return triggers_if_not_found
        else:
            trigger_sequence = parameters[key]
            if trigger_sequence.ndim == 1:
                trigger_n = np.count_nonzero(trigger_sequence)
            else:
                trigger_n = np.any(trigger_sequence != 0, axis=1).sum()    
            return trigger_n
    else:
        if print_information:
            print(f"\033[33mNo parameters dictionary found, assigning {triggers_if_not_found} trigger(s).\033[0m")
        return triggers_if_not_found # return specified value for protocols without parameters dict
    

def check_trigger_n(trigger_onset_times, expected_triggers):
    """
    Checks if number of extracted triggers matches number of expected triggers.

    ------ Parameters ------
    trigger_onset_times: dict
        Keys are trigger types, values array of trigger times in seconds.
    expected_triggers: dict
        Keys are trigger types,values are the expected number of triggers.
            The keys should match keys in trigger_channels dict in configuration file.
    """
    for trigger_type, onsets in trigger_onset_times.items():
        n_triggers = len(onsets)
        try:
            expected_n = expected_triggers[trigger_type]
        except:
            raise KeyError(f"Trigger type {trigger_type} not found in expected_triggers, please make sure keys in trigger channels (see configuration file) and expected triggers dict match.")
        if n_triggers != expected_n:
            raise ValueError(f"Number of {trigger_type} triggers ({n_triggers}) does not match amount of expected triggers ({expected_n})!")
        else:
            print(f"Number of {trigger_type} triggers matches expected amount ({int(expected_n)})")


def reshape_trigger_times(trigger_onsets_synced, trigger_offsets_synced):
    """
    Reformats separate trigger onset dicts into one dict with dataframes containing onset & offset times of each trigger type.

    ------ Parameters ------
    trigger_onsets_synced: dict
        Keys are probes that trigger times were synced to, contain dicts with trigger types as keys storing arrays with trigger onset times
        -> {"probe": {"stimulus_A_triggers": array([onset_1, onset_2, ...]), "stimulus_B_triggers": array([...]), ...}, ...}
    trigger_offsets_synced: dict
        Keys are probes that trigger times were synced to, contain dicts with trigger types as keys storing arrays with trigger offset times
        -> {"probe": {"stimulus_A_triggers": array([offset_1, offset_2, ...]), "stimulus_B_triggers": array([...]), ...}, ...}
    
    ------ Returns ------
    trigger_times_synced: dict
        Keys are probes that trigger times were synced to, values are dicts with trigger types as keys that contain dataframes with onset & offset times
        -> {"probe_1": {"stimulus_A_triggers": pd.DataFrame with ["onset", "offset"], "stimulus_B_triggers": pd.DataFrame with ["onset", "offset"], ...},
            "probe_2": {"stimulus_A_triggers": pd.DataFrame with ["onset", "offset"], "stimulus_B_triggers": pd.DataFrame with ["onset", "offset"], ...}, ...}
    """
    trigger_times_synced = {}
    for stream_id in trigger_onsets_synced:
        trigger_times_synced[stream_id] = {trigger_type: pd.DataFrame({
            "onset":  trigger_onsets_synced[stream_id][trigger_type],
            "offset": trigger_offsets_synced[stream_id][trigger_type]}) for trigger_type in trigger_onsets_synced[stream_id]}
    return trigger_times_synced


def create_recording_path_list(recordings_folder, recordings_to_process): # checks if recordings to process exist in recordings folder and creates new list with final file paths
    recording_path_list = []
    for recording in recordings_to_process:
        recording_path = recordings_folder / recording
        if recording_path.exists():
            recording_path_list.append(recording_path)
        else:
            print(f"Warning: recording {recording} not found in {recordings_folder}, skipping.")
    return recording_path_list


def check_folder_structure(parent_folder: Path, folder_structure: list): # checks if folder structure exists in parent folder and creates it if not, input is Path() object
    if all((parent_folder / folder).exists() for folder in folder_structure):
        print("Folder structure exists as expected.")
    else:
        for folder in folder_structure:
            (parent_folder / folder).mkdir(parents=True, exist_ok=True)
        print("Folder structure created.")


def get_probe_name(probe_id, recording_software="spikeGLX"):
    match recording_software:
        case "spikeGLX":
            probe_name = re.search(r"imec\d+", probe_id).group()
        case _:
            raise ValueError(f"Recording software {recording_software} is unknown, please choose from: 'spikeGLX' or add support.")
    return probe_name


def save_pkl(data, output_path: Path, filename: str):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    with open(output_path / filename, "wb") as file: pickle.dump(data, file)


def load_pkl(file_path):
    file_path = str(file_path)
    if not file_path.endswith(".pkl"):
        file_path += ".pkl"
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_json(data, output_path: Path, filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(output_path / filename, "w", encoding="utf-8") as file: json.dump(data, file, indent = 4)


def load_json(file_path):
    file_path = str(file_path)
    if not file_path.endswith(".json"):
        file_path += ".json"
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    

def pad_intervalset(intervalset, pre_period=0, post_period=0):
    """
    ------ Parameters ------
    intervalset: nap.IntervalSet
        pre_period/post_period will be subtracted/added to start/end of each item in intervalset
    pre_period: float (default = 0)
        Time in seconds to subtract from start of each interval
    post_period: float (default = 0)
        Time in seconds to add to end of each interval
    """
    new_start = [max(0, start_time) for start_time in intervalset["start"] - pre_period]
    new_end = intervalset["end"] + post_period
    return nap.IntervalSet(new_start, new_end, metadata=intervalset.metadata)


def format_p_value(p_value, format = "APA"):
    if not 0 <= p_value <= 1:
        raise ValueError(f"p-value must be between 0 and 1.")
    
    match format:
        case "APA":
            if p_value < 0.001:
                return "$p$ < .001"
            else:
                return f"$p$ = {p_value:.3f}".replace("0.", ".")
        case _:
            raise ValueError(f"Format {format} not recognized. Please choose from: 'APA' or add support for other formats.")
        

def save_results_dict(dict_to_save, savepath, filename, overwrite_existing_data = False):
    if (savepath / filename).exists():
        existing_dict = load_json(savepath / filename)
        if overwrite_existing_data:
            existing_dict.update(dict_to_save) 
        else:
            for key, value in dict_to_save.items():
                existing_dict.setdefault(key, value) # only adds values not already in existing dict
        dict_to_save = existing_dict

    save_json(dict_to_save, savepath, filename)
    print(f"Dictionary saved under {savepath / filename}.")
    return dict_to_save


def nested_defaultdict(depth, inner_type = dict):
    """
    ------ Parameters ------
    depth: int
        Depth of the nested defaultdict, e.g.: 2 -> level 1: defaultdict, level 2: defaultdict of inner_type
    inner_type: type (default: dict)
        Type of innermost level of nested dict
        E.g.: dict, list, int
    """
    if depth == 1:
        return defaultdict(inner_type)
    return defaultdict(lambda: nested_defaultdict(depth - 1, inner_type))