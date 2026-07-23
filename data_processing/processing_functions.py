## custom processing functions

import spikeinterface as si
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.sorters as si_sorters
import spikeinterface.curation as si_curation
from kilosort.run_kilosort import close_logger
import numpy as np
from pprint import pformat
from typing import Literal
import gc
import copy
from utility_functions import *
from visualization_functions import *
from manual_outside_channel_detector import manual_outside_channel_detector


def lfp_outside_channel_detection(recording, plot = False, output = None, method = "ibl", **kwargs):
    """
    Detects outside channels based on LFP signal and removes them from recording.
    """
    from ibldsp.voltage import detect_bad_channels, spikeglx

    # read LFP binary file
    path_lfp_binary = get_lfp_path(recording)
    lf_recording = spikeglx.Reader(path_lfp_binary)

    match method:
        case "ibl":
            channel_labels, computed_features = detect_outside_channels_batched(lf_recording, detection_function = detect_bad_channels, **kwargs)
    
    outside_indices = np.where(channel_labels == 3)[0]
    outside_channel_ids = recording.get_channel_ids()[outside_indices]
    recording_without_outside_channels = recording.remove_channels(outside_channel_ids) 

    print(f"Removed {len(outside_channel_ids)} outside channels based on LFP band.")
    print(f"{recording_without_outside_channels}\n")

    if plot:
        outside_boundary =  outside_indices[0] if len(outside_indices) > 0 else None
        fig, ax = show_outside_channels(lf_recording, outside_boundary, computed_features, detection_method = method)
        plt.show()
        (output.figures_folder/"LFP_outside_channel_detection").mkdir(parents=True, exist_ok=True)
        fig.savefig(output.figures_folder/"LFP_outside_channel_detection"/f"{output.recording_identifier}_LFP_outside_channel_detection.png", dpi=300)
        print(f"Figure saved under '{output.figures_folder/output.recording_identifier}_LFP_outside_channel_detection.png'")

    return recording_without_outside_channels


def remove_manually_selected_channels(recording, plot = True, output = None):
    """
    Removes outside channels detected based on LFP
    """
    filepath = output.final_output_folder/"manually_selected_channels.json"

    if not filepath.exists():
        raise FileNotFoundError(f"No file with manually selected channels found under {filepath}. Please do manual selection first or change processing parameters!")
    
    manual_selection = load_json(filepath)
    outside_boundary_probe = manual_selection.get(output.recording_identifier, "missing")

    if outside_boundary_probe == "missing":
        raise ValueError(f"No manually selected channels found for {output.recording_identifier} in {filepath}. Please do manual selection first or change processing parameters!")
        
    outside_channel_ids = recording.get_channel_ids()[outside_boundary_probe:] # get_channel_ids() -> starts with ap0
    recording_without_outside_channels = recording.remove_channels(outside_channel_ids)

    if plot:
        from ibldsp.voltage import spikeglx
        path_lfp_binary = get_lfp_path(recording)
        lf_recording = spikeglx.Reader(path_lfp_binary)
        fig, ax = show_outside_channels(lf_recording, outside_boundary = outside_boundary_probe, xfeats = None, detection_method = "manual")
        plt.show()
        (output.figures_folder/"LFP_outside_channel_detection").mkdir(parents=True, exist_ok=True)
        fig.savefig(output.figures_folder/"LFP_outside_channel_detection"/f"{output.recording_identifier}_LFP_outside_channel_detection.png", dpi=300)
        print(f"Figure saved under '{output.figures_folder/output.recording_identifier}_LFP_outside_channel_detection.png'")

    return recording_without_outside_channels


def bad_channel_correction(recording, **kwargs): # detects bad channels based on provided kwargs, removes channels outside brain & interpolates bad channels inside brain, prints which channels were removed and reason for removal
    bad_channel_ids, channel_labels = si_preprocessing.detect_bad_channels(recording, **kwargs)

    # get labels for bad channels only
    all_channel_ids = recording.get_channel_ids()
    bad_channel_labels = channel_labels[np.isin(all_channel_ids, bad_channel_ids)]

    outside_channel_ids = [channel_id for channel_id, label in zip(bad_channel_ids, bad_channel_labels) if label == "out"]
    recording_without_outside_channels = recording.remove_channels(outside_channel_ids) # remove channels outside brain
    print(f"Removed {len(outside_channel_ids)} outside channels based on AP band.")

    other_bad_channel_ids = list(set(bad_channel_ids) - set(outside_channel_ids)) # remaining bad channels inside brain -> interpolate
    recording_corrected_channels = si_preprocessing.interpolate_bad_channels(recording_without_outside_channels, other_bad_channel_ids)
    print(f"Interpolated {len(other_bad_channel_ids)} bad channels:")
    for ch, label in zip(bad_channel_ids, bad_channel_labels):
        if ch in other_bad_channel_ids:
            print(f"    Channel {ch}: {label}")
    return recording_corrected_channels


def detect_and_correct_drift(recording, method, **kwargs): # motion estimation
    print(f"Correcting drift using {method}...")
    motion = si_preprocessing.compute_motion(recording, preset=method, **kwargs)
    recording_motion_corrected = si.sortingcomponents.motion.interpolate_motion(recording, motion)
    return recording_motion_corrected


def continue_si_pipeline(recording, **kwargs): # uses spike interface to continue with specified preprocessing steps
    if kwargs:
        recording = si_preprocessing.apply_preprocessing_pipeline(recording, kwargs)
    return recording


def print_parameter_information(step, step_parameters, spike_sorter=None, curation_method=None): # move to utilities?
    match step:
        case "preprocessing":
            print(f"Chosen spike sorter: {spike_sorter}\n")
            print(f"{' preprocessing parameters '.center(60, '-')}\n{pformat(step_parameters, sort_dicts=False)}\n") # can also use si_preprocessing.PreprocessingPipeline(preprocessing_parameters[spike_interface]) to print parameters for spike interface part
            print(f"{"-"*60}\n")
            
        case "spike_sorting":
            print(f"Chosen spike sorter: {spike_sorter}\n")
            print(f"{' spike sorting parameters '.center(60, '-')}\n{pformat(step_parameters, sort_dicts=False)}")
            print(f"Defaults for {spike_sorter}: {(si_sorters.get_default_sorter_params(spike_sorter))}")
            print(f"{"-"*60}\n")

        case "postprocessing":
            print(f"{' extensions to be computed '.center(60, '-')}\n{pformat(step_parameters, sort_dicts=False)}")
            print(f"{"-"*60}\n")
        
        case "curation":
            print(f"Chosen curation method: {curation_method}\n")
            print(f"{' curation thresholds '.center(60, '-')}\n{pformat(step_parameters, sort_dicts=False)}")
            print(f"{"-"*60}\n")


def pre_preprocessing(recording, custom_preprocessing_parameters):
    custom_preprocessing_parameters = copy.deepcopy(custom_preprocessing_parameters) # prevents overwriting original dict
    saturation_parameters = custom_preprocessing_parameters.pop("detect_saturation_periods", None)
    if saturation_parameters is not None:
        print("Detecting saturation periods...")
        saturation_periods = si_preprocessing.detect_saturation_periods(recording, **saturation_parameters)
            
        for step_name, step_values in custom_preprocessing_parameters.items():
            if step_name.startswith("spike_interface") and ("silence_periods" in step_values): # silence periods is spikeinterface preprocessing step, so will be nested in spike_interface step
                step_values["silence_periods"]["periods"] = saturation_periods
                return custom_preprocessing_parameters
            
        raise ValueError("Pipeline detects saturation periods but doesn't silence them." \
                         "Either remove 'detect_saturation_periods' from preprocessing parameters or add 'silence_periods' to handle detected saturation.")

    return custom_preprocessing_parameters


def apply_preprocessing(recording, custom_preprocessing_parameters, output_information): # move to NeuropixelsData?
    CUSTOM_PREPROCESSING_FUNCTION_MAP = {"correct_bad_channels": bad_channel_correction,
                                        "detect_and_correct_drift": detect_and_correct_drift,
                                        "lfp_outside_channel_detection": lfp_outside_channel_detection,
                                        "remove_manually_selected_channels": remove_manually_selected_channels} # custom processing function names should not start with "spike_interface"!
    
    STEPS_WITH_OUTPUT = ["lfp_outside_channel_detection", "remove_manually_selected_channels"] # specify which steps have plots or need extra information

    for step_name, step_parameters in custom_preprocessing_parameters.items():
        if step_name.startswith("spike_interface"):
            processing_function = continue_si_pipeline # spikeinterface preprocessing functions 
        else:
            processing_function = CUSTOM_PREPROCESSING_FUNCTION_MAP[step_name]
        
        if step_name in STEPS_WITH_OUTPUT:
            step_parameters["output"] = output_information
        
        recording = processing_function(recording, **step_parameters)

    return recording



### classes ###
class NeuropixelsData:
    def __init__(self, recordings_folder, recordings_to_process, spike_sorter, local_output_folder, final_output_folder):
        self.recordings_to_process = create_recording_path_list(recordings_folder, recordings_to_process) # create list of file paths to process
        self.spike_sorter = spike_sorter
        self.local_output_folder = local_output_folder
        self.final_output_folder = final_output_folder

        # set parallel processing variables
        si.set_global_job_kwargs(n_jobs=determine_optimal_n_jobs(), chunk_duration = "1s", progress_bar = True)


    def _preprocess_probe(self, recording, probe, output_folder, preprocessing_parameters, overwrite_existing_files = False):
        preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing)   # check if there is already data for current recording/probe
        spike_sorting_exists = is_folder_with_files(output_folder.sorting_local) or is_folder_with_files(output_folder.sorting_final) # check if there is already spike sorting output for current recording/probe

        run_preprocessing = overwrite_existing_files or (not preprocessing_file_exists and not spike_sorting_exists) # only run pre-processing if no file yet or user wants to overwrite existing files

        if run_preprocessing:
            raw_recording = si_extractors.read_spikeglx(recording, stream_id=probe) # load recording                
            print(raw_recording)
            
            preprocessing_parameters = pre_preprocessing(raw_recording, preprocessing_parameters)
            preprocessed_recording = apply_preprocessing(raw_recording, preprocessing_parameters, output_folder)

            # save pre-processed recording as binary file on local drive; needed for kilosort, for other spike sorters may be faster to continue working with recording in memory
            preprocessed_recording.save(folder = output_folder.preprocessing, format="binary", overwrite = overwrite_existing_files) # assign saved_preprocessed_recording if you want to continue directly afterwards

        else:
            print(f"\033[33mWarning: Already processed probe before, skipping\033[0m\n")


    def run_preprocessing(self, preprocessing_configurations, overwrite_existing_files = False):
        """
        Runs preprocessing for each probe in NeuropixelsData.recordings_to_process according to specified parameters in preprocessing_configurations (defined in processing_parameters.py)

        ------ Parameters ------
        preprocessing_configurations: dict
        overwrite_existing_files: bool (default: False)
            If True, existing preprocessed recording is overwritten by new computations 
        """
        # load corresponding parameters (defined in processing parameters.py)
        preprocessing_parameters = get_parameters_for_sorter(self.spike_sorter, preprocessing_configurations) # load corresponding parameters (defined in processing parameters.py)

        check_folder_structure(self.local_output_folder, ["recordings_preprocessed"])
        check_folder_structure(self.final_output_folder, ["recordings_preprocessed"])

        print_parameter_information("preprocessing", preprocessing_parameters, spike_sorter=self.spike_sorter)

        total_size_recordings = get_size_of_folders(self.recordings_to_process) # gets total size of all recordings to process in GB
        required_GB = total_size_recordings * 2  # estimate of required space to run spike sorting ~2*recording size
        check_free_space(self.local_output_folder, required_GB) # check if there is enough free space

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)
            
                self._preprocess_probe(recording_path, probe_stream, output_folder, preprocessing_parameters, overwrite_existing_files = overwrite_existing_files)


    def _handle_preprocessed_recording(self, output_folder, action: Literal["copy", "delete", "keep"] = "copy"):
        match action:
            case "copy":
                preprocessed_relative_path = output_folder.preprocessing.relative_to(self.local_output_folder)
                copy_data(output_folder.preprocessing, self.final_output_folder/preprocessed_relative_path)
            case "delete":
                shutil.rmtree(output_folder.preprocessing) # deletes preprocessing data
                print("Deleted preprocessed recording.")
            case "keep":
                pass


    def _spike_sort_probe(self, output_folder: OutputPaths, spike_sorter, spike_sorting_parameters, overwrite_existing_files = False):
        spike_sorting_exists = is_folder_with_files(output_folder.sorting_local) or is_folder_with_files(output_folder.sorting_final) # check if there is already spike sorting output for current recording/probe
        preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing) # check if data has been preprocessed

        if (overwrite_existing_files or not spike_sorting_exists) and preprocessing_file_exists: # run spike sorting
            saved_preprocessed_recording = si.load(output_folder.preprocessing) # load preprocessed recording
            sorted_spikes = si_sorters.run_sorter(sorter_name=spike_sorter, recording=saved_preprocessed_recording, folder = output_folder.sorting_local, remove_existing_folder = overwrite_existing_files, **spike_sorting_parameters) # set other parameters here if different from default sorter parameters, you can pass a full dict with the parameters
            close_logger() # spike interface isnt properly closing kilosort4 logger, necesarry to close it here so data can be copied
            print(f"{spike_sorter.capitalize()} found {len(sorted_spikes.get_unit_ids())} units for probe.")
            copy_data(output_folder.sorting_local, output_folder.sorting_final) # copy spike sorting to final folder, delete locally saved spike sorting data
        elif not preprocessing_file_exists:
            print(f"\033[33mWarning: No corresponding preprocessing file could be found, skipping\033[0m\n")
        else:
            print(f"\033[33mWarning: Already spike sorted data of probe with {spike_sorter} before, skipping\033[0m\n")


    def run_spike_sorting(self, spike_sorting_configurations, overwrite_existing_files = False):
        """
        Runs spike sorting for each probe in NeuropixelsData.recordings_to_process according to specified parameters in spike_sorting_configurations (defined in processing_parameters.py)

        ------ Parameters ------
        spike_sorting_configurations: dict
        overwrite_existing_files: bool
        """

        spike_sorting_parameters = get_parameters_for_sorter(self.spike_sorter, spike_sorting_configurations) # load corresponding parameters (defined in processing parameters.py)
        
        check_cuda_availability()

        # check folder structure
        check_folder_structure(self.local_output_folder, ["recordings_spike_sorted"])
        check_folder_structure(self.final_output_folder, ["recordings_spike_sorted"])

        print_parameter_information("spike_sorting", spike_sorting_parameters, spike_sorter=self.spike_sorter)

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                self._spike_sort_probe(output_folder, self.spike_sorter, spike_sorting_parameters, overwrite_existing_files)


    def _postprocess_probe(self, output_folder, postprocessing_configurations, handle_preprocessed_recording: Literal["copy", "delete", "keep"] = "copy", 
                           delete_raw_spike_sorting = False, overwrite_existing_files = False):
        preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing) # check if data has been preprocessed
        spike_sorting_exists = is_folder_with_files(output_folder.sorting_final) # check if there is spike sorting output for current recording/probe
        post_processing_exists = is_folder_with_files(output_folder.analyzer_local) or is_folder_with_files(output_folder.analyzer_final) # check if there is already post-processing output for current recording/probe

        data_exists = preprocessing_file_exists and spike_sorting_exists
        run_postprocessing = overwrite_existing_files or (data_exists and not post_processing_exists)
        
        if run_postprocessing and data_exists: # re-create sorting analyzer
            saved_preprocessed_recording = si.load(output_folder.preprocessing) # load preprocessed recording
            sorted_spikes = si_sorters.read_sorter_folder(output_folder.sorting_final) # load spike sorting data
            sorting_analyzer = si.create_sorting_analyzer(sorting=sorted_spikes, recording=saved_preprocessed_recording, format="memory") # sorting analyzer combines recording & sorting object
        elif run_postprocessing and not data_exists: # load sorting analyzer
            print(f"\033[33mLoading existing sorting analyzer\033[0m\n")
            sorting_analyzer = si.load(output_folder.analyzer_final)
        elif not data_exists:
            print(f"\033[31mWarning: No preprocessed data or spike sorting output found for probe, skipping\033[0m\n")
            return
        else:
            print(f"\033[33mWarning: Already post-processed data of probe before, skipping\033[0m\n")
            return

        print("Computing extensions...")
        sorting_analyzer.compute(postprocessing_configurations)
        sorting_analyzer.save_as(folder=output_folder.analyzer_local, format="binary_folder") # save
        
        if (output_folder.curation).exists(): # delete previous curation if it exists
            shutil.rmtree(output_folder.curation)
            # warning
        copy_data(output_folder.analyzer_local, output_folder.analyzer_final)

        if data_exists:
            del sorting_analyzer, sorted_spikes, saved_preprocessed_recording # prevents permission error
            gc.collect()
            self._handle_preprocessed_recording(output_folder, handle_preprocessed_recording) 
            if delete_raw_spike_sorting: shutil.rmtree(output_folder.sorting_final) # deletes raw spike sorting data


    def run_postprocessing(self, postprocessing_configurations, handle_preprocessed_recording: Literal["copy", "delete", "keep"] = "copy", delete_raw_spike_sorting = False, overwrite_existing_files = False):
        """
        Runs postprocessing for each probe in NeuropixelsData.recordings_to_process according to specified parameters in postprocessing_configurations (defined in processing_parameters.py)
        
        ------ Parameters ------
        postprocessing_configurations: dict
        handle_preprocessed_recording: str (default: "copy")
            Decides what is done with preprocessed recording after finishing postprocessing
            Options: "copy", "delete", "keep"
                "copy" -> copies pre-processed recording to final output folder and removes it from local output folder
                "delete" -> deletes pre-processed recording permanently
                "keep" -> pre-processed recording remains in local output folder only
        delete_raw_spike_sorting: bool (default: False)
            If True, deletes raw spike sorting data after creating sorting analyzer
        overwrite_existing_files: bool (default: False)
            If True, existing files are overwritten by current computations
        """

        ## check folder structure for postprocessing
        check_folder_structure(self.local_output_folder, ["sorting_analyzers"])
        check_folder_structure(self.final_output_folder, ["sorting_analyzers"])

        print_parameter_information("postprocessing", postprocessing_configurations)

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
                
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")
                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                self._postprocess_probe(output_folder, postprocessing_configurations, handle_preprocessed_recording, delete_raw_spike_sorting, overwrite_existing_files = overwrite_existing_files) # run postprocessing for current probe


    def _curate_probe(self, output_folder, method, curation_thresholds, plot = True, overwrite_existing_files = False):
        # check if analyzer exists
        if is_folder_with_files(output_folder.analyzer_final):
            sorting_analyzer = si.load(output_folder.analyzer_final)
        else:
            print(f"\033[31mWarning: No corresponding sorting analyzer found for probe, skipping\033[0m\n")
            return
        
        if is_folder_with_files(output_folder.curation) and not overwrite_existing_files:
            print(f"\033[33mWarning: Already curated data of probe before, skipping\033[0m\n")
            return

        match method:
            case "bombcell":
                bombcell_labels = si_curation.bombcell_label_units(sorting_analyzer, thresholds=curation_thresholds, label_non_somatic=True, split_non_somatic_good_mua=True)
                curation_labels = bombcell_labels["bombcell_label"]
                print(curation_labels.value_counts()) # classification result (mua, noise, good, non_soma_mua)
                non_noisy_units = curation_labels != "noise"
            case "simple_thresholds":
                all_metrics = sorting_analyzer.get_metrics_extension_data()
                simple_thresholds_labels = si_curation.threshold_metrics_label_units(all_metrics, thresholds=curation_thresholds, column_name="simple_threshold")
                curation_labels = simple_thresholds_labels["simple_threshold"]
                print(curation_labels.value_counts()) # classification result (good, noise)
                non_noisy_units = curation_labels != "noise"
            case _:
                raise ValueError(f"Curation method {method} not recognized.\n   Please choose from: bombcell , simple_thresholds")
            
        sorting_analyzer.sorting.set_property(f'{method}_label', curation_labels)
            
        if plot:
            curation_plot(sorting_analyzer, method, curation_labels, output_folder)

        sorting_analyzer_curated = sorting_analyzer.select_units(sorting_analyzer.unit_ids[non_noisy_units]) # remove noisy units
        sorting_analyzer_curated.save_as(folder=output_folder.curation, format="binary_folder")
        print(f"Curated sorting analyzer saved under {output_folder.curation}.\n")


    def automatic_curation(self, method, curation_thresholds, plot = True, overwrite_existing_files = False):
        """
        Runs automatic curation for each probe in NeuropixelsData.recordings_to_process according to specified method and parameters in curation_thresholds (defined in processing_parameters.py)

        ------ Parameters ------
        method: str
            Specifies method used for automatic curation
            Options: "bombcell", "simple_thresholds"
        curation_thresholds: dict
        """

        if method == "bombcell":
            curation_thresholds = update_dict(si_curation.bombcell_get_default_thresholds(), curation_thresholds) # modify bombcell defaults based on curation_thresholds
        
        print_parameter_information("curation", curation_thresholds, curation_method=method)

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                self._curate_probe(output_folder, method, curation_thresholds, plot = plot, overwrite_existing_files = overwrite_existing_files)


    def run_processing_pipeline(self, all_configurations: dict, preprocessing = True, spike_sorting = True, postprocessing = True, curation_method = None,
                                handle_preprocessed_recording: Literal["copy", "delete", "keep"] = "copy", delete_raw_spike_sorting = False, overwrite_existing_files = False):
        """
        Runs specified processing steps in order for each probe in NeuropixelsData.recordings_to_process
            Pre-processing -> spike sorting -> post-processing -> curation
        
        ------ Parameters ------
        all_configurations: dict
            Contains parameters for the different processing steps, is defined in processing_parameters.py with keys "preprocessing", "spike_sorting", "postprocessing", "curation_thresholds"
        preprocessing: bool (default: True)
            If True, preprocessing step is run
        spike_sorting: bool (default: True)
            If True, spike sorting step is run
        postprocessing: bool (default: True)
            If True, postprocessing step is run
        curation_method: str (default: None)
            Specifies method used for automatic curation
            Curation is only run if curation_method is specified!
        handle_preprocessed_recording: str (default: "copy")
            Decides what is done with preprocessed recording after finishing processing
            Options: "copy", "delete", "keep"
        delete_raw_spike_sorting: bool (default: False)
            If True, deletes raw spike sorting data after creating sorting analyzer
        overwrite_existing_files: bool (default: False)
            If True, existing files are overwritten by current computations
        """
        
        # def get all parameters? return pre, spikesorting, post, curation
        preprocessing_parameters, spike_sorting_parameters = get_parameters_for_sorter(self.spike_sorter, all_configurations["preprocessing"], all_configurations["spike_sorting"])
        postprocessing_configurations = all_configurations["postprocessing"]
        curation_thresholds = all_configurations["curation"]

        check_folder_structure(self.local_output_folder, ["recordings_preprocessed", "recordings_spike_sorted", "sorting_analyzers"])
        check_folder_structure(self.final_output_folder, ["recordings_preprocessed", "recordings_spike_sorted", "sorting_analyzers", "figures"])

        if handle_preprocessed_recording != "keep": # if delteting or copying preprocessed recording to network drive only enough space for one recording necessary
            average_size_recordings = get_size_of_folders(self.recordings_to_process)/len(self.recordings_to_process)
            required_GB = average_size_recordings * 2  # estimate of required space to run spike sorting ~2*recording size
            check_free_space(self.local_output_folder, required_GB) # check if there is enough free space
        else: 
            total_size_recordings = get_size_of_folders(self.recordings_to_process) # gets total size of all recordings to process in GB
            required_GB = total_size_recordings * 2  # estimate of required space to run spike sorting ~2*recording size
            check_free_space(self.local_output_folder, required_GB) # check if there is enough free space
       
        if preprocessing:
            print_parameter_information("preprocessing", preprocessing_parameters, spike_sorter=self.spike_sorter)
        if spike_sorting:
            check_cuda_availability()
            print_parameter_information("spike_sorting", spike_sorting_parameters, spike_sorter=self.spike_sorter)
        if postprocessing:
            print_parameter_information("postprocessing", postprocessing_configurations)
        
        if curation_method == "bombcell":
            curation_thresholds = update_dict(si_curation.bombcell_get_default_thresholds(), curation_thresholds) # modify bombcell defaults based on curation_thresholds
        if curation_method != None:
            print_parameter_information("curation", curation_thresholds, curation_method=curation_method)
        else:
            print(f"\033[33mWarning: Curation method set to None, skipping curation\033[0m\n")

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                if preprocessing:
                    self._preprocess_probe(recording_path, probe_stream, output_folder, preprocessing_parameters, overwrite_existing_files = overwrite_existing_files)
                if spike_sorting:
                    self._spike_sort_probe(output_folder, self.spike_sorter, spike_sorting_parameters, overwrite_existing_files = overwrite_existing_files)
                if postprocessing:
                    self._postprocess_probe(output_folder, postprocessing_configurations, handle_preprocessed_recording, delete_raw_spike_sorting, overwrite_existing_files) # run postprocessing for current probe
                if curation_method:
                    self._curate_probe(output_folder, curation_method, curation_thresholds, overwrite_existing_files = overwrite_existing_files) # run curation for current probe

                if preprocessing and not postprocessing: # usually handled in self._postprocess_probe:
                    self._handle_preprocessed_recording(output_folder, handle_preprocessed_recording)

    
    def run_manual_outside_channel_detection(self, duration_sample=10, **kwargs):
        return manual_outside_channel_detector(self, duration_sample=duration_sample, **kwargs)