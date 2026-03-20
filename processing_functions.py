## custom processing functions

import spikeinterface as si
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.sorters as si_sorters
import spikeinterface.curation as si_curation
from pprint import pformat
from utility_functions import *

def bad_channel_correction(recording, **kwargs): # detects bad channels based on provided kwargs and removes them from recording, prints which channels were removed and reason for removal
    bad_channel_ids, bad_channel_labels = si_preprocessing.detect_bad_channels(recording, **kwargs)

    outside_channel_ids = [channel_id for channel_id, label in zip(bad_channel_ids, bad_channel_labels) if label == "out"]
    recording_without_outside_channels = recording.remove_channels(outside_channel_ids)
    print(f"Removed {len(outside_channel_ids)} outside channels")

    other_bad_channel_ids = set(bad_channel_ids) - set(outside_channel_ids) # remaining bad channels inside brain -> interpolate
    recording_corrected_channels = si_preprocessing.interpolate_bad_channels(recording_without_outside_channels, other_bad_channel_ids)
    print(f"Interpolated {len(other_bad_channel_ids)} bad channels:")
    for ch, label in zip(bad_channel_ids, bad_channel_labels):
        if ch in other_bad_channel_ids:
            print(f"    Channel {ch}: {label}")
    return recording_corrected_channels


def detect_and_correct_drift(recording, method): # motion estimation
    motion = si_preprocessing.compute_motion(recording, preset=method)
    recording_motion_corrected = si.sortingcomponents.motion.interpolate_motion(recording, motion)
    return recording_motion_corrected


def continue_si_pipeline(recording, **kwargs): # uses spike interface to continue with specified preprocessing steps
    if kwargs:
        recording = si_preprocessing.apply_preprocessing_pipeline(recording, **kwargs)
    return recording


CUSTOM_PREPROCESSING_FUNCTION_MAP = {"correct_bad_channels": bad_channel_correction,
                                     "spike_interface": continue_si_pipeline,
                                     "detect_and_correct_drift": detect_and_correct_drift}


def apply_custom_preprocessing(recording, custom_preprocessing_parameters):
    for step_name, step_parameters in custom_preprocessing_parameters.items():
        processing_function = CUSTOM_PREPROCESSING_FUNCTION_MAP[step_name]
        recording = processing_function(recording, **step_parameters)
    return recording


def preprocess_probe(recording, output_folder, preprocessing_parameters, ignore_existing_files = False):
    preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing) # check if there is already data for current recording/probe
    spike_sorting_exists = is_folder_with_files(output_folder.sorting_local) or is_folder_with_files(output_folder.sorting_final) # check if there is already spike sorting output for current recording/probe
    
    run_preprocessing = ignore_existing_files or (not preprocessing_file_exists and not spike_sorting_exists) # only run pre-processing if no file yet or user wants to ignore existing files

    if run_preprocessing:
        preprocessed_recording = si_preprocessing.apply_preprocessing_pipeline(recording, preprocessing_parameters["spike_interface"]) # first pre-processing steps via spike interface function
        preprocessed_recording = apply_custom_preprocessing(preprocessed_recording, preprocessing_parameters["custom_steps"]) # custom pre-processing steps requiring more control/flexibility

        # save pre-processed recording as binary file on local drive; needed for kilosort, for other spike sorters may be faster to continue working with recording in memory
        preprocessed_recording.save(folder = output_folder.preprocessing, format="binary") # assign saved_preprocessed_recording if you want to continue directly afterwards 
    else:
        print(f"\033[33mWarning: Already processed probe before, skipping\033[0m")


def spike_sort_probe(output_folder: OutputPaths, spike_sorter, spike_sorting_parameters):
    spike_sorting_exists = is_folder_with_files(output_folder.sorting_local) or is_folder_with_files(output_folder.sorting_final) # check if there is already spike sorting output for current recording/probe
    preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing) # check if data has been preprocessed

    if not spike_sorting_exists and preprocessing_file_exists: # run spike sorting
        required_GB =  output_folder.preprocessing.stat().st_size*2/(1024**3) # estimate of required space to run spike sorting ~2*recording size
        check_free_space(output_folder.sorting_raw, required_GB) # check if there is enough free space
        
        saved_preprocessed_recording = si.load(output_folder.preprocessing) # load preprocessed recording
        sorted_spikes = si_sorters.run_sorter(sorter_name=spike_sorter, recording=saved_preprocessed_recording, folder = output_folder.sorting_raw, **spike_sorting_parameters) # set other parameters here if different from default sorter parameters, you can pass a full dict with the parameters
        print(f"{spike_sorter.capitalize()} found {len(sorted_spikes.get_unit_ids)} units for probe.")
        sorted_spikes.save(folder=output_folder.sorting_local, format="binary_folder")
    elif not preprocessing_file_exists:
        print(f"\033[33mWarning: No corresponding preprocessing file could be found, skipping\033[0m")
    else:
        print(f"\033[33mWarning: Already spike sorted data of probe with {spike_sorter} before, skipping\033[0m")


def postprocess_probe(output_folder, postprocessing_configurations, overwrite_existing_files = False):
    preprocessing_file_exists = is_folder_with_files(output_folder.preprocessing) # check if data has been preprocessed
    spike_sorting_exists = is_folder_with_files(output_folder.sorting_final) # check if there is spike sorting output for current recording/probe
    post_processing_exists = is_folder_with_files(output_folder.analyzer_local) or is_folder_with_files(output_folder.analyzer_final) # check if there is already post-processing output for current recording/probe

    if preprocessing_file_exists and spike_sorting_exists and not post_processing_exists: # run postprocessing
        saved_preprocessed_recording = si.load(output_folder.preprocessing) # load preprocessed recording
        sorted_spikes = si.load(output_folder.sorting_final) # load spike sorting data
        sorting_analyzer = si.create_sorting_analyzer(sorting=sorted_spikes, recording=saved_preprocessed_recording, format="binary_folder") # sorting analyzer combines recording & sorting object
        print("Computing extensions...")
        sorting_analyzer.compute(postprocessing_configurations)
        sorting_analyzer.save_as(folder=output_folder.analyzer_local, format="binary_folder") # save
        copy_data(output_folder.analyzer_local, output_folder.analyzer_final)
    elif post_processing_exists:
        if overwrite_existing_files:
            print(f"\033[33mWarning: Already post-processed data of probe before, loading sorting analyzer\033[0m")
            sorting_analyzer = si.load(output_folder.analyzer_final)
            print("Computing extensions...")
            sorting_analyzer.compute(postprocessing_configurations)
            sorting_analyzer.save_as(folder=output_folder.analyzer_local, format="binary_folder") # save
            copy_data(output_folder.analyzer_local, output_folder.analyzer_final)
        else:
            print(f"\033[33mWarning: Already post-processed data of probe before, skipping\033[0m")
    else:
        print(f"\033[31mWarning: No preprocessed data or spike sorting output found for probe, skipping\033[0m")


def curate_probe(output_folder, method, curation_thresholds):
    # check if analyzer exists
    if is_folder_with_files(output_folder.analyzer_final):
        sorting_analyzer = si.load(output_folder.analyzer_final)
    else:
        print(f"\033[31mWarning: No corresponding sorting analyzer found for probe, skipping\033[0m")
        return

    match method:
        case "bombcell":
            bombcell_labels = si_curation.bombcell_label_units(sorting_analyzer, thresholds=curation_thresholds, label_non_somatic=True, split_non_somatic_good_mua=True)
            print(bombcell_labels["bombcell_label"].value_counts()) # classification result (mua, noise, good, non_soma_mua)
            non_noisy_units = bombcell_labels["bombcell_label"] != "noise"
        case "simple_thresholds":
            all_metrics = sorting_analyzer.get_metrics_extension_data()
            curation_labels = si_curation.threshold_metrics_label_units(all_metrics, thresholds=curation_thresholds, column_name="simple_threshold")
            print(curation_labels["simple_threshold"].value_counts()) # classification result (good, noise)
            non_noisy_units = curation_labels["simple_threshold"] != "noise"
        case _:
            raise ValueError(f"Curation method {method} not recognized.\n   Please choose from: bombcell , simple_thresholds")
        
    sorting_analyzer_curated = sorting_analyzer.select_units(sorting_analyzer.unit_ids[non_noisy_units]) # remove noisy units
    sorting_analyzer_curated.save_as(folder=output_folder.analyzer_final/"curated", format="binary_folder")


### classes ###
class NeuropixelsData:
    def __init__(self, recordings_folder, recordings_to_process, spike_sorter, local_output_folder, final_output_folder):
        self.recordings_to_process = create_recording_path_list(recordings_folder, recordings_to_process) # create list of file paths to process
        self.spike_sorter = spike_sorter
        self.local_output_folder = local_output_folder
        self.final_output_folder = final_output_folder

        # set parallel processing variables
        si.set_global_job_kwargs(n_jobs=determine_optimal_n_jobs(), chunk_duration = "1s", progress_bar = True)


    def run_preprocessing(self, preprocessing_configurations, ignore_existing_files = False):
        # load corresponding parameters (defined in processing parameters.py)
        preprocessing_parameters = get_parameters_for_sorter(self.spike_sorter, preprocessing_configurations) # load corresponding parameters (defined in processing parameters.py)

        check_folder_structure(self.local_output_folder, ["recordings_preprocessed"])

        print(f"Chosen spike sorter: {self.spike_sorter}\n")
        print(f"{'preprocessing parameters'.center(60, '-')}\n{pformat(preprocessing_parameters, sort_dicts=False)}\n") # can also use si_preprocessing.PreprocessingPipeline(preprocessing_parameters[spike_interface]) to print parameters for spike interface part
        print(f"{"-"*60}\n")
        
        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)
            
                raw_recording = si_extractors.read_spikeglx(recording_path, stream_id=probe_stream) # load recording
                print(raw_recording)
                preprocess_probe(raw_recording, output_folder, preprocessing_parameters, ignore_existing_files = ignore_existing_files)


    def run_spike_sorting(self, spike_sorting_configurations):
        spike_sorting_parameters = get_parameters_for_sorter(self.spike_sorter, spike_sorting_configurations) # load corresponding parameters (defined in processing parameters.py)
        
        check_cuda_availability()

        # check folder structure
        check_folder_structure(self.local_output_folder, ["recordings_spike_sorted"])
        check_folder_structure(self.final_output_folder, ["recordings_spike_sorted"])

        print(f"Chosen spike sorter: {self.spike_sorter}\n")
        print(f"{'spike sorting parameters'.center(60, '-')}\n{pformat(spike_sorting_parameters, sort_dicts=False)}")
        print(f"Defaults for {self.spike_sorter}: {(si_sorters.get_default_sorter_params("kilosort4"))}")
        print(f"{"-"*60}\n")

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                spike_sort_probe(output_folder, self.spike_sorter, spike_sorting_parameters)

                copy_data(output_folder.sorting_local, output_folder.sorting_final) # copy spike sorting to final folder, delete locally saved spike sorting data


    def run_postprocessing(self, postprocessing_configurations, delete_preprocessing_data = False, delete_raw_spike_sorting = False, overwrite_existing_files = False):

        ## check folder structure for postprocessing
        check_folder_structure(self.local_output_folder, ["sorting_analyzers", "postprocessing/figures"])
        check_folder_structure(self.final_output_folder, ["sorting_analyzers", "postprocessing/figures"])

        print(f"{'extensions to be computed'.center(60, '-')}\n{pformat(postprocessing_configurations, sort_dicts=False)}")
        print(f"{"-"*60}\n")

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
                
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")
                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)
                
                postprocess_probe(output_folder, postprocessing_configurations, overwrite_existing_files = overwrite_existing_files) # run postprocessing for current probe

                if delete_preprocessing_data:
                    shutil.rmtree(output_folder.preprocessing) # deletes preprocessing data
                if delete_raw_spike_sorting:
                    shutil.rmtree(output_folder.sorting_raw) # deletes raw spike sorting data


    def automatic_curation(self, method, curation_thresholds):

        print(f"Chosen curation method: {method}\n")
        if method == "bombcell":
            bombcell_default_thresholds = si_curation.bombcell_get_default_thresholds()
            curation_thresholds = update_dict(bombcell_default_thresholds, curation_thresholds) # modify bombcell defaults based on curation_thresholds
        print(f"{'curation thresholds'.center(60, '-')}\n{pformat(curation_thresholds, sort_dicts=False)}")
        print(f"{"-"*60}\n")

        for recording_path in self.recordings_to_process:
            print(f"Processing recording {recording_path.name}...")
            
            ap_streams = check_probe_n(recording_path)

            for probe_i, probe_stream in enumerate(ap_streams): # loop over probes
                print(f"Processing probe {probe_i+1}/{len(ap_streams)}...")

                output_folder = OutputPaths(self.local_output_folder, self.final_output_folder, recording_path.name, probe_stream, self.spike_sorter)

                curate_probe(output_folder, method, curation_thresholds)   