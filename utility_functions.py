### utility functions for data processing ###

# imports
import torch
from pathlib import Path
import spikeinterface.extractors as si_extractors
import shutil
import filecmp
import psutil

# functions
def get_parameters_for_sorter(spike_sorter, *configurations): # checks if key exists for configurations dictionaries and returns values associated with the key
    parameters_for_sorter = []
    for config in configurations:
        if spike_sorter not in config:
            raise KeyError(f"Unknown spike sorter '{spike_sorter}' in configuration file. Please check processing_parameters.py and adjust accordingly.")
        parameters_for_sorter.append(config[spike_sorter])
    return parameters_for_sorter[0] if len(parameters_for_sorter) == 1 else parameters_for_sorter


def create_recording_path_list(recordings_folder, recordings_to_process): # checks if recordings to process exist in recordings folder and creates new list with final file paths
    recording_path_list = []
    for recording in recordings_to_process:
        recording_path = recordings_folder / recording
        if recording_path.exists():
            recording_path_list.append(recording_path)
        else:
            print(f"Warning: recording {recording} not found in {recordings_folder}, skipping.")
    return recording_path_list


def check_cuda_availability(): # checks for cuda availability
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError(f"CUDA is not available.\nPyTorch version: {torch.__version__}\n"
            "Ensure you have a CUDA-enabled PyTorch build, compatible NVIDIA GPU, and correct drivers installed.\n"
            "(Use pip uninstall torch pip3 install torch --index-url https://download.pytorch.org/whl/cu118 to install the correct version of torch)")


def check_folder_structure(parent_folder: Path, folder_structure: list): # checks if folder structure exists in parent folder and creates it if not, input is Path() object
    if all((parent_folder / folder).exists() for folder in folder_structure):
        print("Folder structure exists as expected.\n")
    else:
        for folder in folder_structure:
            (parent_folder / folder).mkdir(parents=True, exist_ok=True)
        print("Folder structure created.\n")


def show_available_sorters(sorter_module):
    print(f"Available sorters: {", ".join(sorter_module.available_sorters())}")
    print(f"Installed sorters: {", ".join(sorter_module.installed_sorters())}")


def check_probe_n(recording_path):
    stream_names, _ = si_extractors.get_neo_streams("spikeglx", recording_path)
    ap_streams = [stream for stream in stream_names if stream.endswith(".ap")]
    print(f"{len(ap_streams)} probe(s) found")
    return ap_streams


def is_folder_with_files(folder): # returns boolean
    files_exist = folder.exists() and any(folder.iterdir()) # check if folder exists and if there are files inside
    return files_exist


def same_files_in_folder(folder1:Path, folder2:Path, shallow = False): # checks if files in folder and all subfolders are the same
        
    comparison = filecmp.dircmp(folder1, folder2)

    if comparison.left_only or comparison.right_only: # differences in folder contents
        return False
    
    if shallow and comparison.diff_files: # differences in size or modification time
        return False
    elif not shallow: # byte-by-byte check
        for file in comparison.common_files:
            if not filecmp.cmp(folder1/file, folder2/file, shallow = False):
                return False
            
    for subfolder in comparison.subdirs.values(): # check subfolders recursively
        if not same_files_in_folder(subfolder.left, subfolder.right, shallow = shallow):
            return False
    
    return True


def copy_data(origin_folder, destination_folder): # copies spike sorting results from local to final storage, then deletes local copy
    if not is_folder_with_files(origin_folder): # check if spike sorting data exists locally
        raise ValueError("There is no data in local output folder")
    
    print("Copying files to final folder...")
    shutil.copytree(origin_folder, destination_folder, dirs_exist_ok=True) # copy files

    print("Checking if copied correctly...")
    if same_files_in_folder(origin_folder, destination_folder): # check if everything has been copied correctly
        shutil.rmtree(origin_folder)
        print(f"Copied spike sorting to {destination_folder} and removed local copy.")
    else:
        raise RuntimeError(f"Files from {origin_folder} were not copied correctly to {destination_folder}, please copy manually!")


def update_dict(original_dict, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            update_dict(original_dict[key], value)
        else:
            original_dict[key] = value
    return original_dict


def check_free_space(folder, required_space): # checks if there is enough free space in given folder, required_space in GB
    usage = shutil.disk_usage(folder)
    free_space_gb = usage.free / (1024 ** 3) # convert bytes to GB
    if free_space_gb < required_space:
        raise RuntimeError(f"Not enough free space in {folder}. Required: {required_space} GB, Available: {free_space_gb:.2f} GB.")
    

def determine_optimal_n_jobs():
    physical_cores = psutil.cpu_count(logical=False) # n_jobs should match physical cores
    return max(1, int(0.75*physical_cores)) # leave 25% of cores for system processes



### utility classes ###

class OutputPaths:
    def __init__(self, local_output_folder: Path, final_output_folder: Path, recording, probe, spike_sorter):
        self.preprocessing = local_output_folder/"recordings_preprocessed"/recording/probe
        self.sorting_raw = local_output_folder/"spike_sorting_raw"/recording/probe
        self.sorting_local = local_output_folder/"recordings_spike_sorted"/recording/probe/spike_sorter
        self.sorting_final = final_output_folder/"recordings_spike_sorted"/recording/probe/spike_sorter
        self.analyzer_local = local_output_folder/"sorting_analyzers"/recording/probe/spike_sorter
        self.analyzer_final = final_output_folder/"sorting_analyzers"/recording/probe/spike_sorter