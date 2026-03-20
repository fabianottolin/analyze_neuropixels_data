## This file contains configurations for Neuropixels data processing pipeline
## For processing steps where optimal configurations differ based on spike sorter chosen, parameter settings are stored in a nested dictionary accessed via spike sorter name as key

## configurations for pre-processing
    # dict keys are order of preprocessing steps
    # current custom steps available: "correct_bad_channels" -> detects bad channels, interpolates channels inside brain, removes channels out of brain
                                    # "spike_interface" -> continues spikeinterface preprocessing pipeline with given parameters
            # can be seen in CUSTOM_PREPROCESSING_FUNCTION_MAP (in processing_functions.py))
preprocessing_configurations = {"kilosort4": {"spike_interface": {"highpass_filter": {"n_channel_pad": 60},
                                                                  "phase_shift": {}}, # default parameters
                                                                  "bandpass_filter": {"freq_min": 300, "freq_max": 6000}}, # REMOVE because u have highpass?
                                                                  # ADD WHITENING, etc.?
                                              "custom_steps":    {"correct_bad_channels": {"method": "coherence+psd"}, # default method -> "coherence+PSD" # what does IBL use?
                                                                  "spike_interface": {"highpass_spatial_filter": {}},  # destriping
                                                                  "detect_and_correct_drift": {"method": "dredge"}}, # motion correction with dredge
                                "other_sorter": {"spike_interface": {}, # add other sorters here as needed, can also have different custom steps for different sorters if needed
                                                 "custom_steps": {}}}
                  

## configurations for spike sorting
spike_sorting_configurations = {"kilosort4": {"verbose": True, "progress_bar": True, "skip_kilosort_preprocessing": True, "torch_device": "cuda"}, # WHAT ELSE DO I NEED TO EXCLUDE IF PREPROCESSING DONE BEFORE
                                "other_sorter": {}} # add configurations for other sorters here as needed

## configuration for post-processing
ms_before, ms_after = 1.5, 2.5 # define ms before and after (used for computation of waveforms, spike_locations, random_spikes, spike_amplitudes)?
postprocessing_configurations = {"random_spikes": {"max_spikes_per_unit": 500},
                                 "waveforms": {"ms_before": ms_before, "ms_after": ms_after},
                                 "templates": {},
                                 "noise_levels": {},
                                 "spike_amplitudes": {},
                                 "unit_locations": {},
                                 "quality_metrics": {}}


## thresholds for quality metrics parameters
curation_thresholds = {} # currently using bombcell's default thresholds, if you want to update these format has to match si_curation.bombcell_get_default_thresholds()

# %%
###### utility code to decide on parameters ######
### preprocessing ###
# import spikeinterface.preprocessing as si_preprocessing

# print(si_preprocessing.pipeline.pp_names_to_functions.keys()) # see what preprocessing steps are available
# print(si_preprocessing.bandpass_filter.__doc__) # -> see what arguments are available for given preprocessing step


### spike sorting ###
# import spikeinterface.sorters as si_sorters
# from utility_functions import show_available_sorters
# from pprint import pformat

# show_available_sorters(si_sorters) # check available sorters
# print(f"\nDefaults for chosen sorter:\n{pformat(si_sorters.get_default_sorter_params("kilosort4"))}") # check default sorter parameters
# # explanations -> https://kilosort.readthedocs.io/en/latest/parameters.html


### curation ###
# import spikeinterface.curation as si_curation
# from pprint import pprint

# pprint(si_curation.bombcell_get_default_thresholds()) # see bombcell default thresholds & format








