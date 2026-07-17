## This file contains configurations for Neuropixels data processing pipeline
## For processing steps where optimal configurations differ based on spike sorter chosen, parameter settings are stored in a nested dictionary accessed via spike sorter name as key

## configurations for pre-processing
    # dict keys are order of preprocessing steps
      # detect_saturation_periods -> extra step applied to raw recording before pre-processing
      # "spike_interface" -> spikeinterface preprocessing pipeline with given steps/parameters, if using multiple times call spike_intreface1, spike_interface2, ...
      # current custom steps available: "correct_bad_channels" -> detects bad channels, interpolates channels inside brain, removes channels out of brain
                                        # " remove_manually_selected_channels" -> removes channels based on manual selection
                                        # "detect_and_correct_drift" -> motion correction                         
            # can be seen in CUSTOM_PREPROCESSING_FUNCTION_MAP (in processing_functions.py))
preprocessing_configurations = {"kilosort4": {"detect_saturation_periods": {"saturation_threshold_uV": 1200, "diff_threshold_uV": 300}, # NP1.0 saturation_threshold = 1200uV, diff_threshold=300uV/sample; NP2.0 6250uV, 300 (from IBL whitepaper)
                                                  # maybe saturation period detection after outside channel removal better? -> test and compare
                                              "remove_manually_selected_channels": {"plot": True}, # removes outside channels detected based on LFP
                                              "spike_interface1": {"highpass_filter": {"freq_min": 300, "filter_order": 3}, # default "freq_min": 300Hz
                                                                 "phase_shift": {}}, # default parameters
                                              "correct_bad_channels": {"method": "coherence+psd"}, # default method -> "coherence+PSD", removes channels outside brain based on AP, interpolates bad channels inside brain
                                              "spike_interface2": {"highpass_spatial_filter": {}, # destriping
                                                                   "silence_periods": {}, # uses periods detected in detect_saturation_periods()
                                                                   "whiten": {"dtype": "float32"}}, # DO I NEED ANY PARAMETERS DIFFERENT FROM DEFAULTS HERE?
                                              "detect_and_correct_drift": {"method": "dredge"}}, # motion correction with dredge
                                              # in documentation they recommend not to use whitening before motion correction, but IBL whitens before?
                                "other_sorter": {"spike_interface1": {}, # add other sorters here as needed, can also have different custom steps for different sorters if needed
                                                 "custom_steps": {}}}


## configurations for spike sorting
spike_sorting_configurations = {"kilosort4": {"verbose": True, "progress_bar": True, "skip_kilosort_preprocessing": True, "torch_device": "cuda", "do_CAR": False, "do_correction": False}, # WHAT ELSE DO I NEED TO EXCLUDE IF PREPROCESSING DONE BEFORE (do car = False?, do_correction= false?)
                                "other_sorter": {}} # add configurations for other sorters here as needed

## configuration for post-processing
ms_before, ms_after = 1.5, 2.5 # define ms before and after (used for computation of waveforms, spike_locations, random_spikes, spike_amplitudes)?
postprocessing_configurations = {"random_spikes": {"max_spikes_per_unit": 500},
                                 "waveforms": {"ms_before": ms_before, "ms_after": ms_after},
                                 "templates": {},
                                 "noise_levels": {},
                                 "spike_amplitudes": {},
                                 "spike_locations": {},
                                 "unit_locations": {},
                                 "template_metrics": {},
                                 "quality_metrics": {}}


## thresholds for quality metrics parameters
curation_thresholds = {} # currently using bombcell's default thresholds, if you want to update these format has to match si_curation.bombcell_get_default_thresholds()

all_processing_configurations = {"preprocessing": preprocessing_configurations,
                                 "spike_sorting": spike_sorting_configurations,
                                 "postprocessing": postprocessing_configurations,
                                 "curation": curation_thresholds}

# %%
###### utility code to decide on parameters ######
### preprocessing ###
# import spikeinterface.preprocessing as si_preprocessing

# print(si_preprocessing.pipeline.pp_names_to_functions.keys()) # see what preprocessing steps are available
# print(si_preprocessing.phase_shift.__doc__) # -> see what arguments are available for given preprocessing step


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








