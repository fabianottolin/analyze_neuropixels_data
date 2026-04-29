# Common analysis pipeline for neuropixels experiments: signal processings, universal graphs and other analyses routines (cane_sugar)
Automated pipeline for processing Neuropixels data with spikeinterface based on parameters specified in processing_parameters.py

## Overview
The basic steps of the processing pipeline are: preprocessing -> spike sorting -> postprocessing -> curation

### Processing parameters
The parameters used in the different processing steps are specified in **processing_parameters.py**.

For more details on parameters consult the documentation of spikeinterface.

### User inputs
After specifying parameters via *processing_parameters.py*, users only have to provide five variables:
- **spike_sorter**: the spike sorter to use, needs to match key in processing_parameter.py
    - available options can be seen by running utility code in processing parameters.py
- **recordings_folder**: Folder containing the raw recordings (SpikeGLX format)
- **recordings_to_process**: list
- **local_output_folder**: used for temporary data
- **final_output_folder**: network folder where final output is saved

### Options for processing
all in one vs single steps (more details below)

**Note:** if processing multiple recordings .run_processing_pipeline uses available temporary disk memory more efficiently (if copying or deleting preprocessing data)

### Outputs
The code creates the following folder structure to save the final outputs

## Methods to process data
### NeuropixelsData class
describe class, below description of methods

#### Standalone processing steps
- **.run_preprocessing()**: methods to run specifed method for all probes
    - Inputs:
- .run_spike_sorting():
- .run_postprocessing():
- .automatic_curation():

#### Running all processing steps for each probe with *.run_processing_pipeline()*
- Function runs specified processing steps in order for each probe in NeuropixelsData.recordings_to_process
- Option to remove certain processing steps
- Decide what happens to temporary data

## Notes
### Estimated processing times
~1h recording -> processing times (with 24 physical cores, NVIDIA RTX A4000)
- preprocessing: 90-120 minutes
- spike sorting: 30-90 minutes
- post processing: 20-30 minutes
- curation: 10-20 minutes
- **total processing time per probe**: 3-4 hours

## Put in correct place above
processing_parameters.py
-> processing steps are specified here

Folder structure
    /recordings_preprocessed -> temporary pre-processing data saved here, local folder only
    /recordings_spike_sorted -> contains sorting analyzer objects
    /postprocessing
        /figures

streams recorded by SpikeGLX
- imec0.ap # neuropixels data AP stream
- imec0.lf # neuropixels LFP channel
- nidq # data from NI-DAQ device