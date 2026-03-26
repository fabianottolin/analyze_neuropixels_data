preprocessing -> spike sorting -> postprocessing -> curation -> visualization & QC -> result collection -> NWB packaging

~1h recording -> processing times (with 24 physical cores, NVIDIA RTX A4000)
- preprocessing: 01:30h
- spike sorting: 00:30h
- post processing:
- curation: 00:00h

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