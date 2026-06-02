# based on detect_bad_channels from IBL (https://github.com/int-brain-lab/ibl-neuropixel)
# modified outside brain detection, median doesn't work for recordings where >=50% of channels may be outside brain

# but keep separate so you can still use ibl version as well, maybe switch in function call where u detect outside based on LFP

# clean up at some point, see how it works on other recordings

import scipy
import numpy as np
from pathlib import Path
from ibldsp.voltage import spikeglx, detect_bad_channels


# either way a plot of the final selected channels should be saved

# in pipeline you should still be able to use ibl methods for outside channel detection, best if theyd work with cbin

# separate ipynb for outside channel detection that saves a dictionary for all files with channel boundary

# then in preprocessing pipeline outside_channel_removal with input arg path of dict

# regardless plot!

# manual -> n iterations, take average integer, save



def detect_outside_channels_batched(
    bin_file: Path | str | spikeglx.Reader,
    n_batches: int = 10, # 20 better?
    batch_duration: float = None,
    # display: bool = False,
    detection_function = detect_bad_channels,
    **detection_kwargs) -> np.ndarray:
    """
    # modified from ibldsp.voltage.detect_bad_channels_cbin()
    Detect faulty channels in a SpikeGLX binary or compressed binary file.

    This function scans an electrophysiology recording file in multiple batches throughout
    its duration to identify problematic channels. It uses the `detect_outside_channels` function
    on each batch and takes the mode of the detections across all batches to produce a robust
    channel quality assessment.

    Parameters
    ----------
    bin_file : Path | str | spikeglx.Reader
        Full file path to the binary or compressed binary file from SpikeGLX, or an existing
        spikeglx.Reader object. If a path is provided, a Reader will be created automatically.
    n_batches : int, optional
        Number of batches to sample throughout the file for channel quality assessment.
        Defaults to 10. More batches provide more robust detection but increase computation time.
    batch_duration : float, optional
        Duration of each batch in seconds. If None, defaults to 0.33 seconds for AP band
        recordings (fs ~ 30 kHz) or 4 seconds for LF band recordings (fs <= 2500 Hz).
    display : bool, optional
        If True, displays a diagnostic figure showing channel features and an excerpt of the
        raw data using the `ephys_bad_channels` plotting function. Defaults to False.

    Returns
    -------
    numpy.ndarray
        Integer array of shape (nc,) containing channel quality labels, where nc is the number
        of channels (excluding sync channels):
        - 0: good/ok channel
        - 1: dead channel (low coherence/amplitude)
        - 2: high noise channel
        - 3: outside of the brain
    """
    recording = (bin_file if isinstance(bin_file, spikeglx.Reader) else spikeglx.Reader(bin_file))
    # this is 0.33s for AP and 4s for LF
    batch_duration = 4
    if batch_duration is None:
        batch_duration = 1e4 / recording.fs

    # if n batch = 1 select in middle of recording!

    print(recording.fs)

    n_channels = recording.nc - recording.nsync
    channel_labels = np.zeros((n_channels, n_batches))
    # loop over the file and take the mode of detections
    boundaries = np.zeros(n_batches)

    if n_batches > 1:
        batch_times = np.linspace(0, recording.rl - batch_duration, n_batches)
    else:
        batch_times = [recording.rl*0.5 - batch_duration/2] # take sample from middle of recording

    slices = []
    for i, t0 in enumerate(batch_times):
        sl = slice(int(t0 * recording.fs), int((t0 + batch_duration) * recording.fs))
        slices.append(sl)
        channel_labels[:, i], _xfeats = detection_function(recording[sl, :n_channels].T, fs=recording.fs, **detection_kwargs)
        if i == 0:  # init the features dictionary if necessary
            xfeats = {key: np.zeros((n_channels, n_batches)) for key in _xfeats}
        for k in xfeats:
            xfeats[k][:, i] = _xfeats[k]
        # boundaries[i] = _xfeats["outside_boundary"][0]
    # the features are averaged  so there may be a discrepancy between the mode and applying
    # the thresholds to the average of the features - the goal of those features is for display only
    # final_boundary = int(np.median(boundaries)) # or go bak to mode?/median # mean like 10 channels too much :()
    # final_channel_labels = np.zeros(n_channels)
    # final_channel_labels[final_boundary:] = 3
    xfeats_median = {k: np.median(xfeats[k], axis=-1) for k in xfeats}
    final_channel_labels, _ = scipy.stats.mode(channel_labels, axis=1)
    # if display:
    #     raw = recording[sl, :n_channels].T
    #     from ibllib.plots.figures import ephys_bad_channels
    #     ephys_bad_channels(raw, recording.fs, channel_flags, xfeats_med)
    return final_channel_labels, xfeats_median #, slices