import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from pprint import pformat
from utility_functions import check_probe_n, OutputPaths, load_json, save_json
from ibldsp.voltage import spikeglx
from ibldsp.plots import voltageshow


def check_backend(verbose = False):
    backend = matplotlib.get_backend()
    if verbose: print("Current backend:", backend)

    if "widget" not in backend.lower():
        raise RuntimeError("Interactive 'widget' backend is not active. Run `%matplotlib widget` in Jupyter before starting user interface.")


def initilize_recordings_dict(neuropixels_data):
    recordings = []
    for recording_path in neuropixels_data.recordings_to_process:
        
        ap_streams = check_probe_n(recording_path)
        recording_name = recording_path.name
        for probe_stream in ap_streams:
            
            probe = probe_stream.split(".")[0]
            path_lfp_binary = recording_path/f"{recording_name}_{probe}"/f"{recording_name}_t0.{probe}.lf.bin"

            recordings.append({"recording_name": recording_name, "recording_path": recording_path, "probe_stream": probe_stream, "probe": probe, "lfp_path": path_lfp_binary,
                               "output_folder": OutputPaths(neuropixels_data.local_output_folder, neuropixels_data.final_output_folder, recording_name, probe_stream)})
    return recordings


def get_lfp_window(path_lfp_binary, duration_sample = 10):
    # read file
    sr = spikeglx.Reader(path_lfp_binary)
    start_sample = np.random.randint(0, int(sr.ns - duration_sample * sr.fs)) # random start sample for batch
    raw = sr[start_sample:start_sample + int(duration_sample * sr.fs), :sr.nc - sr.nsync].T

    
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset

    return raw, start_sample, sr.fs


def manual_outside_channel_detector(neuropixels_data, duration_sample = 10, overwrite_existing_data = False, subset = None):
    """
    Interactive UI to detect outside channels manually based on raw LFP voltage heatmap.

    ------ Parameters ------
    neuropixels_data: NeuropixelsData
        Object containing all relevant paths and parameters for analysis
    duration_sample: float
        Length of plotted LFP window in seconds
    overwrite_existing_data: bool
    subset: list
        List of recording identifiers ('recording}_{probe}') to do manual outside channel detection for.
        If None all recordings in neuropixels_data.recordings_to_process will be included.
    """
    check_backend()
    print("Initializing widget for manual outside channel detection...")
    recording_information_list = initilize_recordings_dict(neuropixels_data)
    
    if subset is not None:
        if not isinstance(subset, list):
            subset = [subset]
        recording_information_list = [rec for rec in recording_information_list if rec["output_folder"].recording_identifier in subset]

    final_output_folder = recording_information_list[0]["output_folder"].final_output_folder # same for all recordings
    filename = "manually_selected_channels.json"
    filepath = final_output_folder/filename

    if not filepath.exists():
        results_manual_selection = {}
    else:
        results_manual_selection = load_json(filepath)
        if not overwrite_existing_data:
            recording_information_list = [rec for rec in recording_information_list if rec["output_folder"].recording_identifier not in results_manual_selection]
            print(f"Already selected outside channels for {len(results_manual_selection)} recordings, skipping these.\nSet overwrite_existing_data to True if you want to redo manual outside channel selection.")
            if len(recording_information_list) == 0:
                raise ValueError("All recordings in recordings_to_process have already been processed. Choose different recordings or set overwrite_existing_data = True")
        else:
            print(f"You selected 'overwrite_existing_data' = True. Existing manual outside channel selections for chosen recordings will be overwritten.")

    state = {"recordings": recording_information_list,
            "i": 0,
            "last_outside_channel": None,
            "results": results_manual_selection,
            "selected_channel": None}

    plt.ioff() # prevent automatic figure display
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    fig.canvas.layout.flex = '0 0 auto'

    out = widgets.Output()

    def message(msg, clear_output = True):
        if clear_output:
            out.clear_output(wait=True)
        with out:
            print(msg)


    def draw_LFP_heatmap():
        ax.clear()

        raw, start_sample, fs = get_lfp_window(state["recordings"][state["i"]]["lfp_path"], duration_sample=duration_sample)
        n_channels, _ = raw.shape

        im = voltageshow(raw, fs, ax=ax)
        ax.set(ylabel="Channel #", ylim=[0, n_channels])
        ax.set_title(f"LFP voltage ({state['recordings'][state['i']]['output_folder'].recording_identifier})")
        fig.suptitle("Manual outside channel detection", fontsize = 18)
        ax.set_xticklabels([round(i + start_sample/fs, 0) for i in ax.get_xticks()])
        fig.tight_layout()

        state["hline"] = ax.axhline(0, color="cyan")
        state["hline"].set_visible(False)

        fig.canvas.draw_idle()


    def draw_boundary(click):
        if click.inaxes != ax:
            return

        selected_channel = int(round(click.ydata))
        state["selected_channel"] = selected_channel

        state["hline"].set_ydata([selected_channel, selected_channel])
        state["hline"].set_visible(True)
        fig.canvas.draw_idle()
        message(f"Selected channel {selected_channel}")


    def confirm(allow_none = False):

        if allow_none:
            last_outside_channel = None
        else:
            if state["selected_channel"] is None:
                message("You need to select a channel before confirming")
                return
            last_outside_channel = state["selected_channel"]
        
        state["results"][f"{state['recordings'][state['i']]['output_folder'].recording_identifier}"] = last_outside_channel

        state["i"] += 1
        state["selected_channel"] = None

        next_plot()


    def next_plot():
        if state["i"] >= len(state["recordings"]):
            message(f"Finished manual outside channel detection for all recordings in recordings_to_process\nResults: {pformat(state["results"])}")
            btn_confirm.disabled = True
            btn_resample.disabled = True
            btn_confirm_none.disabled = True
            ax.clear()
            ax.set_visible(False)
            fig.canvas.draw_idle()
            save_json(state["results"], final_output_folder, filename)
            message(f"Results saved under {filepath}", clear_output=False)
            return state["results"]

        draw_LFP_heatmap()
        message(f"Please choose the last outside channel.")


    def resample(_):
        state["last_outside_channel"] = None
        draw_LFP_heatmap()
        if state["selected_channel"] is not None:
            state["hline"].set_ydata([state["selected_channel"], state["selected_channel"]])
            state["hline"].set_visible(True)
            message(f"New random sample selected.\nSelected channel {state['selected_channel']}")
        else:
            message(f"New random sample selected.\nPlease choose the last outside channel.")


    # buttons
    btn_w = widgets.Layout(width="400px") 
    btn_confirm = widgets.Button(description="Confirm selected channel and continue", layout = btn_w)
    btn_resample = widgets.Button(description="Choose another random sample", layout = btn_w)
    btn_confirm_none = widgets.Button(description="No outside channels in recording", layout=btn_w)

    # click events
    btn_confirm.on_click(lambda _: confirm(allow_none=False))
    btn_resample.on_click(resample)
    btn_confirm_none.on_click(lambda _: confirm(allow_none=True))
    fig.canvas.mpl_connect("button_press_event", draw_boundary)

    sidebar = widgets.VBox([btn_confirm, btn_resample, btn_confirm_none, out], layout=widgets.Layout(padding="8px", width="auto"))

    draw_LFP_heatmap()
    message(f"Please choose the last outside channel.")
    container = widgets.HBox([fig.canvas, sidebar], layout=widgets.Layout(align_items="flex-start"))
    display(container)

