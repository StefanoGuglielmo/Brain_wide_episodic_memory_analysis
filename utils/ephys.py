import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json

def plot_recording_with_amplitude(rec, channel_idx=0, time_range=(0, 1), y_label="($\mu V$)"):
    fs = rec.get_sampling_frequency()
    start_frame = int(time_range[0] * fs)
    end_frame = int(time_range[1] * fs)
    
    traces = rec.get_traces(start_frame=start_frame, 
                            end_frame=end_frame, 
                            return_scaled=False)
    
    time_vector = np.linspace(time_range[0], time_range[1], traces.shape[0])
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_vector, traces[:, channel_idx], color='black', lw=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Amplitude {y_label}')
    plt.title(f'Channel {rec.channel_ids[channel_idx]}')
    plt.grid(True, alpha=0.3)
    plt.show()



def select_top_channels(recording, n_channels=4):
    traces = recording.get_traces()
    channel_std = np.std(traces, axis=0)

    top_channel_indices = np.argsort(channel_std)[-n_channels:][::-1]
    top_channels = [
        recording.channel_ids[index]
        for index in top_channel_indices
    ]

    return recording.select_channels(top_channels), top_channels


from probeinterface import Probe

def add_probe_geometry(recording, positions, contact_radius=5):
    n_channels = recording.get_num_channels()

    probe = Probe(ndim=2, si_units="um")

    probe.set_contacts(
        positions=positions[:n_channels],
        shapes="circle",
        shape_params={"radius": contact_radius},
    )

    probe.set_device_channel_indices(np.arange(n_channels))

    return recording.set_probe(probe)



def prepare_exploration_epochs(
    exploration_file,
    led_file,
    fps=30.02,
    sampling_rate=30000,
):    
    try:
        exploration = pd.read_csv(exploration_file)

        if "start_frame" not in exploration.columns:
            exploration = pd.read_csv(
                exploration_file,
                sep=";",
            )

    except Exception:
        exploration = pd.read_csv(
            exploration_file,
            sep=";",
        )

    led_data = pd.read_csv(led_file)
    frame_offset = int(led_data["Frame"].iloc[0])

    exploration = exploration.copy()

    exploration["start_frame"] -= frame_offset
    exploration["end_frame"] -= frame_offset

    exploration = (
        exploration[exploration["start_frame"] >= 0]
        .reset_index(drop=True)
    )

    half_window_frames = int(round(fps * 1.5))

    midpoint = (
        exploration["start_frame"]
        + exploration["end_frame"]
    ) / 2

    exploration["start_frame_window"] = (
        midpoint - half_window_frames
    ).round().clip(lower=0).astype(int)
    
    exploration["end_frame_window"] = (
        midpoint + half_window_frames
    ).round().astype(int)

    exploration["start_time_window_s"] = (
        exploration["start_frame_window"] / fps
    )

    exploration["end_time_window_s"] = (
        exploration["end_frame_window"] / fps
    )

    exploration["start_time_s"] = (
        exploration["start_frame"] / fps
    )

    exploration["end_time_s"] = (
        exploration["end_frame"] / fps
    )

    exploration["start_sample"] = (
        exploration["start_frame_window"]
        / fps
        * sampling_rate
    ).round().astype(int)

    exploration["end_sample"] = (
        exploration["end_frame_window"]
        / fps
        * sampling_rate
    ).round().astype(int)

    epochs = exploration[
        ["label", "start_sample", "end_sample"]
    ].copy()

    epochs["start_time_s"] = (
        epochs["start_sample"] / sampling_rate
    )

    epochs["end_time_s"] = (
        epochs["end_sample"] / sampling_rate
    )

    epochs["start_exploration_s"] = (
        exploration["start_frame"] / fps
    )

    epochs["end_exploration_s"] = (
        exploration["end_frame"] / fps
    )

    return exploration, epochs



def extract_exploration_aligned_activity(
    mouse_id,
    region,
    sorting,
    epochs_df,
    output_file,
    psth_window=(-3, 3),
    rate_window=(0, 2),
    bin_size=0.1,
    sampling_rate=30000,
):

    psth_bins = np.arange(
        psth_window[0],
        psth_window[1] + bin_size,
        bin_size,
    )

    results = {
        "mouse_id": str(mouse_id),
        "region": str(region),
        "num_units": int(len(sorting.unit_ids)),
        "data_by_label": {},
    }

    for label in epochs_df["label"].unique():

        label_df = epochs_df[
            epochs_df["label"] == label
        ]

        num_trials = len(label_df)
        unit_psths = []

        for unit_id in sorting.unit_ids:

            spike_times = (
                sorting.get_unit_spike_train(unit_id)
                / sampling_rate
            )

            spike_offsets = []

            for start_time in label_df["start_exploration_s"]:

                relative_spikes = spike_times - start_time

                spike_offsets.extend(
                    relative_spikes[
                        (relative_spikes >= psth_window[0])
                        & (relative_spikes <= psth_window[1])
                    ]
                )

            histogram, _ = np.histogram(
                spike_offsets,
                bins=psth_bins,
            )

            if num_trials > 0:
                histogram = histogram / (
                    num_trials * bin_size
                )

            unit_psths.append(histogram)

        mean_psth = np.mean(unit_psths, axis=0)

        rate_mask = (
            (psth_bins[:-1] >= rate_window[0])
            & (psth_bins[:-1] < rate_window[1])
        )

        mean_firing_rate = float(
            np.mean(mean_psth[rate_mask])
        )

        results["data_by_label"][str(label)] = {
            "mean_firing_rate": mean_firing_rate,
            "psth_x": psth_bins[:-1].tolist(),
            "psth_y": mean_psth.tolist(),
        }

    with output_file.open("w") as file:
        json.dump(results, file, indent=2)

    print(f"Saved exploration data to: {output_file}")



def normalize_and_smooth_psth(
    psth,
    baseline,
    sigma=1.5,
):

    psth = np.asarray(psth, dtype=float)

    if pd.isna(baseline) or baseline == 0:
        return np.zeros_like(psth)

    normalized = psth / baseline

    return gaussian_filter1d(
        normalized,
        sigma=sigma,
    )



def calculate_window_mean(row, start_time, end_time, column):
    
    time = np.asarray(row["psth_x"])
    values = np.asarray(row[column])

    mask = (time >= start_time) & (time < end_time)

    return float(values[mask].mean()) if mask.any() else np.nan



def load_unit_metadata(file_path, UNIT_COLUMNS):

    units = pd.read_csv(file_path)

    units = units[
        [column for column in UNIT_COLUMNS if column in units.columns]
    ].copy()

    filename = file_path.stem

    parts = filename.split("_")

    mouse_id = parts[0]
    region = parts[1]

    units["mouse"] = mouse_id
    units["region"] = region

    return units