"""Timestamp synchronization and frame matching across multiple cameras."""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_timestamps(video_path, video_name, view):
    """Load timestamps CSV for a given camera view."""
    video_name_base = "_".join(video_name.split("_")[:-1])
    video_number = video_name.split("_")[-1]
    timestamp_file = video_name_base + f"_{view}_{video_number}_Timestamps.csv"
    timestamp_path = os.path.join(os.path.dirname(video_path), timestamp_file)
    return pd.read_csv(timestamp_path)


def zero_timestamps(timestamps):
    """Normalize timestamps so the first value is zero."""
    timestamps["Timestamp"] = timestamps["Timestamp"] - timestamps["Timestamp"][0]
    return timestamps


def adjust_timestamps(side_timestamps, other_timestamps):
    """Correct drift between the side camera and another camera's timestamps
    using linear regression on single-frame intervals."""
    mask = other_timestamps["Timestamp"].diff() < 4.045e+6
    other_single = other_timestamps[mask]
    side_single = side_timestamps[mask]
    diff = other_single["Timestamp"] - side_single["Timestamp"]

    # Fit initial model to straighten the diff curve
    model = LinearRegression().fit(
        side_single["Timestamp"].values.reshape(-1, 1), diff.values
    )
    slope = model.coef_[0]
    intercept = model.intercept_
    straightened_diff = diff - (slope * side_single["Timestamp"] + intercept)
    correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

    # Refit on the lower half of the straightened data for a cleaner estimate
    model_true = LinearRegression().fit(
        side_single["Timestamp"].values[correct_diff_idx].reshape(-1, 1),
        diff.values[correct_diff_idx],
    )
    slope_true = model_true.coef_[0]
    intercept_true = model_true.intercept_
    adjusted = other_timestamps["Timestamp"] - (
        slope_true * other_timestamps["Timestamp"] + intercept_true
    )
    return adjusted


def match_frames_by_timestamp(timestamps_side, timestamps_front, timestamps_overhead):
    """Match frames across three cameras using nearest-timestamp merge.

    Returns a list of (side_frame, front_frame, overhead_frame) tuples.
    Unmatched frames are set to -1.
    """
    buffer_ns = int(4.04e+6)

    timestamps_side = timestamps_side.sort_values().reset_index(drop=True)
    timestamps_front = timestamps_front.sort_values().reset_index(drop=True)
    timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True)

    side_df = pd.DataFrame({
        "Timestamp": timestamps_side,
        "Frame_number_side": range(len(timestamps_side)),
    })
    front_df = pd.DataFrame({
        "Timestamp": timestamps_front,
        "Frame_number_front": range(len(timestamps_front)),
    })
    overhead_df = pd.DataFrame({
        "Timestamp": timestamps_overhead,
        "Frame_number_overhead": range(len(timestamps_overhead)),
    })

    matched_front = pd.merge_asof(
        side_df, front_df, on="Timestamp",
        direction="nearest", tolerance=buffer_ns,
        suffixes=("_side", "_front"),
    )
    matched_all = pd.merge_asof(
        matched_front, overhead_df, on="Timestamp",
        direction="nearest", tolerance=buffer_ns,
        suffixes=("_side", "_overhead"),
    )

    matched_frames = (
        matched_all[["Frame_number_side", "Frame_number_front", "Frame_number_overhead"]]
        .applymap(lambda x: int(x) if pd.notnull(x) else -1)
        .values.tolist()
    )
    return matched_frames
