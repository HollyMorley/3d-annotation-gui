"""Frame extraction tool — synchronize and extract frames from multi-camera videos."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from annotation_tool.config import (
    DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST, FRAME_SAVE_PATH_TEMPLATE,
    REFERENCE_VIEW, VIEWS,
)
from annotation_tool.gui.utils import (
    apply_contrast_brightness, get_video_name_with_view, parse_video_path,
    get_corresponding_video_path,
)
from annotation_tool.gui.sync import (
    load_timestamps, zero_timestamps, adjust_timestamps, match_frames_by_timestamp,
)


class ExtractFramesTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.caps = {}
        self.total_frames = 0
        self.current_frame_index = 0
        self.matched_frames = []
        self.contrast_var = tk.DoubleVar(value=DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=DEFAULT_BRIGHTNESS)

        self.extract_frames()

    def extract_frames(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(
            title=f"Select {REFERENCE_VIEW.capitalize()} Video File"
        )
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = parse_video_path(self.video_path)
        if self.camera_view != REFERENCE_VIEW:
            messagebox.showerror(
                "Wrong Camera View",
                f"Please select the {REFERENCE_VIEW} video (the reference view "
                f"configured in config.REFERENCE_VIEW). You picked the "
                f"{self.camera_view} video.",
            )
            self.main_tool.main_menu()
            return
        self.video_name_stripped = "_".join(self.video_name.split("_")[:-1])

        # Open video captures for every view (reference uses the picked path,
        # others are derived from it).
        self.caps[REFERENCE_VIEW] = cv2.VideoCapture(self.video_path)
        for view in VIEWS:
            if view != REFERENCE_VIEW:
                self.caps[view] = cv2.VideoCapture(
                    get_corresponding_video_path(self.video_path, self.camera_view, view)
                )
        self.total_frames = int(self.caps[REFERENCE_VIEW].get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        frame_counts = {v: int(self.caps[v].get(cv2.CAP_PROP_FRAME_COUNT)) for v in VIEWS}
        print("Total frames - " + ", ".join(
            f"{v.capitalize()}: {frame_counts[v]}" for v in VIEWS
        ))

        # Load and synchronize timestamps. The reference view's timeline is
        # taken as ground truth; each other view's timestamps are linearly
        # adjusted to align with it.
        timestamps = {
            v: zero_timestamps(load_timestamps(self.video_path, self.video_name, v))
            for v in VIEWS
        }
        ts_adj = {REFERENCE_VIEW: timestamps[REFERENCE_VIEW]["Timestamp"].astype(float)}
        for v in VIEWS:
            if v != REFERENCE_VIEW:
                ts_adj[v] = adjust_timestamps(timestamps[REFERENCE_VIEW], timestamps[v])

        self.matched_frames = match_frames_by_timestamp(ts_adj, REFERENCE_VIEW, VIEWS)

        self.show_frames_extraction()

    def show_frames_extraction(self):
        self.main_tool.clear_root()

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.slider = tk.Scale(
            control_frame, from_=0, to=len(self.matched_frames) - 1,
            orient=tk.HORIZONTAL, length=600, command=self.update_frame_label,
        )
        self.slider.pack(side=tk.LEFT, padx=5)

        ref_idx = VIEWS.index(REFERENCE_VIEW)
        self.frame_label = tk.Label(
            control_frame, text=f"Frame: {self.matched_frames[0][ref_idx]}"
        )
        self.frame_label.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(side=tk.TOP, pady=10)

        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000),
        ]
        for i, (text, step) in enumerate(buttons):
            tk.Button(skip_frame, text=text,
                      command=lambda s=step: self.skip_frames(s)).grid(row=0, column=i, padx=5)

        control_frame_right = tk.Frame(self.root)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        tk.Button(control_frame_right, text="Extract Frames",
                  command=self.save_extracted_frames).pack(pady=5)
        tk.Button(control_frame_right, text="Back to Main Menu",
                  command=self.main_tool.main_menu).pack(pady=5)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.display_frame(0)

    def skip_frames(self, step):
        new_index = self.current_frame_index + step
        new_index = max(0, min(new_index, len(self.matched_frames) - 1))
        self.slider.set(new_index)
        self.display_frame(new_index)

    def read_matched(self, index):
        """Read the matched frame for every view at the given matched index.

        Returns dict[view -> BGR image] or None if any view failed to read.
        """
        frame_nums = dict(zip(VIEWS, self.matched_frames[index]))
        imgs = {}
        for view in VIEWS:
            self.caps[view].set(cv2.CAP_PROP_POS_FRAMES, frame_nums[view])
            ret, img = self.caps[view].read()
            if not ret:
                return None, frame_nums
            imgs[view] = img
        return imgs, frame_nums

    def display_frame(self, index):
        self.current_frame_index = index
        imgs, _ = self.read_matched(index)
        if imgs is None:
            return

        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()

        for ax, view in zip(self.axs, VIEWS):
            adjusted = apply_contrast_brightness(imgs[view], contrast, brightness)
            ax.cla()
            ax.imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{view.capitalize()} View")

        self.canvas.draw()

    def update_frame_label(self, val):
        index = int(val)
        ref_idx = VIEWS.index(REFERENCE_VIEW)
        self.frame_label.config(text=f"Frame: {self.matched_frames[index][ref_idx]}")
        self.display_frame(index)

    def save_extracted_frames(self):
        imgs, frame_nums = self.read_matched(self.current_frame_index)
        if imgs is None:
            return

        video_names = {
            view: get_video_name_with_view(self.video_name, view)
            for view in VIEWS
        }

        for view in VIEWS:
            path = os.path.join(
                FRAME_SAVE_PATH_TEMPLATE[view].format(video_name=video_names[view]),
                f"img{frame_nums[view]}.png",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, imgs[view])

        messagebox.showinfo("Info", "Frames saved successfully")
