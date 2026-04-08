"""Frame extraction tool — synchronize and extract frames from multi-camera videos."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from annotation_tool import config
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
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.matched_frames = []
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)

        self.extract_frames()

    def extract_frames(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = parse_video_path(self.video_path)
        self.video_name_stripped = "_".join(self.video_name.split("_")[:-1])
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(
            get_corresponding_video_path(self.video_path, self.camera_view, "front")
        )
        self.cap_overhead = cv2.VideoCapture(
            get_corresponding_video_path(self.video_path, self.camera_view, "overhead")
        )

        total_frames_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_overhead = int(self.cap_overhead.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames - Side: {total_frames_side}, Front: {total_frames_front}, Overhead: {total_frames_overhead}")

        # Load and synchronize timestamps
        timestamps_side = zero_timestamps(load_timestamps(self.video_path, self.video_name, "side"))
        timestamps_front = zero_timestamps(load_timestamps(self.video_path, self.video_name, "front"))
        timestamps_overhead = zero_timestamps(load_timestamps(self.video_path, self.video_name, "overhead"))

        timestamps_front_adj = adjust_timestamps(timestamps_side, timestamps_front)
        timestamps_overhead_adj = adjust_timestamps(timestamps_side, timestamps_overhead)
        timestamps_side_adj = timestamps_side["Timestamp"].astype(float)

        self.matched_frames = match_frames_by_timestamp(
            timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj
        )

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

        self.frame_label = tk.Label(control_frame, text=f"Frame: {self.matched_frames[0][0]}")
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

    def display_frame(self, index):
        self.current_frame_index = index
        frame_side, frame_front, frame_overhead = self.matched_frames[index]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            contrast = self.contrast_var.get()
            brightness = self.brightness_var.get()
            imgs = [
                apply_contrast_brightness(frame_side_img, contrast, brightness),
                apply_contrast_brightness(frame_front_img, contrast, brightness),
                apply_contrast_brightness(frame_overhead_img, contrast, brightness),
            ]
            titles = ["Side View", "Front View", "Overhead View"]

            for ax, img, title in zip(self.axs, imgs, titles):
                ax.cla()
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title)

            self.canvas.draw()

    def update_frame_label(self, val):
        index = int(val)
        self.frame_label.config(text=f"Frame: {self.matched_frames[index][0]}")
        self.display_frame(index)

    def save_extracted_frames(self):
        frame_side, frame_front, frame_overhead = self.matched_frames[self.current_frame_index]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            video_names = {
                view: get_video_name_with_view(self.video_name, view)
                for view in ["side", "front", "overhead"]
            }
            frame_nums = {"side": frame_side, "front": frame_front, "overhead": frame_overhead}

            for view in ["side", "front", "overhead"]:
                path = os.path.join(
                    config.FRAME_SAVE_PATH_TEMPLATE[view].format(video_name=video_names[view]),
                    f"img{frame_nums[view]}.png",
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)

            cv2.imwrite(os.path.join(
                config.FRAME_SAVE_PATH_TEMPLATE["side"].format(video_name=video_names["side"]),
                f"img{frame_side}.png"), frame_side_img)
            cv2.imwrite(os.path.join(
                config.FRAME_SAVE_PATH_TEMPLATE["front"].format(video_name=video_names["front"]),
                f"img{frame_front}.png"), frame_front_img)
            cv2.imwrite(os.path.join(
                config.FRAME_SAVE_PATH_TEMPLATE["overhead"].format(video_name=video_names["overhead"]),
                f"img{frame_overhead}.png"), frame_overhead_img)

            messagebox.showinfo("Info", "Frames saved successfully")
