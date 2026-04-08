"""Camera calibration tool — label known landmarks for camera pose estimation."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton

from annotation_tool import config
from annotation_tool.gui.base import BaseAnnotationTool
from annotation_tool.gui.utils import (
    parse_video_path, get_corresponding_video_path, generate_label_colors, view_index,
)
from annotation_tool.gui.sync import (
    load_timestamps, zero_timestamps, adjust_timestamps, match_frames_by_timestamp,
)


class CalibrateCamerasTool(BaseAnnotationTool):
    def __init__(self, root, main_tool):
        super().__init__(root, main_tool)
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.mode = "calibration"

        self.calibration_points_static = {}
        self.labels = config.CALIBRATION_LABELS
        self.label_colors = generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.calibrate_cameras_menu()

    def calibrate_cameras_menu(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = parse_video_path(self.video_path)
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(
            get_corresponding_video_path(self.video_path, self.camera_view, "front")
        )
        self.cap_overhead = cv2.VideoCapture(
            get_corresponding_video_path(self.video_path, self.camera_view, "overhead")
        )

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

        self.calibration_file_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(
            video_name=self.video_name
        )
        enhanced_calibration_file = self.calibration_file_path.replace(".csv", "_enhanced.csv")
        default_calibration_file = config.DEFAULT_CALIBRATION_FILE_PATH

        # Build UI
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.create_settings_controls(control_frame)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(
            frame_control, from_=0, to=len(self.matched_frames) - 1,
            orient=tk.HORIZONTAL, length=400, command=self.update_frame_label,
        )
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        tk.Button(button_frame, text="Home", command=self.reset_view).pack(pady=5)
        tk.Button(button_frame, text="Save Calibration Points",
                  command=self.save_calibration_points).pack(pady=5)
        tk.Button(button_frame, text="Back to Main Menu",
                  command=self.main_tool.main_menu).pack(pady=5)

        # Label selector
        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            tk.Label(label_frame, bg=color, width=2).pack(side=tk.LEFT, padx=5)
            tk.Radiobutton(
                label_frame, text=label, variable=self.current_label,
                value=label, indicatoron=0, width=20,
            ).pack(side=tk.LEFT)

        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(
                control_frame_right, text=view.capitalize(),
                variable=self.current_view, value=view,
            ).pack(pady=2)

        self.create_3panel_canvas(main_frame)
        self.connect_mouse_events()

        self.calibration_points_static = {
            label: {"side": None, "front": None, "overhead": None}
            for label in self.labels
        }

        # Load existing calibration if available
        if os.path.exists(enhanced_calibration_file):
            self.load_calibration_points(enhanced_calibration_file)
        elif os.path.exists(self.calibration_file_path):
            response = messagebox.askyesnocancel(
                "Calibration Found",
                "Calibration labels found. Do you want to load them? "
                "(Yes to load current, No to load default, Cancel to skip)",
            )
            if response is None:
                pass
            elif response:
                self.load_calibration_points(self.calibration_file_path)
            else:
                if os.path.exists(default_calibration_file):
                    self.load_calibration_points(default_calibration_file)
                else:
                    messagebox.showinfo("Default Calibration Not Found",
                                        "Default calibration file not found.")
        else:
            if os.path.exists(default_calibration_file):
                if messagebox.askyesno(
                    "Default Calibration",
                    "No specific calibration file found. Load default calibration labels?",
                ):
                    self.load_calibration_points(default_calibration_file)
            else:
                messagebox.showinfo("Default Calibration Not Found",
                                    "Default calibration file not found.")

        self.show_frames()

    def load_calibration_points(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.set_index(["bodyparts", "coords"], inplace=True)
            for label in df.index.levels[0]:
                for view in ["side", "front", "overhead"]:
                    if not pd.isna(df.loc[(label, "x"), view]):
                        x, y = df.loc[(label, "x"), view], df.loc[(label, "y"), view]
                        self.calibration_points_static[label][view] = self.axs[
                            view_index(view)
                        ].scatter(
                            x, y, c=self.label_colors[label],
                            s=self.marker_size_var.get() * 10, label=label,
                        )
            self.canvas.draw()
            messagebox.showinfo("Info", "Calibration points loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration points: {e}")

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames()

    def skip_frames(self, step):
        new_index = self.current_frame_index + step
        new_index = max(0, min(new_index, self.total_frames - 1))
        self.current_frame_index = new_index
        self.slider.set(new_index)
        self.frame_label.config(text=f"Frame: {new_index}/{self.total_frames - 1}")
        self.show_frames()

    def show_frames(self, val=None):
        frame_number = self.current_frame_index
        self.frame_label.config(text=f"Frame: {frame_number}/{len(self.matched_frames) - 1}")

        frame_side, frame_front, frame_overhead = self.matched_frames[frame_number]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            self.display_three_views(frame_side_img, frame_front_img, frame_overhead_img)
            self.show_static_points()
            self.canvas.draw_idle()

    def _refresh_display(self):
        self.show_frames()

    def show_static_points(self):
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    ax = self.axs[view_index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size_var.get() * 10])
        self.canvas.draw()

    def save_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(
            video_name=self.video_name
        )
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            for coord in ["x", "y"]:
                data["bodyparts"].append(label)
                data["coords"].append(coord)
                for view in ["side", "front", "overhead"]:
                    if coords[view] is not None:
                        x, y = coords[view].get_offsets()[0]
                        data[view].append(x if coord == "x" else y)
                    else:
                        data[view].append(None)

        df = pd.DataFrame(data)
        df.to_csv(calibration_path, index=False)
        messagebox.showinfo("Info", "Calibration points saved successfully")

    # ----- Mouse interaction -----

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[view_index(view)]
        color = self.label_colors[self.current_label.get()]
        marker_size = self.marker_size_var.get()

        if event.button == MouseButton.RIGHT:
            if event.key == "shift":
                self.delete_closest_point(ax, event)
            else:
                label = self.current_label.get()
                if self.calibration_points_static[label][view] is not None:
                    self.calibration_points_static[label][view].remove()
                self.calibration_points_static[label][view] = ax.scatter(
                    event.xdata, event.ydata, c=color, s=marker_size * 10, label=label,
                )
                self.canvas.draw()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event)

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return
        if event.button == MouseButton.LEFT:
            self.dragging_point.set_offsets((event.xdata, event.ydata))
            self.canvas.draw()

    def find_closest_point(self, ax, event):
        min_dist = float("inf")
        closest_point = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event):
        min_dist = float("inf")
        closest_label = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = label

        if closest_label:
            self.calibration_points_static[closest_label][self.current_view.get()].remove()
            self.calibration_points_static[closest_label][self.current_view.get()] = None
            self.canvas.draw()
