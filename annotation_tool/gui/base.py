"""Base class with shared UI logic for CalibrateCamerasTool and LabelFramesTool."""

import tkinter as tk

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from annotation_tool import config
from annotation_tool.gui.utils import VIEWS, view_index, apply_contrast_brightness


class BaseAnnotationTool:
    """Shared state and interaction handlers for multi-view annotation tools.

    Subclasses must implement:
        _refresh_display()  — redraw the current frame content
        on_click(event)     — handle mouse clicks for placing/selecting points
        on_drag(event)      — handle mouse drag for moving points
        skip_frames(step)   — navigate forward/backward through frames
    """

    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.current_view = tk.StringVar(value="side")
        self.crosshair_lines = []
        self.dragging_point = None
        self.panning = False
        self.pan_start = None
        self.fig = None
        self.axs = None
        self.canvas = None
        self.current_frame_index = 0
        self.matched_frames = []
        self.frame_label = None

    # ----- UI construction helpers -----

    def create_settings_controls(self, parent):
        """Build marker size, contrast, and brightness sliders."""
        settings_frame = tk.Frame(parent)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size").pack(side=tk.LEFT, padx=5)
        tk.Scale(
            settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE,
            orient=tk.HORIZONTAL, resolution=config.MARKER_SIZE_STEP,
            variable=self.marker_size_var, command=self.update_marker_size,
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(
            settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST,
            orient=tk.HORIZONTAL, resolution=config.CONTRAST_STEP,
            variable=self.contrast_var, command=self.update_contrast_brightness,
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(
            settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS,
            orient=tk.HORIZONTAL, resolution=config.BRIGHTNESS_STEP,
            variable=self.brightness_var, command=self.update_contrast_brightness,
        ).pack(side=tk.LEFT, padx=5)

        return settings_frame

    def create_3panel_canvas(self, parent, figsize=(10, 12)):
        """Create a 3-row matplotlib figure embedded in tkinter."""
        self.fig, self.axs = plt.subplots(3, 1, figsize=figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def connect_mouse_events(self):
        """Wire up all shared mouse event handlers to the canvas."""
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def add_skip_buttons(self, parent):
        """Add frame skip buttons (+-1, +-10, +-100, +-1000)."""
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000),
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def display_three_views(self, frame_side, frame_front, frame_overhead):
        """Clear axes and display three BGR frames with contrast/brightness applied."""
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        frames = [
            apply_contrast_brightness(frame_side, contrast, brightness),
            apply_contrast_brightness(frame_front, contrast, brightness),
            apply_contrast_brightness(frame_overhead, contrast, brightness),
        ]
        titles = ["Side View", "Front View", "Overhead View"]

        for ax, frame, title in zip(self.axs, frames, titles):
            ax.cla()
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(title)

    # ----- Mouse interaction handlers -----

    def on_scroll(self, event):
        if event.inaxes:
            ax = self.axs[view_index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            if xdata is not None and ydata is not None:
                zoom_factor = 0.9 if event.button == "up" else 1.1
                ax.set_xlim([xdata + (x - xdata) * zoom_factor for x in xlim])
                ax.set_ylim([ydata + (y - ydata) * zoom_factor for y in ylim])
                self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 2:
            self.panning = False
            self.pan_start = None

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)
            ax = self.axs[view_index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]
            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.canvas.draw_idle()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color="cyan", linestyle="--", linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color="cyan", linestyle="--", linewidth=0.5))
            self.canvas.draw_idle()

    # ----- Controls that delegate to subclass -----

    def update_marker_size(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self._refresh_display()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def update_contrast_brightness(self, val):
        self._refresh_display()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self._refresh_display()

    # ----- Abstract methods -----

    def _refresh_display(self):
        raise NotImplementedError

    def on_click(self, event):
        raise NotImplementedError

    def on_drag(self, event):
        raise NotImplementedError

    def skip_frames(self, step):
        raise NotImplementedError
