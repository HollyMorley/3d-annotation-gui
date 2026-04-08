"""Main application window with tool selection menu."""

import tkinter as tk


class MainTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Parts Labeling Tool")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20, padx=30)

        tk.Button(main_frame, text="Extract Frames from Videos",
                  command=self.extract_frames_menu).pack(pady=5)
        tk.Button(main_frame, text="Calibrate Camera Positions",
                  command=self.calibrate_cameras_menu).pack(pady=5)
        tk.Button(main_frame, text="Label Frames",
                  command=self.label_frames_menu).pack(pady=5)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def extract_frames_menu(self):
        from annotation_tool.gui.extract import ExtractFramesTool
        self.clear_root()
        ExtractFramesTool(self.root, self)

    def calibrate_cameras_menu(self):
        from annotation_tool.gui.calibrate import CalibrateCamerasTool
        self.clear_root()
        CalibrateCamerasTool(self.root, self)

    def label_frames_menu(self):
        from annotation_tool.gui.label import LabelFramesTool
        self.clear_root()
        LabelFramesTool(self.root, self)
