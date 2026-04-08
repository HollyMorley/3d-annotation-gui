# 3D Annotation GUI

### Multi-camera labelling tool with 3D projection assistance

A tkinter-based GUI for manual annotation of body parts across three synchronized camera views (side, front, overhead),
with real-time 3D projection lines to guide labelling.

<p align="center">
  <img src="docs/gui_screenshot_edited.png" alt="Annotation GUI screenshot" width="100%"><br>
  <em>The labelling interface, showing the three synchronized camera views and projection lines used to guide placement.</em>
</p>

<p align="center">
  <img src="docs/labelling_demo.gif" alt="3D reconstruction of a running mouse" width="100%"><br>
  <em>Downstream result: labels produced with this tool were used to train DeepLabCut models (single model per camera). These multi-view predictions are then triangulated to reconstruct 3D motion.</em>
</p>

_Note: This is a personal research tool. Future work will focus on generalising the pipeline so it can be released as a
reusable package._

## Setup

**Sample data:** a Dropbox folder with an example file structure and a pre-made calibration file is available
[here](https://www.dropbox.com/scl/fo/ifppb8f3ss8z1ijvun3ry/AAAgY5bUVrrvuz4W0q-rYWA?rlkey=1tjdthbuqur7kuqb4a1t0bcza&dl=0).
Update the paths in `annotation_tool/config.py` to match your local directory structure after downloading.

Create environment:

```bash
conda env create -f environment.yml
```

Run the GUI:

```bash
# First set the "dir" path in annotation_tool/config.py to point to your data directory.

conda activate 3d-annotation-gui
python -m annotation_tool
```

Run tests:

```bash
pytest tests/ -v
```

## Project structure

```
annotation_tool/
├── __main__.py              # Entry point (python -m annotation_tool)
├── config.py                # User settings, labels, paths
├── camera/
│   ├── calibration.py       # BasicCalibration wrapper
│   └── reconstruction.py    # CameraData, BeltPoints, DLT triangulation
└── gui/
    ├── app.py               # Main menu window
    ├── base.py              # Shared base class (pan, zoom, crosshair, sliders)
    ├── extract.py           # Frame extraction from synchronized videos
    ├── calibrate.py         # Camera calibration point labelling
    ├── label.py             # Body part labelling with 3D projections
    ├── sync.py              # Timestamp synchronization and frame matching
    └── utils.py             # Pure utility functions
tests/
├── test_config.py           # Config consistency checks
├── test_gui_utils.py        # Utility function tests
└── test_triangulation.py    # DLT triangulation tests
```

## Data structure

The tool expects the following layout under the directory set as `dir` in `annotation_tool/config.py`
(this is how the sample dataset is organised):

```
<data_dir>/
├── <session_name>/                          # Raw videos and timestamps for one recording
│   ├── <name>_side_<n>.avi
│   ├── <name>_side_<n>_Timestamps.csv
│   ├── <name>_front_<n>.avi
│   ├── <name>_front_<n>_Timestamps.csv
│   ├── <name>_overhead_<n>.avi
│   └── <name>_overhead_<n>_Timestamps.csv
├── CameraCalibration/
│   ├── default_calibration_labels.csv       # Fallback calibration points
│   └── <video_name>/
│       └── calibration_labels.csv           # Per-video calibration points
├── Side/
│   └── <video_name>/                        # Extracted frames + body part labels
├── Front/
│   └── <video_name>/
└── Overhead/
    └── <video_name>/
```

- `CameraCalibration/`, `Side/`, `Front/`, and `Overhead/` are created automatically by the tool as
  you calibrate, extract, and label.
- The raw videos and their timestamp CSVs live together in a session folder (three `.avi` files with
  matching `_Timestamps.csv` files, one per camera view).

## Tools

### Extract Frames

_Pick synchronized video frames to be labelled._

- Choose a video file from the pop-up file manager.
- Scroll or skip through the synchronized videos and extract frame trios for labelling.
- Timestamps are used to correct for any frame misalignment across cameras.

### Calibrate Cameras

_Label known landmarks across all three camera views for camera pose estimation (via OpenCV's `solvePnP`)._

- Choose a video file from the pop-up file manager.
- Scroll through the video and label the calibration points in each camera view.
- Calibration landmarks: the 4 corners of the first belt, the corners of the starting step edge, and the `x`
  sticker on the door.

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move.

### Label Body Parts

_Label pre-defined body parts, with 3D projection estimates of each point displayed across views to guide placement._

- Choose a video folder from the pop-up file manager (extracted frames and calibration labels must both already
  exist for that video).
- **Label View** — the camera view to label in.
- **Projection View** — the view to calculate projection lines from. If labels are present in the selected
  Projection View, projection lines (from the camera centre, crossing through the platform edges) are drawn on
  the other two views to guide labelling.
- **Spacer Lines** — click once, then right-click two points on the active frame to display 12 equally-spaced
  vertical guide lines along the x-axis.
- **Optimize Calibration** — adjusts the manually labelled calibration points to minimize reprojection error
  between camera views, improving the projection estimates.
- **Save Labels** — saves the labels to the video folder under the respective camera names (`Side`, `Front`,
  `Overhead`).

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move, Hover for label name.
