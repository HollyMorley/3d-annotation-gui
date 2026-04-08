# 3D Annotation GUI

### Multi-camera labelling tool with 3D projection assistance

A tkinter-based GUI for manual annotation of body parts across three synchronized camera views (side, front, overhead),
with real-time 3D projection lines to guide labelling.

_This is a personal research tool. Future work will focus on generalising the pipeline so it can be released as a
reusable package._

## Setup

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
│   └── reconstruction.py    # CameraData, BeltPoints, 3D geometry
└── gui/
    ├── app.py               # Main menu window
    ├── base.py              # Shared base class (pan, zoom, crosshair, sliders)
    ├── extract.py           # Frame extraction from synchronized videos
    ├── calibrate.py         # Camera calibration point labelling
    ├── label.py             # Body part labelling with 3D projections
    ├── utils.py             # Pure utility functions
    └── sync.py              # Timestamp synchronization and frame matching
tests/
├── test_config.py           # Config consistency checks
└── test_gui_utils.py        # Utility function tests
```

## Tools

### Extract Frames

Select a video file and scroll through synchronized multi-camera footage to extract frame trios for labelling.
Timestamps are used to correct frame misalignment across cameras with different frame rates.

### Calibrate Cameras

Label known landmarks (belt corners, step edges, door marker) across all three camera views for camera pose estimation
via OpenCV's solvePnP.

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move.

### Label Body Parts

Label anatomical landmarks with real-time 3D projection lines displayed across views to guide placement.

- **Label View** — select the camera view to label in
- **Projection View** — select the view to calculate projection lines from
- **Spacer Lines** — right-click two points to display 12 equally-spaced guide lines
- **Optimize Calibration** — minimize reprojection error to improve projection estimates

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move, Hover for label name.
