"""Pure utility functions shared across GUI tools."""

import os
import time

VIEWS = ["side", "front", "overhead"]


def view_index(view):
    """Return the axis index (0, 1, 2) for a camera view name."""
    return VIEWS.index(view)


def get_video_name_with_view(video_name, view):
    """Insert camera view name before the final segment of a video name."""
    split_video_name = video_name.split("_")
    split_video_name.insert(-1, view)
    return "_".join(split_video_name)


def parse_video_path(video_path):
    """Extract (name, date, camera_view) from a video file path."""
    video_file = os.path.basename(video_path)
    parts = video_file.split("_")
    date = parts[1]
    camera_view = [p for p in parts if p in VIEWS][0]
    name_parts = [p for p in parts if p not in VIEWS]
    name = "_".join(name_parts).replace(".avi", "")
    return name, date, camera_view


def get_corresponding_video_path(video_path, current_view, target_view):
    """Get the path to the same video from a different camera view."""
    base_path = os.path.dirname(video_path)
    video_file = os.path.basename(video_path)
    corresponding_file = video_file.replace(current_view, target_view).replace(".avi", "")
    return os.path.join(base_path, f"{corresponding_file}.avi")


def extract_date_from_folder_path(folder_path):
    """Find an 8-digit date string in a folder path."""
    parts = folder_path.split(os.sep)
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return part
    return None


def rgb_to_hex(color):
    """Convert an (r, g, b, ...) tuple with 0-1 floats to a hex color string."""
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )


def generate_label_colors(labels, calibration_labels=None):
    """Generate a distinct color for each label using the HSV colormap.

    If calibration_labels is provided, those labels get white and the colormap
    is distributed only across the remaining body-part labels.
    """
    from matplotlib import pyplot as plt
    colormap = plt.get_cmap("hsv")
    if calibration_labels:
        body_part_labels = [l for l in labels if l not in calibration_labels]
        colors = [colormap(i / len(body_part_labels)) for i in range(len(body_part_labels))]
        label_colors = {}
        color_idx = 0
        for label in labels:
            if label in calibration_labels:
                label_colors[label] = "#ffffff"
            else:
                label_colors[label] = rgb_to_hex(colors[color_idx])
                color_idx += 1
        return label_colors
    else:
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: rgb_to_hex(color) for label, color in zip(labels, colors)}


def apply_contrast_brightness(frame, contrast, brightness):
    """Apply contrast and brightness adjustments to a BGR frame."""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrast = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(img_contrast)
    img_brightness = enhancer.enhance(brightness)
    return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)


def find_t_for_coordinate(val, coord_index, point_3d, camera_center):
    """Find the parametric t value where the line through two 3D points
    reaches a given coordinate value along one axis."""
    coords = list(zip(point_3d, camera_center))
    if coord_index not in (0, 1, 2):
        raise ValueError("coord_index must be 0 (x), 1 (y), or 2 (z)")
    p1, p2 = coords[coord_index]
    return (val - p1) / (p2 - p1)


def get_line_equation(point_3d, camera_center):
    """Return a closure that computes 3D coordinates along the line
    from point_3d (t=0) to camera_center (t=1)."""
    x1, y1, z1 = point_3d
    x2, y2, z2 = camera_center

    def line_at_t(t):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        return x, y, z

    return line_at_t


def debounce(wait):
    """Decorator that suppresses calls within `wait` seconds of the last call."""
    def decorator(fn):
        last_call = [0]

        def debounced(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                last_call[0] = current_time
                return fn(*args, **kwargs)

        return debounced

    return decorator
