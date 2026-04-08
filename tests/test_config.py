"""Tests for config consistency."""

import pytest

from annotation_tool import config


# ---------------------------------------------------------------------------
# flatten / reshape calibration points round-trip
# ---------------------------------------------------------------------------
class TestFlattenReshapeCalibrationPoints:
    def test_round_trip(self):
        np = pytest.importorskip("numpy")
        labels = config.CALIBRATION_LABELS
        original = {
            label: {
                "side": (float(i * 6), float(i * 6 + 1)),
                "front": (float(i * 6 + 2), float(i * 6 + 3)),
                "overhead": (float(i * 6 + 4), float(i * 6 + 5)),
            }
            for i, label in enumerate(labels)
        }

        # Flatten
        flat = []
        for label in labels:
            for view in ["side", "front", "overhead"]:
                if original[label][view] is not None:
                    flat.extend(original[label][view])
        flat = np.array(flat, dtype=float)

        # Reshape
        reshaped = {label: {"side": None, "front": None, "overhead": None} for label in labels}
        idx = 0
        for label in labels:
            for view in ["side", "front", "overhead"]:
                if original[label][view] is not None:
                    reshaped[label][view] = [flat[idx], flat[idx + 1]]
                    idx += 2

        for label in labels:
            for view in ["side", "front", "overhead"]:
                assert reshaped[label][view][0] == pytest.approx(original[label][view][0])
                assert reshaped[label][view][1] == pytest.approx(original[label][view][1])


# ---------------------------------------------------------------------------
# Timestamp zeroing
# ---------------------------------------------------------------------------
class TestZeroTimestamps:
    def test_zeros_relative_to_first(self):
        pd = pytest.importorskip("pandas", reason="pandas required")
        from annotation_tool.gui.sync import zero_timestamps
        df = pd.DataFrame({"Timestamp": [100, 200, 300]})
        df = zero_timestamps(df)
        assert list(df["Timestamp"]) == [0, 100, 200]

    def test_single_timestamp(self):
        pd = pytest.importorskip("pandas", reason="pandas required")
        from annotation_tool.gui.sync import zero_timestamps
        df = pd.DataFrame({"Timestamp": [500]})
        df = zero_timestamps(df)
        assert list(df["Timestamp"]) == [0]


# ---------------------------------------------------------------------------
# Config consistency checks
# ---------------------------------------------------------------------------
class TestConfig:
    def test_optimization_labels_subset_of_body_parts(self):
        for label in config.OPTIMIZATION_REFERENCE_LABELS:
            assert label in config.BODY_PART_LABELS, (
                f"{label} in OPTIMIZATION_REFERENCE_LABELS but not in BODY_PART_LABELS"
            )

    def test_calibration_labels_subset_of_body_parts(self):
        for label in config.CALIBRATION_LABELS:
            assert label in config.BODY_PART_LABELS, (
                f"{label} in CALIBRATION_LABELS but not in BODY_PART_LABELS"
            )

    def test_reference_weights_cover_all_optimization_labels(self):
        for label in config.OPTIMIZATION_REFERENCE_LABELS:
            assert label in config.REFERENCE_LABEL_WEIGHTS, (
                f"Missing weight for {label}"
            )

    def test_marker_size_bounds(self):
        assert config.MIN_MARKER_SIZE < config.MAX_MARKER_SIZE
        assert config.DEFAULT_MARKER_SIZE >= config.MIN_MARKER_SIZE
        assert config.DEFAULT_MARKER_SIZE <= config.MAX_MARKER_SIZE

    def test_contrast_bounds(self):
        assert config.MIN_CONTRAST < config.MAX_CONTRAST
        assert config.DEFAULT_CONTRAST >= config.MIN_CONTRAST
        assert config.DEFAULT_CONTRAST <= config.MAX_CONTRAST

    def test_brightness_bounds(self):
        assert config.MIN_BRIGHTNESS < config.MAX_BRIGHTNESS
        assert config.DEFAULT_BRIGHTNESS >= config.MIN_BRIGHTNESS
        assert config.DEFAULT_BRIGHTNESS <= config.MAX_BRIGHTNESS

    def test_no_duplicate_body_part_labels(self):
        assert len(config.BODY_PART_LABELS) == len(set(config.BODY_PART_LABELS))

    def test_no_duplicate_calibration_labels(self):
        assert len(config.CALIBRATION_LABELS) == len(set(config.CALIBRATION_LABELS))
