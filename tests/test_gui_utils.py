"""Tests for the pure utility functions in annotation_tool/gui/utils.py."""

import os
import time
import pytest

from annotation_tool.gui.utils import (
    get_video_name_with_view, parse_video_path, rgb_to_hex,
    extract_date_from_folder_path, find_t_for_coordinate, get_line_equation,
    debounce,
)


class TestGetVideoNameWithView:
    def test_inserts_view_before_last_segment(self):
        assert get_video_name_with_view("HM_20220801_1", "side") == "HM_20220801_side_1"

    def test_front_view(self):
        assert get_video_name_with_view("HM_20220801_1", "front") == "HM_20220801_front_1"

    def test_overhead_view(self):
        assert get_video_name_with_view("HM_20220801_1", "overhead") == "HM_20220801_overhead_1"

    def test_single_segment_name(self):
        assert get_video_name_with_view("video", "side") == "side_video"

    def test_many_underscores(self):
        assert get_video_name_with_view("A_B_C_D_E", "front") == "A_B_C_D_front_E"


class TestParseVideoPath:
    def test_side_video(self):
        name, date, view = parse_video_path("HM_20220801_side_1.avi")
        assert name == "HM_20220801_1"
        assert date == "20220801"
        assert view == "side"

    def test_front_video(self):
        name, date, view = parse_video_path("HM_20220801_front_1.avi")
        assert name == "HM_20220801_1"
        assert view == "front"

    def test_overhead_video(self):
        name, date, view = parse_video_path("HM_20220801_overhead_1.avi")
        assert name == "HM_20220801_1"
        assert view == "overhead"

    def test_strips_avi_extension(self):
        name, _, _ = parse_video_path("HM_20220801_side_1.avi")
        assert ".avi" not in name

    def test_no_extension(self):
        name, date, view = parse_video_path("HM_20220801_side_1")
        assert name == "HM_20220801_1"
        assert view == "side"

    def test_no_matching_view_raises(self):
        with pytest.raises(IndexError):
            parse_video_path("HM_20220801_unknown_1.avi")


class TestRgbToHex:
    def test_white(self):
        assert rgb_to_hex((1.0, 1.0, 1.0, 1.0)) == "#ffffff"

    def test_black(self):
        assert rgb_to_hex((0.0, 0.0, 0.0, 1.0)) == "#000000"

    def test_red(self):
        assert rgb_to_hex((1.0, 0.0, 0.0, 1.0)) == "#ff0000"

    def test_mid_gray(self):
        assert rgb_to_hex((0.5, 0.5, 0.5, 1.0)) == "#7f7f7f"


class TestExtractDateFromFolderPath:
    def test_finds_date(self):
        path = os.sep.join(["C:", "data", "20220801", "video"])
        assert extract_date_from_folder_path(path) == "20220801"

    def test_no_date_returns_none(self):
        path = os.sep.join(["C:", "data", "videos"])
        assert extract_date_from_folder_path(path) is None

    def test_ignores_short_numbers(self):
        path = os.sep.join(["C:", "data", "12345", "video"])
        assert extract_date_from_folder_path(path) is None

    def test_returns_first_matching(self):
        path = os.sep.join(["C:", "20220101", "20220802", "video"])
        assert extract_date_from_folder_path(path) == "20220101"


class TestFindTForCoordinate:
    def test_x_coordinate(self):
        assert find_t_for_coordinate(5, 0, (0, 0, 0), (10, 10, 10)) == pytest.approx(0.5)

    def test_y_coordinate(self):
        assert find_t_for_coordinate(5, 1, (0, 0, 0), (10, 10, 10)) == pytest.approx(0.5)

    def test_z_coordinate(self):
        assert find_t_for_coordinate(5, 2, (0, 0, 0), (10, 10, 10)) == pytest.approx(0.5)

    def test_invalid_coord_index(self):
        with pytest.raises(ValueError):
            find_t_for_coordinate(5, 3, (0, 0, 0), (10, 10, 10))

    def test_endpoint_t_zero(self):
        assert find_t_for_coordinate(0, 0, (0, 0, 0), (10, 10, 10)) == pytest.approx(0.0)

    def test_endpoint_t_one(self):
        assert find_t_for_coordinate(10, 0, (0, 0, 0), (10, 10, 10)) == pytest.approx(1.0)


class TestGetLineEquation:
    def test_at_t0_returns_point(self):
        assert get_line_equation((1, 2, 3), (4, 5, 6))(0) == (1, 2, 3)

    def test_at_t1_returns_camera_center(self):
        assert get_line_equation((1, 2, 3), (4, 5, 6))(1) == (4, 5, 6)

    def test_midpoint(self):
        assert get_line_equation((0, 0, 0), (10, 10, 10))(0.5) == (5, 5, 5)


class TestDebounce:
    def test_first_call_executes(self):
        call_count = [0]

        @debounce(0.5)
        def fn():
            call_count[0] += 1

        fn()
        assert call_count[0] == 1

    def test_rapid_calls_are_suppressed(self):
        call_count = [0]

        @debounce(10)
        def fn():
            call_count[0] += 1

        fn()
        fn()
        assert call_count[0] == 1

    def test_calls_after_wait_execute(self):
        call_count = [0]

        @debounce(0.05)
        def fn():
            call_count[0] += 1

        fn()
        time.sleep(0.1)
        fn()
        assert call_count[0] == 2
