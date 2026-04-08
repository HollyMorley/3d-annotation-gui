"""Tests for the DLT triangulation routine.

Each test projects a known 3D point through a set of pinhole cameras and
checks that triangulate() recovers the original point.
"""

import pytest


def project(P, X):
    """Project a 3D point X (3,) with projection matrix P (3, 4)."""
    import numpy as np
    Xh = np.append(X, 1.0)
    xh = P @ Xh
    return xh[:2] / xh[2]


class TestTriangulate:
    def test_two_views(self):
        np = pytest.importorskip("numpy")
        from annotation_tool.camera.reconstruction import triangulate

        P1 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]], dtype=float)
        P2 = np.array([[1000, 0, 320, -500], [0, 1000, 240, 0], [0, 0, 1, 0]], dtype=float)

        X_true = np.array([1.0, 2.0, 10.0])
        X_est = triangulate([project(P1, X_true), project(P2, X_true)], [P1, P2])

        assert X_est.shape == (4,)
        assert X_est[-1] == pytest.approx(1.0)
        assert X_est[:3] == pytest.approx(X_true, rel=1e-6)

    def test_three_views(self):
        np = pytest.importorskip("numpy")
        from annotation_tool.camera.reconstruction import triangulate

        P1 = np.array([[800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=float)
        P2 = np.array([[800, 0, 320, -400], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=float)
        P3 = np.array([[800, 0, 320, 0], [0, 800, 240, -300], [0, 0, 1, 0]], dtype=float)

        X_true = np.array([-0.5, 1.5, 8.0])
        pts = [project(P, X_true) for P in (P1, P2, P3)]
        X_est = triangulate(pts, [P1, P2, P3])

        assert X_est[:3] == pytest.approx(X_true, rel=1e-6)
