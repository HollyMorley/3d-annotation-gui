"""Tests for the DLT triangulation routine."""

import pytest


class TestTriangulate:
    """Verify that the linear (DLT) triangulation recovers known 3D points.

    A minimal smoke test uses two pinhole cameras with known projection
    matrices, synthesizes image points by projecting a ground-truth 3D point,
    then checks that triangulate() recovers the original point.
    """

    def _project(self, P, X):
        """Project a 3D point X (3,) with projection matrix P (3, 4)."""
        np = pytest.importorskip("numpy")
        Xh = np.append(X, 1.0)
        xh = P @ Xh
        return xh[:2] / xh[2]

    def test_recovers_point_from_two_views(self):
        np = pytest.importorskip("numpy")
        from annotation_tool.camera.reconstruction import triangulate

        # Two canonical cameras: one at the origin, one translated along x.
        P1 = np.array([
            [1000, 0, 320, 0],
            [0, 1000, 240, 0],
            [0, 0, 1, 0],
        ], dtype=float)
        P2 = np.array([
            [1000, 0, 320, -500],
            [0, 1000, 240, 0],
            [0, 0, 1, 0],
        ], dtype=float)

        X_true = np.array([1.0, 2.0, 10.0])
        x1 = self._project(P1, X_true)
        x2 = self._project(P2, X_true)

        X_est = triangulate([x1, x2], [P1, P2])
        assert X_est.shape == (4,)
        assert X_est[-1] == pytest.approx(1.0)
        assert X_est[:3] == pytest.approx(X_true, rel=1e-6)

    def test_recovers_point_from_three_views(self):
        np = pytest.importorskip("numpy")
        from annotation_tool.camera.reconstruction import triangulate

        P1 = np.array([
            [800, 0, 320, 0],
            [0, 800, 240, 0],
            [0, 0, 1, 0],
        ], dtype=float)
        P2 = np.array([
            [800, 0, 320, -400],
            [0, 800, 240, 0],
            [0, 0, 1, 0],
        ], dtype=float)
        P3 = np.array([
            [800, 0, 320, 0],
            [0, 800, 240, -300],
            [0, 0, 1, 0],
        ], dtype=float)

        X_true = np.array([-0.5, 1.5, 8.0])
        pts = [self._project(P, X_true) for P in (P1, P2, P3)]

        X_est = triangulate(pts, [P1, P2, P3])
        assert X_est[:3] == pytest.approx(X_true, rel=1e-6)
