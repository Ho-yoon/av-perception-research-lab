"""Tests for camera-LiDAR projection."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from perception.fusion.camera_lidar_projection import CameraLidarProjection


def make_projector():
    K = CameraLidarProjection.make_kitti_intrinsic()
    T = np.eye(4)  # identity: lidar = camera frame
    return CameraLidarProjection(K, T, image_width=1280, image_height=720)


class TestProjection:
    def test_point_in_front_projects_to_image(self):
        proj = make_projector()
        # Point directly in front, on optical axis
        K = proj.K
        cx, cy = K[0, 2], K[1, 2]
        z = 10.0
        pts = np.array([[0.0, 0.0, z]])
        pixels, depths, valid = proj.project_points(pts)
        assert valid[0]
        assert abs(pixels[0, 0] - cx) < 2.0
        assert abs(pixels[0, 1] - cy) < 2.0

    def test_point_behind_camera_rejected(self):
        proj = make_projector()
        pts = np.array([[0.0, 0.0, -5.0]])
        _, _, valid = proj.project_points(pts)
        assert not valid[0]

    def test_point_out_of_image_rejected(self):
        proj = make_projector()
        pts = np.array([[1000.0, 0.0, 5.0]])  # Far to the right
        _, _, valid = proj.project_points(pts)
        assert not valid[0]

    def test_depth_map_has_nonzero_entries(self):
        proj = make_projector()
        K = proj.K
        z = 10.0
        pts = np.array([[0.0, 0.0, z]])
        depth_map = proj.create_depth_map(pts)
        assert np.any(depth_map > 0)

    def test_extrinsic_builder(self):
        T = CameraLidarProjection.make_extrinsic_from_rpy_t(0, 0, 0, np.zeros(3))
        assert np.allclose(T, np.eye(4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
