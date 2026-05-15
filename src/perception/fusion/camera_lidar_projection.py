"""Camera-LiDAR projection utilities."""
import numpy as np
from typing import Optional


class CameraLidarProjection:
    """
    Projects LiDAR points onto camera image plane.
    Requires extrinsic (T_cam_lidar) and intrinsic (K) calibration.
    """

    def __init__(
        self,
        K: np.ndarray,           # 3x3 camera intrinsic matrix
        T_cam_lidar: np.ndarray, # 4x4 extrinsic: lidar→camera transform
        image_width: int,
        image_height: int,
        min_depth: float = 0.5,
        max_depth: float = 60.0,
    ):
        self.K = K
        self.T = T_cam_lidar
        self.W = image_width
        self.H = image_height
        self.min_depth = min_depth
        self.max_depth = max_depth

    def project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points (N, 3) onto image.
        Returns:
            pixel_coords: (M, 2) [u, v] valid pixel coordinates
            depths: (M,) depth values
            valid_mask: (N,) boolean mask of valid projections
        """
        N = len(points)
        pts_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)

        # Transform to camera frame
        pts_cam = (self.T @ pts_h.T).T  # (N, 4)
        depths = pts_cam[:, 2]

        # Filter by depth
        valid = (depths >= self.min_depth) & (depths <= self.max_depth)

        pts_valid = pts_cam[valid]
        depths_valid = depths[valid]

        # Project to image
        pts_2d_h = (self.K @ pts_valid[:, :3].T).T  # (M, 3)
        u = pts_2d_h[:, 0] / pts_2d_h[:, 2]
        v = pts_2d_h[:, 1] / pts_2d_h[:, 2]

        # Filter by image bounds
        in_image = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        pixel_coords = np.stack([u[in_image], v[in_image]], axis=1)

        # Update valid mask
        valid[valid] = in_image

        return pixel_coords, depths_valid[in_image], valid

    def create_depth_map(self, points: np.ndarray) -> np.ndarray:
        """Create a sparse depth map image (H, W) from LiDAR points."""
        depth_map = np.zeros((self.H, self.W), dtype=np.float32)
        pixels, depths, _ = self.project_points(points)

        for (u, v), d in zip(pixels.astype(int), depths):
            if 0 <= v < self.H and 0 <= u < self.W:
                if depth_map[v, u] == 0 or d < depth_map[v, u]:
                    depth_map[v, u] = d

        return depth_map

    @staticmethod
    def make_kitti_intrinsic(fx=721.54, fy=721.54, cx=609.56, cy=172.85) -> np.ndarray:
        """Example KITTI camera intrinsic matrix."""
        return np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

    @staticmethod
    def make_extrinsic_from_rpy_t(
        r: float, p: float, y: float, t: np.ndarray
    ) -> np.ndarray:
        """Build 4x4 extrinsic from roll-pitch-yaw and translation."""
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy_, sy = np.cos(y), np.sin(y)

        R = np.array([
            [cy_*cp, cy_*sp*sr - sy*cr, cy_*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy_*cr, sy*sp*cr - cy_*sr],
            [-sp,    cp*sr,              cp*cr],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
