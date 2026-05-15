"""LiDAR point cloud preprocessing: ground removal, voxel downsampling, clustering."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import open3d as o3d
    OPEN3D = True
except ImportError:
    OPEN3D = False


@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    score: float = 1.0
    label: str = "unknown"


class PointCloudPreprocessor:
    def __init__(
        self,
        voxel_size: float = 0.2,
        ground_z_threshold: float = -1.5,
        roi_x: tuple[float, float] = (-50.0, 50.0),
        roi_y: tuple[float, float] = (-50.0, 50.0),
        roi_z: tuple[float, float] = (-3.0, 5.0),
    ):
        self.voxel_size = voxel_size
        self.ground_z = ground_z_threshold
        self.roi = {"x": roi_x, "y": roi_y, "z": roi_z}

    def process(self, points: np.ndarray) -> np.ndarray:
        """
        points: (N, 3+) array [x, y, z, intensity?, ...]
        Returns filtered points (M, 3).
        """
        pts = points[:, :3]
        pts = self._roi_filter(pts)
        pts = self._ground_removal(pts)
        pts = self._voxel_downsample(pts)
        return pts

    def _roi_filter(self, pts: np.ndarray) -> np.ndarray:
        mask = (
            (pts[:, 0] >= self.roi["x"][0]) & (pts[:, 0] <= self.roi["x"][1]) &
            (pts[:, 1] >= self.roi["y"][0]) & (pts[:, 1] <= self.roi["y"][1]) &
            (pts[:, 2] >= self.roi["z"][0]) & (pts[:, 2] <= self.roi["z"][1])
        )
        return pts[mask]

    def _ground_removal(self, pts: np.ndarray) -> np.ndarray:
        return pts[pts[:, 2] > self.ground_z]

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts

        if OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            down = pcd.voxel_down_sample(self.voxel_size)
            return np.asarray(down.points)

        # Fallback: numpy-based voxel grid
        voxel_indices = np.floor(pts / self.voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return pts[unique_idx]


class EuclideanClusterer:
    def __init__(
        self,
        eps: float = 0.8,
        min_points: int = 5,
        max_cluster_size: int = 5000,
        min_cluster_size: int = 5,
    ):
        self.eps = eps
        self.min_points = min_points
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size

    def cluster(self, points: np.ndarray) -> list[np.ndarray]:
        """Return list of point clusters."""
        if len(points) < self.min_points:
            return []

        if OPEN3D:
            return self._open3d_cluster(points)
        return self._numpy_cluster(points)

    def _open3d_cluster(self, points: np.ndarray) -> list[np.ndarray]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.array(pcd.cluster_dbscan(
            eps=self.eps, min_points=self.min_points, print_progress=False))

        clusters = []
        for label in set(labels):
            if label < 0:
                continue
            cluster = points[labels == label]
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(cluster)
        return clusters

    def _numpy_cluster(self, points: np.ndarray) -> list[np.ndarray]:
        """Simplified DBSCAN without open3d."""
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_points).fit(points[:, :2])
        clusters = []
        for label in set(db.labels_):
            if label < 0:
                continue
            cluster = points[db.labels_ == label]
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(cluster)
        return clusters

    def clusters_to_boxes(self, clusters: list[np.ndarray]) -> list[BoundingBox3D]:
        boxes = []
        for cluster in clusters:
            center = cluster.mean(axis=0)
            extents = cluster.max(axis=0) - cluster.min(axis=0)
            boxes.append(BoundingBox3D(
                x=center[0], y=center[1], z=center[2],
                length=max(extents[0], 0.5),
                width=max(extents[1], 0.5),
                height=max(extents[2], 0.3),
                yaw=0.0,
            ))
        return boxes
