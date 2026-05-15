"""Kalman filter-based multi-object tracker (SORT-like)."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    id: int
    x: np.ndarray          # state: [x, y, z, vx, vy, vz]
    P: np.ndarray          # covariance
    label: str = "unknown"
    age: int = 0
    hits: int = 0
    misses: int = 0


class KalmanTracker:
    def __init__(
        self,
        max_misses: int = 3,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        q_pos: float = 0.5,
        q_vel: float = 1.0,
        r_pos: float = 1.0,
    ):
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id = 0
        self._tracks: list[Track] = []

        n = 6
        self.F = np.eye(n)
        self.F[0, 3] = 1.0  # x += vx * dt (dt=1 here, scaled externally)
        self.F[1, 4] = 1.0
        self.F[2, 5] = 1.0

        self.H = np.zeros((3, n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.Q = np.diag([q_pos]*3 + [q_vel]*3)
        self.R = np.diag([r_pos]*3)

    def update(self, detections: list[np.ndarray]) -> list[Track]:
        """
        detections: list of [x, y, z] positions
        Returns confirmed tracks.
        """
        # Predict all tracks
        for track in self._tracks:
            track.x = self.F @ track.x
            track.P = self.F @ track.P @ self.F.T + self.Q
            track.age += 1

        # Greedy assignment by distance
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        for det_idx in list(unmatched_dets):
            best_track = None
            best_dist = float('inf')
            for i, track in enumerate(self._tracks):
                if i in matched_tracks:
                    continue
                dist = np.linalg.norm(detections[det_idx] - track.x[:3])
                if dist < best_dist and dist < 5.0:
                    best_dist = dist
                    best_track = i

            if best_track is not None:
                self._kf_update(self._tracks[best_track], detections[det_idx])
                self._tracks[best_track].hits += 1
                self._tracks[best_track].misses = 0
                matched_tracks.add(best_track)
                unmatched_dets.remove(det_idx)

        # New tracks for unmatched detections
        for det_idx in unmatched_dets:
            x0 = np.zeros(6)
            x0[:3] = detections[det_idx]
            self._tracks.append(Track(
                id=self._next_id,
                x=x0,
                P=np.eye(6) * 10.0,
            ))
            self._next_id += 1

        # Increment misses for unmatched tracks
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks and track.age > 0:
                track.misses += 1

        # Remove dead tracks
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        # Return confirmed tracks
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def _kf_update(self, track: Track, z: np.ndarray):
        S = self.H @ track.P @ self.H.T + self.R
        K = track.P @ self.H.T @ np.linalg.inv(S)
        innov = z - self.H @ track.x
        track.x = track.x + K @ innov
        track.P = (np.eye(6) - K @ self.H) @ track.P
