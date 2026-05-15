# AV Perception Research Lab

## 1. One-line Summary
Camera, LiDAR, and radar perception pipeline with 3D detection, BEV fusion, object tracking, and ROS2 integration.

## 2. What This Proves
- Camera 2D detection and instance segmentation
- LiDAR point cloud preprocessing (ground removal, clustering)
- Camera-LiDAR extrinsic calibration and projection
- BEV (Bird's Eye View) representation
- Multi-object tracking with Kalman filter
- ROS2 perception node integration
- Evaluation metrics (mAP, MOTA, IoU) on public datasets

## 3. Research Topics Covered

```
1. Camera 2D detection / segmentation
2. LiDAR point cloud object detection (clustering-based)
3. Camera-LiDAR projection and calibration
4. BEV feature fusion
5. Occupancy grid prediction (baseline)
6. Multi-object tracking
7. Evaluation on nuScenes / KITTI
```

## 4. Tech Stack
- Python 3.10+
- PyTorch
- Open3D (point cloud)
- ROS2 Humble (C++ nodes)
- nuScenes, KITTI, Waymo Open Dataset (evaluation)
- GitHub Actions CI

## 5. How to Run

```bash
# Dataset setup
python3 src/datasets/nuscenes_loader.py --data-root /data/nuscenes --version v1.0-mini

# Point cloud visualization
python3 notebooks/pointcloud_visualization.ipynb

# Run full evaluation
python3 scripts/evaluate.py --dataset kitti --model lidar_cluster --split val

# ROS2 inference from rosbag
python3 scripts/infer_rosbag.py --bag data/sample.mcap --model lidar_cluster
```

## 6. Experiments

| Model           | Dataset  | Modality | mAP  | Latency |
|-----------------|----------|----------|------|---------|
| YOLO baseline   | KITTI    | Camera   | 58.2 | 22 ms   |
| LiDAR cluster   | KITTI    | LiDAR    | 61.5 | 15 ms   |
| Late fusion     | nuScenes | Cam+LiDAR| 67.3 | 38 ms   |

## 7. Failure Cases
- Clustering fails with sparse LiDAR returns at long range (>60m)
- Camera detection fails in direct sunlight / night without HDR
- Tracker loses identity during fast lateral movements

## 8. Limitations
- No deep learning LiDAR model (PointPillars placeholder)
- nuScenes mini set only (not full 700 scenes)
- BEV fusion is late fusion only (no end-to-end BEV backbone)
