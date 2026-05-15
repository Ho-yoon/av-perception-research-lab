# Data

This directory contains dataset manifests and sample data descriptions.
Large files (rosbag, pointcloud, model checkpoints) are NOT committed to git.

## How to obtain data

See `docs/datasets.md` for download instructions.

## Git LFS

This repository uses Git LFS for:
- `*.bag` — ROS1 bags
- `*.mcap` — ROS2 bags
- `*.db3` — ROS2 bag SQLite
- `*.onnx`, `*.pth`, `*.pt` — model weights

Install: `git lfs install && git lfs pull`
