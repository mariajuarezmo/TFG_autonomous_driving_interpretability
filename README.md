# ANAID Dataset

**Autonomous Naturalistic Avoidance Interaction Dataset**  
Dataset of real driving data for PilotNet neural network training.

---

## Overview

This dataset was built from real driving data collected with an instrumented vehicle (Hyundai Tucson Hybrid equipped with a Comma-3X autonomous-driving development kit). The raw data consists of segmented videos and CAN-bus telemetry logs. The goal of the construction pipeline is to unify all segments of each recording session into a coherent, temporally aligned structure ready for neural network training.

The dataset is organized into **four independent recording sessions**: `run1`, `run2`, `run3`, and `run4`, each corresponding to a distinct driving session.

---

## Dataset Structure

\```
Dataset/
├── run1/
│   ├── telemetry_data/
│   │   ├── telemetry.csv
│   │   └── frame-torque.txt
│   └── video_data/
│       ├── frame_videos/
│       │   ├── 00000.jpg
│       │   ├── 00001.jpg
│       │   └── ...
│       ├── evasion_driving_videos/
│       │   ├── evasion_001/
│       │   │   ├── evasion_001.mp4
│       │   │   ├── evasion_001.csv
│       │   │   └── frames/
│       │   │       ├── 00000.jpg
│       │   │       └── ...
│       │   ├── evasion_002/
│       │   └── ...
│       └── normal_driving_videos/
│           ├── normal_001/
│           │   ├── normal_001.mp4
│           │   ├── normal_001.csv
│           │   └── frames/
│           │       ├── 00000.jpg
│           │       └── ...
│           ├── normal_002/
│           └── ...
├── run2/  (same structure)
├── run3/  (same structure)
└── run4/  (same structure)
\```

---

## telemetry_data/

Contains the unified CAN-bus telemetry data for the entire recording session.

### telemetry.csv

A CSV file with **one row per video frame**. The first column, `frame_id`, identifies the frame by its filename (e.g., `00000.jpg`) and directly links each telemetry row to its corresponding image in `video_data/frame_videos/`. The remaining columns contain vehicle state parameters from the CAN bus, temporally synchronized to each frame using `merge_asof`.

| Column | Description |
|---|---|
| `frame_id` | Frame filename associated with this row (e.g., `00000.jpg`) |
| `vEgo` | Estimated vehicle speed (m/s) |
| `gas` | Accelerator position (normalized value) |
| `brake` | Brake position (normalized value) |
| `steeringAngleDeg` | Steering wheel angle (degrees) |
| `steeringTorque` | Steering torque applied (Nm) |
| `aEgo` | Longitudinal vehicle acceleration (m/s²) |
| `yawRate` | Yaw rate (rad/s) |
| `gearShifter` | Current gear shifter position |
| `steeringRateDeg` | Steering wheel rotation speed (°/s) |
| `vEgoRaw` | Raw unfiltered vehicle speed (m/s) |
| `standstill` | Boolean indicator: vehicle stopped |
| `leftBlinker` | Boolean indicator: left turn signal active |
| `rightBlinker` | Boolean indicator: right turn signal active |
| `gasPressed` | Boolean indicator: accelerator pressed |
| `brakePressed` | Boolean indicator: brake pressed |
| `steeringPressed` | Boolean indicator: steering wheel intervention |

### frame-torque.txt

A plain text file associating each frame with its synchronized steering torque value (`steeringTorque`) from the CAN bus. Each line follows the format:

\```
<frame_filename> <torque_value>
\```

Example:
\```
00000.jpg 98.0
00001.jpg 98.0
00002.jpg 99.0
\```

Only frames with a valid torque value are included, synchronized using nearest-neighbor time matching (`merge_asof` with a tolerance of 0.1 seconds).

---

## video_data/

Contains all videos and frames of the session, organized into three subfolders by driving type.

### frame_videos/

Contains all frames extracted from the session's videos, treated as a single continuous video. Frames from all segments were concatenated in chronological order and globally renumbered as `00000.jpg`, `00001.jpg`, ..., `NNNNN.jpg`. Frames were extracted at **20 FPS**.

### evasion_driving_videos/

Contains video clips corresponding to **evasive maneuvers** automatically detected by analyzing the steering angle signal (`steeringAngleDeg`) from the CAN bus. Each clip is stored in its own subfolder named `evasion_NNN` (numbered sequentially within each run, restarting at 001 for each run).

Each subfolder contains:

| File | Description |
|---|---|
| `evasion_NNN.mp4` | Video clip covering the evasive maneuver |
| `evasion_NNN.csv` | CAN-bus telemetry synchronized to the clip, one row per frame |
| `frames/` | Individual frames extracted from the clip, numbered from `00000.jpg` |

The `frame_id` column in `evasion_NNN.csv` matches exactly the filenames in `frames/`, making each clip fully self-contained.

### normal_driving_videos/

Contains video clips corresponding to **normal driving**, i.e., the time intervals of each original video not classified as evasive maneuvers. Clips shorter than 2 seconds are discarded. Each clip is stored in its own subfolder named `normal_NNN` with the same internal structure:

| File | Description |
|---|---|
| `normal_NNN.mp4` | Video clip of normal driving |
| `normal_NNN.csv` | CAN-bus telemetry synchronized to the clip, one row per frame |
| `frames/` | Individual frames extracted from the clip, numbered from `00000.jpg` |

---

## File Summary

| File / Folder | Description |
|---|---|
| `telemetry_data/telemetry.csv` | One row per frame with all CAN-bus parameters (global) |
| `telemetry_data/frame-torque.txt` | Frame → synchronized steering torque mapping (global) |
| `video_data/frame_videos/` | Individual frames (.jpg), globally renumbered |
| `video_data/evasion_driving_videos/evasion_NNN/evasion_NNN.mp4` | Evasion video clip |
| `video_data/evasion_driving_videos/evasion_NNN/evasion_NNN.csv` | Per-frame telemetry for this clip |
| `video_data/evasion_driving_videos/evasion_NNN/frames/` | Frames extracted from this clip |
| `video_data/normal_driving_videos/normal_NNN/normal_NNN.mp4` | Normal driving video clip |
| `video_data/normal_driving_videos/normal_NNN/normal_NNN.csv` | Per-frame telemetry for this clip |
| `video_data/normal_driving_videos/normal_NNN/frames/` | Frames extracted from this clip |

---

## Construction Pipeline

The dataset was generated using Python scripts from two raw data sources:

- **Segmented videos (.ts):** approximately one-minute video files recorded with the vehicle's front-facing camera at 20 FPS.
- **Telemetry logs (.json / .jsonl):** CAN-bus log files, one per video segment, containing high-frequency vehicle state data.

The construction pipeline consisted of five main stages:

1. **Frame extraction:** all frames were extracted from each video segment and stored as individual JPEG files in `frame_videos/`.
2. **Telemetry extraction:** telemetry logs were parsed to obtain `carState` parameters and exported to per-segment CSV files.
3. **Unification and synchronization:** frames and telemetry from all segments of each session were concatenated in chronological order, globally renumbered, and temporally synchronized using `merge_asof`.
4. **Maneuver detection and extraction:** the steering angle signal was analyzed to automatically detect evasive maneuvers using hysteresis and peak detection. Evasion clips were extracted to `evasion_driving_videos/` and normal driving clips to `normal_driving_videos/`.
5. **Clip enrichment and renaming:** for each clip, CAN-bus telemetry rows within the clip's time interval were synchronized to each frame at 20 FPS using `merge_asof` and saved as a per-clip CSV. Frames were extracted individually using ffmpeg. Clips without an associated telemetry log were automatically discarded. All clips were finally renamed to sequential descriptive names (`evasion_NNN`, `normal_NNN`), with numbering restarting at 001 for each run.