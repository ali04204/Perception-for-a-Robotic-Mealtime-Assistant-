# Perception for a Robotic Mealtime Assistant

This project builds a camera based perception module for a future mealtime assistant robot.  
The module predicts when a person is **ready for a bite** versus **not ready**, based on upper body pose, head orientation, and hand motion.

It uses

- MediaPipe Holistic to extract pose, face, and hand keypoints
- Engineered window level features computed from those keypoints
- Subject level train / val / test splits
- A Random Forest baseline and neural network baselines
- Temporal smoothing over window predictions

The code and data structure follow a phased plan used in AER1515.

## Folder layout

- `data/`
  - `raw/` – input MP4 clips named by `clip_id`
  - `dataset_index.csv` – list of clips that belong to the dataset, with basic metadata
  - `splits.csv` – subject level split assignment for each clip  
- `results/`
  - `holistic/` – per clip MediaPipe Holistic CSV
  - `features/` – per clip engineered feature CSV and `all_windows.csv`
  - `models/` – trained models, for example `baseline_rf.joblib`
  - `metrics/` – JSON metrics for RF and NN baselines
  - `predictions/` – CSVs with per window predictions
- `scripts/`
  - Small scripts that implement each step of the pipeline
- `configs/`
  - Constant lists such as `WINDOW_FEATURE_COLUMNS`
- `src/`
  - Helper functions shared by multiple scripts

## Pipeline overview

Conceptually the data flow is:

1. Raw video  
2. Segment labeling (ready vs not_ready time ranges)  
3. Holistic keypoints  
4. Engineered features  
5. Sliding windows with labels  
6. `all_windows.csv`  
7. Models and evaluation

In code the training pipeline only uses the stages from raw video onward, with segment labels already baked into the window labels.

### 1. Run Holistic on each clip

Input:

- MP4 files in `data/raw/{clip_id}.mp4`

Output:

- CSV files in `results/holistic/{clip_id}_holistic.csv`

Example command:

```bash
python scripts/run_holistic_on_folder.py data/raw results/holistic
