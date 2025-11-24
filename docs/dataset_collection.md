# Dataset collection notes

Each clip uses a numeric clip_id.

Folder layout:
- data/raw/{clip_id}.mp4
- data/segments/{clip_id}_segments.csv
- results/holistic/{clip_id}_holistic.csv
- results/features/{clip_id}_features.csv
- results/features/{clip_id}_windows.csv

We track metadata in data/dataset_index.csv with columns:
- clip_id
- participant_id
- raw_path
- notes
- has_labels (yes or no)

Id ranges:
- Participant 1: clips 4 to 9
- Participant 2: clips 10 to 17
- Future participants: start at 20, 30, 40

Update due to time constraint:
- Compressed dataset target for Nov deadline
- 4 to 5 visual participants, 3 to 4 clips each, total 12 to 20 clips.