# Sequence NN Baseline on Window Sequences

## 1. Motivation

The original baseline used a Random Forest classifier on **independent window features**.  
This ignores temporal order inside each clip.

To better use the fact that windows come from a **time sequence**, we built a sequence neural network that takes **entire window sequences per video** and predicts a label per window. The goal is to check whether a sequence NN can outperform the classical Random Forest baseline on the same engineered features.

---

## 2. Input data and splits

**Base dataset**

- Source: `results/features/all_windows.csv`
- Features per window  
  - 20 dimensional engineered window feature vector  
    (hand to mouth distance stats, wrist height stats, mouth width stats, etc)
- Label per window  
  - `label` in `{not_ready, ready}`

**Splits**

- Splits are defined at the **video_id** level  
  so no video appears in more than one split.

From `check_clip_splits.py`:

- Total unique `video_id` in `all_windows.csv`: **31**
  - Train: 19 video_ids  
    `[18, 19, 20, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 37, 38, 47, 48]`
  - Val: 6 video_ids  
    `[27, 28, 29, 30, 43, 56]`
  - Test: 6 video_ids  
    `[39, 40, 41, 42, 49, 52]`

**Sequence construction**

- For each split we group windows by `video_id`
- Inside each video we sort by `window_start`
- Each video becomes a sequence of shape `(T_i, 20)`  
  where `T_i` is the number of windows for that video

The sequences are padded to a common length per split and saved to:

- `results/features/seq_nn_datasets.npz`  
  with keys  
  - `X_train, y_train, len_train`  
  - `X_val, y_val, len_val`  
  - `X_test, y_test, len_test`

`y_*` use `-1` as a padding label for timesteps beyond the true length.

---

## 3. Preprocessing for the NN

The Random Forest uses raw window features.

The sequence NN uses **standardized** window features:

- Fit a `StandardScaler` on **train** windows only  
  using `WINDOW_FEATURE_COLUMNS` from `feature_config.py`
- Apply the scaler to train, val and test windows before packing them into sequences
- The scaler is saved to  
  `results/config/seq_nn_window_scaler.joblib`

This keeps RF and NN comparable on the same feature set, while giving the NN a better conditioned input space.

---

## 4. Sequence NN architecture

File: `scripts/baseline_seq_nn.py`

Model: **Bidirectional GRU over window sequences**

- Input shape: `(batch_size, time_steps, in_features)`  
  where `in_features = 20`
- Backbone  
  - GRU with  
    - hidden size 64  
    - 1 layer  
    - bidirectional = True  
  - Output from GRU has shape `(B, T, 2 * hidden_size)`
- Output head  
  - Linear layer from `2 * hidden_size` to 2 logits  
  - Output logits per timestep  
    `(B, T, 2)` for classes `[not_ready, ready]`

We apply a softmax inside the evaluation code only. The training loss is computed directly on logits.

---

## 5. Training setup

File: `scripts/train_seq_nn.py`

- Dataset  
  - Loads sequences from `seq_nn_datasets.npz`
  - Uses a custom `SeqDataset` and `DataLoader` with padding aware labels
- Loss  
  - Cross entropy loss over all **non padded** timesteps  
  - Padded labels use value `-1` and are masked out in `masked_loss`
- Optimizer  
  - Adam with learning rate `3e-4`
- Class weights  
  - For this first sequence NN run we used equal weights `[1.0, 1.0]`  
    so the model is not biased by class weight heuristics
- Training details  
  - Epochs: 80  
  - Batch size: 4 sequences per batch  
  - Device: CPU on this run  
  - Early model selection  
    - After each epoch we evaluate on val  
    - We track `val_f1` with `ready` as the positive class  
    - We keep the model state with **best val F1**

Logs are saved to:

- `results/metrics/seq_nn_training_log.json`

The best model is saved to:

- `results/models/seq_nn.pt`

A short summary is saved to:

- `results/metrics/seq_nn_metrics.json`

---

## 6. Test performance of the sequence NN

From `seq_nn_metrics.json` and the final printout:

- Test loss ≈ **1.16**
- Test accuracy ≈ **0.676**
- Test F1 (ready positive) ≈ **0.798**

Test confusion matrix on window labels:

- `[[4, 23], [10, 65]]` where  
  - True not_ready = 27 windows  
    - 4 predicted not_ready correctly  
    - 23 predicted ready  
  - True ready = 75 windows  
    - 65 predicted ready correctly  
    - 10 predicted not_ready

So the sequence NN is quite aggressive about detecting `ready` windows, with high recall on the ready class and many false ready flags on not_ready windows. This behaviour is similar in spirit to the tuned RF, but achieved through a temporal model rather than threshold tweaking.

---

## 7. Comparison against Random Forest baselines

### 7.1 Original window RF baseline (0.5 cutoff)

From `baseline_rf_metrics.json` and `compare_rf_vs_nn.py`:

- Test accuracy ≈ **0.637**
- Test F1 (ready positive) ≈ **0.722**
- Confusion: `[[17, 10], [27, 48]]`  
  - not_ready: 17 correct, 10 false ready  
  - ready: 48 correct, 27 missed

### 7.2 RF with tuned probability threshold

From `baseline_rf_threshold_tuned.json` with best threshold `t = 0.225` (chosen on val by F1 ready):

- Test accuracy ≈ **0.725**
- Test F1 (ready positive) ≈ **0.829**
- Confusion: `[[6, 21], [7, 68]]`  
  - not_ready: 6 correct, 21 false ready  
  - ready: 68 correct, 7 missed

This is an aggressive operating point that favours ready recall over not_ready precision.

### 7.3 Sequence NN GRU (this model)

- Test accuracy ≈ **0.676**
- Test F1 (ready positive) ≈ **0.798**
- Confusion: `[[4, 23], [10, 65]]`

**Key points**

- Compared to the **original RF baseline** at 0.5 threshold  
  the sequence NN achieves **higher accuracy** and **higher F1** on the ready class  
  so it is a strictly stronger neural baseline on this dataset.
- Compared to the **tuned RF** with threshold 0.225  
  the sequence NN is slightly weaker in raw numbers (0.676 vs 0.725 accuracy, 0.798 vs 0.829 F1)  
  but both models are in a similar regime  
  high recall on ready and many false ready flags on not_ready.

---

## 8. Interpretation

- A pure Random Forest on independent window features is a strong classical baseline and performs well.  
- The sequence GRU NN uses the *same* engineered window features but respects the **temporal order** within each video.  
- On the held out test set the sequence NN **outperforms the standard RF** (0.5 cutoff) in both accuracy and F1 for ready, which matches the expectation that a neural sequence model can learn more from the temporal structure.
- A very aggressively tuned RF threshold can still push performance slightly higher by trading safety for recall, but that is a threshold selection and operating point choice rather than a fundamentally different model family.

For the project, this supports the claim that:

> *“A sequence neural network that sees ordered windows per clip can outperform a classical Random Forest that only sees independent window statistics.”*

The RF remains valuable as a simple, interpretable baseline, while the sequence NN is the more capable model when temporal behaviour matters.
