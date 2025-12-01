# Baseline neural network on window features

This report summarizes the performance of the window based MLP baseline.

## Dataset summary

- Total windows: **1400**
- not_ready windows: **848**
- ready windows: **552**
- Features per window: **20**

## Model

- Architecture: MLP on window feature vector
- Output: logits for two classes [not_ready, ready]

## Validation performance

- Best epoch: **9**
- Best val loss: **0.7846**
- Best val accuracy: **0.4630**
- Best val F1 (ready as positive): **0.4082**

## Test performance

- Test loss: **0.8841**
- Test accuracy: **0.5196**
- Test F1 (ready as positive): **0.5586**

### Test confusion matrix

|              | predicted not_ready | predicted ready |
|--------------|---------------------|-----------------|
| actual not_ready | 22 | 5 |
| actual ready     | 44 | 31 |

_Rows are actual labels and columns are predicted labels._
