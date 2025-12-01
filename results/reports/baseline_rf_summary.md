Validation (windows, RF baseline)

- Total windows 108
- Accuracy 0.37

Per class
- not_ready
  - Precision 0.45
  - Recall 0.60
  - F1 0.51
  - Support 60 windows
- ready
  - Precision 0.14
  - Recall 0.08
  - F1 0.11
  - Support 48 windows

Interpretation

On the validation split the RF baseline is biased toward predicting not_ready. It correctly identifies most not_ready windows but rarely predicts ready and misses most true ready windows. This makes val a harsh split and shows that the model is not well calibrated for ready on this subset, which motivates trying a neural baseline and better temporal smoothing.

Test (windows, RF baseline)

- Total windows 102
- Accuracy 0.64

Confusion matrix (labels [not_ready, ready])

|                 | Pred not_ready | Pred ready |
|-----------------|----------------|-----------|
| True not_ready  | 17             | 10        |
| True ready      | 27             | 48        |

Per class

- not_ready
  - Precision 0.39
  - Recall 0.63
  - F1 0.48
  - Support 27 windows
- ready
  - Precision 0.83
  - Recall 0.64
  - F1 0.72
  - Support 75 windows

Interpretation

On the test split the RF baseline performs much better than on validation. It achieves about 0.64 accuracy and a strong F1 of about 0.72 for ready. It still struggles somewhat with not_ready, with lower precision and F1. The model is effectively biased toward ready on this split, which matches the ready heavy class balance in the test windows. This reinforces that the RF is more reliable at detecting ready on test than on val and that dataset differences between val and test play a big role.
