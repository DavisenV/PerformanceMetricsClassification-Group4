# PerformanceMetricsClassification-Group4

Group lab for *Foundation of Machine Learning*: evaluating binary classifiers with performance metrics beyond accuracy.

## Contents

| File | Description |
| --- | --- |
| `PerformanceMetricsClassification.ipynb` | Main lab notebook. Trains an `SGDClassifier` to detect the digit 5 in MNIST, then compares it to a `DummyClassifier` and a `RandomForestClassifier` using cross-validation, confusion matrices, precision, recall, F1, precision-recall curves and ROC/AUC. |
| `Tasks.ipynb` | The group's answers to the "To the student" questions. |
| `pyproject.toml` / `uv.lock` | Project dependencies, managed with [uv](https://docs.astral.sh/uv/). |
| `requirements.txt` | The same dependencies exported for `pip` users. |

## Topics covered

- Binary (one-vs-all) classification on MNIST and Fashion-MNIST
- Cross-validation with `cross_validate` and `cross_val_predict`
- Why accuracy is misleading on imbalanced data (the `DummyClassifier` baseline)
- Confusion matrix: TN, FP, FN, TP
- Precision, recall and F1 score
- The precision/recall trade-off and choosing a decision threshold
- ROC curve and ROC AUC

## Summary of insights

This project demonstrates that evaluating a classifier requires more than reporting
accuracy. A classifier assigns observations to predefined categories based on
patterns in their input features. For the MNIST task, the model identified whether
an image was a `5` or not. Three-fold cross-validation trained the model once per
fold and provided separate fit times, score times, and validation scores.

The `SGDClassifier` achieved approximately **95.7% mean accuracy**, compared with
about **90.97%** for a `DummyClassifier` that always predicted “not 5.” This
baseline shows why accuracy can be misleading for imbalanced binary
classification: a model can appear effective while completely missing the
minority class. The confusion matrix makes these errors explicit through true
negatives, false positives, false negatives, and true positives. For the
`SGDClassifier`, precision was approximately **0.837**, recall was **0.651**, and
the F1 score was **0.733**.

Precision measures how many predicted positives are correct, while recall measures
how many actual positives are detected. Increasing the decision threshold generally
raises precision but lowers recall; decreasing it generally raises recall but
lowers precision. The appropriate balance depends on the cost of each error. For
example, medical screening, fraud detection, and airport security generally favor
high recall to avoid missing dangerous cases, while investment recommendations,
spam filtering, and child-safe content recommendations generally favor high
precision to reduce costly or harmful false alarms.

The same evaluation process was applied to Fashion-MNIST by classifying sandals
versus non-sandals. That model achieved approximately **97.94% test accuracy**,
with **89.70% precision**, **89.70% recall**, and an **F1 score of 0.897**,
substantially outperforming the **90.00%** dummy baseline. A security-drone
example further illustrates the metrics: detecting 8 of 10 people while raising
4 false alarms gives a precision of **8/12 = 0.667** and a recall of
**8/10 = 0.80**. In that setting, recall is especially important because missed
intruders are more serious than false alarms.

## Setup

Requires Python 3.12 or newer.

### With uv (recommended)

```bash
uv sync
uv run jupyter lab   # or open the notebooks in VS Code and select the .venv kernel
```

`jupyter` is not a project dependency. If you want to run Jupyter Lab, use `uv run --with jupyterlab jupyter lab`.

### With pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To add a dependency with uv, run `uv add <package>`. Then refresh the pip file with:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

## Notes

- The MNIST and Fashion-MNIST datasets are downloaded from OpenML with `fetch_openml` the first time the notebooks run, so the first run needs an internet connection.
- Training and cross-validation on 60,000 images take a few minutes on a laptop.
