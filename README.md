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
