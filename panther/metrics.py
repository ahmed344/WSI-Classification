"""Classification metrics shared by PANTHER training and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute fixed-class aggregate metrics, report, and confusion matrix."""
    labels_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predictions = probability_array.argmax(axis=1)
    fixed = list(range(len(class_names)))
    try:
        if len(set(labels_array.tolist())) != len(class_names):
            auc = None
        elif len(class_names) == 2:
            auc = float(roc_auc_score(labels_array, probability_array[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    labels_array,
                    probability_array,
                    labels=fixed,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_array, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels_array,
                predictions,
                labels=fixed,
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_roc_auc": auc,
        "classification_report": classification_report(
            labels_array,
            predictions,
            labels=fixed,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels_array, predictions, labels=fixed
        ).tolist(),
        "predictions": predictions.tolist(),
    }

