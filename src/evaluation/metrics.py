from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@dataclass
class EvalOutputs:
    accuracy: float
    report: Dict[str, Any]
    confusion_df: pd.DataFrame

def evaluate_predictions(y_true, y_pred, labels: List[str]) -> EvalOutputs:
    """
    Computes:
    - accuracy
    - classification report (precision/recall/F1 per class + macro avg)
    - confusion matrix (as a labeled DataFrame)
    """
    acc = float(accuracy_score(y_true, y_pred))

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return EvalOutputs(accuracy=acc, report=report, confusion_df=cm_df)
