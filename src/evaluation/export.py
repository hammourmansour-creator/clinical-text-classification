from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.io import ensure_dir, save_df_csv, save_json


def export_baseline_outputs(
    out_dir: str | Path,
    model_name: str,
    split_name: str,
    outputs: Any,  # your EvalOutputs dataclass instance
) -> dict:
    """
    Saves:
      - metrics.json (accuracy + classification report)
      - confusion_matrix.csv
    Returns the JSON dict that was saved.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    metrics_payload: Dict[str, Any] = {
        "model": model_name,
        "split": split_name,
        "accuracy": float(outputs.accuracy),
        "report": outputs.report,  # already a dict
    }

    save_json(out_dir / "metrics.json", metrics_payload)

    # confusion_df is assumed to be a DataFrame
    if hasattr(outputs, "confusion_df") and isinstance(outputs.confusion_df, pd.DataFrame):
        save_df_csv(out_dir / "confusion_matrix.csv", outputs.confusion_df)

    return metrics_payload
