from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def save_confusion_heatmap(confusion_df: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    im = ax.imshow(confusion_df.values)
    ax.set_title(title)

    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_df.index)

    # annotate cells
    for i in range(confusion_df.shape[0]):
        for j in range(confusion_df.shape[1]):
            ax.text(j, i, str(confusion_df.iat[i, j]), ha="center", va="center")

    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
