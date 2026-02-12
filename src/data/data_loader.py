from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class DatasetPaths:
    train: Path
    dev: Path
    test: Path


def project_root() -> Path:
    """
    Returns the project root assuming this file lives in: <root>/src/data_loader.py
    """
    return Path(__file__).resolve().parents[1]


def get_pubmed_paths(data_dir: str | Path | None = None) -> DatasetPaths:
    """
    Default dataset location:
      <project_root>/data/raw/pubmed_20k_rct/{train.txt, dev.txt, test.txt}
    """
    base = Path(data_dir) if data_dir is not None else (
        project_root() / "data" / "raw" / "pubmed_20k_rct"
    )
    return DatasetPaths(
        train=base / "train.txt",
        dev=base / "dev.txt",
        test=base / "test.txt",
    )


def parse_pubmed_rct_file(path: Path) -> pd.DataFrame:
    """
    PubMed RCT format:
      - Abstracts separated by blank lines
      - Lines look like: LABEL<TAB>sentence
      - Some lines may start with ### (abstract ID) -> skip
    Returns DataFrame with columns: label, text
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Tip: Ensure your dataset is at: {project_root() / 'data/raw/pubmed_20k_rct'}"
        )

    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                continue
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            records.append({"label": label, "text": text})

    return pd.DataFrame(records)


def load_pubmed_20k_rct(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    paths = get_pubmed_paths(data_dir)

    # extra safety: show clear error if folder is wrong
    missing = [p for p in [paths.train, paths.dev, paths.test] if not p.exists()]
    if missing:
        msg = "Missing dataset files:\n" + "\n".join(f"- {p}" for p in missing)
        msg += f"\n\nExpected folder:\n{paths.train.parent}"
        raise FileNotFoundError(msg)

    return {
        "train": parse_pubmed_rct_file(paths.train),
        "dev": parse_pubmed_rct_file(paths.dev),
        "test": parse_pubmed_rct_file(paths.test),
    }

def load_pubmed_rct(file_path: str | Path) -> pd.DataFrame:
    return parse_pubmed_rct_file(Path(file_path))

