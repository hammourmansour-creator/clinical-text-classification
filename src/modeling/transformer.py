# src/modeling/transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class TransformerBundle:
    tokenizer: any
    model: any


def build_distilbert(num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]) -> TransformerBundle:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return TransformerBundle(tokenizer=tokenizer, model=model)


def compute_class_weights(label_ids: List[int], num_labels: int) -> torch.Tensor:
    """
    Balanced weights: total/(num_labels * count(label))
    Helps minority / harder classes like OBJECTIVE.
    """
    counts = np.bincount(np.array(label_ids), minlength=num_labels).astype(np.float32)
    total = counts.sum()
    weights = total / (num_labels * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float)
