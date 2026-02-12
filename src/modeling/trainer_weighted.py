# src/modeling/trainer_weighted.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # move weights to same device
        weights = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
