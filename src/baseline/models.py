"""
Created on 2022/09/11
@author Sangwoo Han
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..modules import MLPLayer


class BaselineModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        linear_size: List[int] = [256],
        linear_dropout: float = 0.2,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.clf = MLPLayer(
            config.hidden_size, num_labels, linear_dropout, linear_size, use_layernorm
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        return self.clf(outputs[0][:, 0])
