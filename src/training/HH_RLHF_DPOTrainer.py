from trl import DPOTrainer
from typing import Optional, Union, Dict
from transformers import PreTrainedModel
import torch.nn as nn


class HH_RLHF_DPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, logs, *args, **kwargs):
        super().log(logs)

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        return feature