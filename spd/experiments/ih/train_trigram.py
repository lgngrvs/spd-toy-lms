from spd.utils.data_utils import TrigramDataset

# from spd.experiments.ih.train_ih import InductionHeadsTrainConfig, get_run_name, linear_lr, constant_lr, warmup_lr

# import torch
# from jaxtyping import Float
# from torch import Tensor
# from torch.utils.data import DataLoader, Dataset
# from spd.experiments.ih.model import InductionModelConfig, InductionTransformer
# from spd.log import logger
# from spd.utils.general_utils import set_seed

vocab_size = 20
seq_len = 15

dataset = TrigramDataset(
    vocab_size, seq_len, "cpu", n_trigrams=3, min_skip_distance=1, max_skip_distance=8
)
batch = dataset.generate_batch(10)
print(batch[0], "\n", batch[1])
print(dataset.trigram_firsts, dataset.trigram_seconds, dataset.trigram_thirds)

"""
# Config Class

# Modified from train_ih.py
class TrigramTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True) # what does this do?
    wandb_project: str | None = None
    ih_model_config: InductionModelConfig # edit this
    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    lr_warmup: int | float
    weight_decay: float
    lr_schedule: Literal["cosine", "constant", "linear"] = "linear"
    seed: int = 0
    attention_maps_n_steps: PositiveInt
    prefix_window: PositiveInt



# Utility functions

import linear_lr, constant_lr, cosine_decay_lr, warmup_lr

# Core training function
import train()? 

# Model and data setup

# Main runner

# Vis functions
import plot_loss_curve()
"""
