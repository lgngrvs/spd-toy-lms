from pathlib import Path
from typing import Any, override

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.experiments.ih.model import (
    PositionalEncoding,
    TransformerBlock,
)
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class TrigramModelPaths(BaseModel):
    """Paths to output files from a TrigramM training run."""

    trigram_train_config: Path
    checkpoint: Path


class TrigramModelConfig(BaseModel):
    vocab_size: PositiveInt
    seq_len: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    ff_fanout: PositiveInt
    use_ff: bool
    use_pos_encoding: bool
    # No layer_norm in a 1-layer TrigramTransformer.
    device: str = "cpu"


class TrigramTransformer(nn.Module):
    def __init__(self, cfg: TrigramModelConfig):
        super().__init__()
        self.config = cfg

        adjusted_vocab_size = cfg.vocab_size + 1  # +1 for BOS token
        self.token_embed = nn.Embedding(adjusted_vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.seq_len)

        self.block = TransformerBlock(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff_fanout=cfg.ff_fanout,
            use_ff=cfg.use_ff,
            use_pos_encoding=cfg.use_pos_encoding,
            use_layer_norm=self.config.use_ff,
            max_len=cfg.seq_len,
        )

        self.unembed = nn.Linear(cfg.d_model, adjusted_vocab_size, bias=False)

    @override
    def forward(self, tokens: Float[Tensor, "B S"], **_):
        x = self.token_embed(tokens)
        x = self.block(x)
        logits = self.unembed(x)
        return logits

    def get_attention_weights(self, tokens: Float[Tensor, "B S"]):
        x = self.token_embed(tokens)
        attn_weights = self.block.attn.get_attention_weights(x)
        return attn_weights

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TrigramModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        trigram_model_config_path = download_wandb_file(run, run_dir, "trigram_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TrigramModelPaths(
            trigram_train_config=trigram_model_config_path, checkpoint=checkpoint_path
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TrigramTransformer", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                paths = TrigramModelPaths(
                    trigram_train_config=run_dir / "trigram_train_config.yaml",
                    checkpoint=run_dir / "trigram.pth",
                )
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = TrigramModelPaths(
                trigram_train_config=Path(path).parent / "trigram_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.trigram_train_config) as f:
            trigram_train_config_dict = yaml.safe_load(f)

        trigram_config = TrigramModelConfig(**trigram_train_config_dict["trigram_model_config"])
        trigram_model = cls(cfg=trigram_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        trigram_model.load_state_dict(params)

        return trigram_model, trigram_train_config_dict
