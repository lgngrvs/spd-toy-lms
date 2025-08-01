import math
from pathlib import Path
from typing import Any, override

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class InductionModelPaths(BaseModel):
    """Paths to output files from an InductionModel training run."""

    induction_train_config: Path
    checkpoint: Path


class InductionModelConfig(BaseModel):
    vocab_size: PositiveInt
    seq_len: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    ff_fanout: PositiveInt
    use_ff: bool
    use_pos_encoding: bool
    use_layer_norm: bool
    device: str = "cpu"


class PositionalEncoding(nn.Module):
    """
    Regular positional encoding as described in the original Transformer paper.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)
        assert d_model % 2 == 0, "This positional encoding implementation breaks if d_model is odd"
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    @override
    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:
        return x + self.pe[:, : x.size(1)]

    def get_pe(self, seq_len: int) -> Float[Tensor, "1 S D"]:
        return self.pe[:, :seq_len]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer with optional positional encoding.
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int, use_pos_encoding: bool):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len
        self.use_pos_encoding = use_pos_encoding

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if self.use_pos_encoding:
            self.pos_enc = PositionalEncoding(d_model, max_len)

    @override
    def forward(self, x: Tensor):
        b, s, d_model = x.shape
        n_heads, dim_head = self.n_heads, self.d_head

        Q_lin = self.q_proj(x)
        K_lin = self.k_proj(x)
        V_lin = self.v_proj(x)

        if self.use_pos_encoding:
            # Shortformer-style positional according
            # According to https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp#introducing-our-toy-attention-only-model
            # this makes induction-heads form MUCH faster.
            pe = self.pos_enc.get_pe(s)
            Q_lin = Q_lin + pe
            K_lin = K_lin + pe

        # (B, S, D) -> (B, S, H, Dh) -> (B, H, S, Dh)
        def split_heads(t: Float[Tensor, "B S D"]) -> Float[Tensor, "B H S Dh"]:
            return t.view(b, s, n_heads, dim_head).transpose(1, 2)

        Q = split_heads(Q_lin)
        K = split_heads(K_lin)
        V = split_heads(V_lin)

        context = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, is_causal=True
        )

        context = context.transpose(1, 2).contiguous().view(b, s, d_model)
        out = self.out_proj(context)
        return out

    def get_attention_weights(self, x: Tensor):
        b, s, _ = x.shape
        Q_lin = self.q_proj(x)
        K_lin = self.k_proj(x)

        if self.use_pos_encoding:
            pe = self.pos_enc.get_pe(s)
            Q_lin = Q_lin + pe
            K_lin = K_lin + pe

        def split_heads(t: Tensor) -> Tensor:
            return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(Q_lin)
        K = split_heads(K_lin)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.d_head))

        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn


class MLPBlock(nn.Module):
    """A simple MLP block with GELU"""

    def __init__(self, d_model: int, ff_fanout: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * ff_fanout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model * ff_fanout, d_model)

    @override
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Stacks MultiHeadSelfAttention layers optionally with MLP blocks, positional encoding, and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_fanout: int,
        use_ff: bool,
        use_pos_encoding: bool,
        use_layer_norm: bool,
        max_len: int,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_len, use_pos_encoding)
        self.use_ff = use_ff
        self.use_pos_encoding = use_pos_encoding
        self.use_layer_norm = use_layer_norm
        self.max_len = max_len
        if use_ff:
            self.ff = MLPBlock(d_model, ff_fanout)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

    @override
    def forward(self, x: Float[Tensor, "B S D"]):
        attn_out = self.attn(x)
        if self.use_layer_norm:
            x = self.ln1(x + attn_out)
        else:
            x = x + attn_out
        if self.use_ff:
            ff_out = self.ff(x)
            if self.use_layer_norm:
                x = self.ln2(x + ff_out)
            else:
                x = x + ff_out
        return x


class InductionTransformer(nn.Module):
    def __init__(self, cfg: InductionModelConfig):
        super().__init__()
        self.config = cfg

        adjusted_vocab_size = cfg.vocab_size + 2  # +2 for BOS and special induction token
        self.token_embed = nn.Embedding(adjusted_vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.seq_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_fanout=cfg.ff_fanout,
                    use_ff=cfg.use_ff,
                    use_pos_encoding=cfg.use_pos_encoding,
                    use_layer_norm=cfg.use_layer_norm,
                    max_len=cfg.seq_len,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        if self.config.use_layer_norm:
            self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, adjusted_vocab_size, bias=False)

    @override
    def forward(self, tokens: Float[Tensor, "B S"], **_):
        x = self.token_embed(tokens)

        for block in self.blocks:
            x = block(x)

        if self.config.use_layer_norm:
            x = self.ln_f(x)
        logits = self.unembed(x)
        return logits

    def get_attention_weights(self, tokens: Float[Tensor, "B S"]):
        x = self.token_embed(tokens)

        attn_weights = []
        for block in self.blocks:
            weights = block.attn.get_attention_weights(x)
            attn_weights.append(weights)
            x = block(x)

        return torch.stack(attn_weights, dim=1)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> InductionModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        induction_model_config_path = download_wandb_file(run, run_dir, "ih_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return InductionModelPaths(
            induction_train_config=induction_model_config_path, checkpoint=checkpoint_path
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["InductionTransformer", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                paths = InductionModelPaths(
                    induction_train_config=run_dir / "ih_train_config.yaml",
                    checkpoint=run_dir / "ih.pth",
                )
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = InductionModelPaths(
                induction_train_config=Path(path).parent / "ih_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.induction_train_config) as f:
            induction_train_config_dict = yaml.safe_load(f)

        induction_config = InductionModelConfig(**induction_train_config_dict["ih_model_config"])
        induction_model = cls(cfg=induction_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        induction_model.load_state_dict(params)

        return induction_model, induction_train_config_dict
