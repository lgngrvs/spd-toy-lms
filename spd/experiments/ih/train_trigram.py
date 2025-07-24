from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from matplotlib import pyplot as plt

from spd.log import logger
from spd.utils.data_utils import DatasetGeneratedDataLoader, TrigramDataset
from spd.utils.general_utils import set_seed

from .train_ih import TrigramTrainConfig, plot_loss_curve, train
from .trigram_model import TrigramModelConfig, TrigramTransformer

# Config Class

"""
Moved to train_ih to prevent circular import
"""


# Utility functions


def get_run_name(
    config: TrigramTrainConfig,
) -> str:
    """Generate a run name based on the config."""
    run_name = ""
    run_name += f"induction_heads_v{config.trigram_model_config.vocab_size}_seq{config.trigram_model_config.seq_len}"
    run_name += f"_heads{config.trigram_model_config.n_heads}_layers1"
    run_name += f"_steps{config.steps}_batch{config.batch_size}_lr{config.lr}"
    if config.trigram_model_config.use_ff:
        run_name += f"_dmodel{config.trigram_model_config.d_model}"
        run_name += f"_ff_fanout{config.trigram_model_config.ff_fanout}"
    if config.lr_schedule:
        run_name += f"_lr_schedule_{config.lr_schedule}"
    run_name += f"use_ff_{config.trigram_model_config.use_ff}"
    run_name += f"use_pos_encoding_{config.trigram_model_config.use_pos_encoding}"
    return run_name


def get_model_and_dataloader(
    config: TrigramTrainConfig, device: str
) -> tuple[TrigramTransformer, DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Create the model and dataloader based on the config.
    """
    model = TrigramTransformer(config.trigram_model_config).to(device)

    # Create the dataset and dataloader
    dataset = TrigramDataset(
        vocab_size=config.trigram_model_config.vocab_size,
        seq_len=config.trigram_model_config.seq_len,
        device=device,
        n_trigrams=config.n_trigrams,
        min_skip_distance=config.min_skip_distance,
        max_skip_distance=config.max_skip_distance,
    )

    dataloader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    return model, dataloader


# Model and data setup


def run_train(config: TrigramTrainConfig, device: str) -> None:
    model, dataloader = get_model_and_dataloader(config, device)

    run_name = get_run_name(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.wandb_project:
        wandb.init(project=config.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "trigram_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    losses, loss_steps = train(
        model=model,
        dataloader=dataloader,
        log_wandb=False,
        steps=config.steps,
        print_freq=100,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_schedule=config.lr_schedule,
        lr_warmup=config.lr_warmup,
    )

    plot_loss_curve(
        losses=losses,
        steps=loss_steps,
        out_dir=out_dir,
    )

    plot_attention_maps_post_training(
        model=model,
        dataloader=dataloader,
        steps=config.attention_maps_n_steps,
        out_dir=out_dir,
    )

    model_path = out_dir / "trigram.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")


def plot_attention_maps_post_training(
    model: TrigramTransformer,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    out_dir: Path,
):
    model.eval()
    with torch.no_grad():
        eval_attention_weights = []
        for _ in range(0, steps):
            tokens, _ = next(iter(dataloader))
            attn_weights = model.get_attention_weights(tokens)
            eval_attention_weights.append(attn_weights)

        eval_attention_weights = torch.cat(eval_attention_weights, dim=0)

        # For each layer, for each head, plot the average and max attention weights
        avg_attn_weights = eval_attention_weights.mean(dim=0)
        max_attn_weights = eval_attention_weights.max(dim=0).values

        for head_index in range(model.config.n_heads):
            avg_attn = avg_attn_weights[0, head_index, :, :].cpu().numpy()
            max_attn = max_attn_weights[0, head_index, :, :].cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            assert isinstance(ax, np.ndarray), "Expected ax to be a numpy array of axes"
            ax[0].imshow(avg_attn, cmap="viridis", aspect="auto")
            ax[0].set_title(f"Layer {1}, Head {head_index + 1} - Avg Attention")
            ax[1].imshow(max_attn, cmap="viridis", aspect="auto")
            ax[1].set_title(f"Layer {1}, Head {head_index + 1} - Max Attention")
            plt.colorbar(ax[0].images[0], ax=ax[0])
            plt.colorbar(ax[1].images[0], ax=ax[1])
            plt.tight_layout()

            fig.savefig(out_dir / f"attention_layer{1}_head{head_index + 1}.png")
            plt.close(fig)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrigramTrainConfig(
        trigram_model_config=TrigramModelConfig(
            vocab_size=128,
            seq_len=64,
            d_model=64,
            n_heads=1,
            ff_fanout=4,
            use_ff=False,
            use_pos_encoding=True,
            device="cpu",
        ),
        wandb_project="trigrams",
        steps=50000,
        batch_size=1024,
        lr=1e-3,
        lr_schedule="constant",
        lr_warmup=1000,
        weight_decay=0.01,
        seed=42,
        attention_maps_n_steps=100,
        n_trigrams=5,
        min_skip_distance=0,
        max_skip_distance=15,
    )

    set_seed(config.seed)

    run_train(
        config=config,
        device=device,
    )
