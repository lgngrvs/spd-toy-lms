from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import wandb

from spd.configs import Config, TrigramTaskConfig
from spd.experiments.ih.trigram_model import TrigramModelConfig, TrigramTransformer
from spd.log import logger
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, TrigramDataset
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, trigram_model_cfg: TrigramModelConfig) -> str:
    suffix = get_common_run_name_suffix(config)
    suffix += f"_vocab{trigram_model_cfg.vocab_size}"
    suffix += f"_seq{trigram_model_cfg.seq_len}"
    suffix += f"_hid{trigram_model_cfg.d_model}"
    suffix += f"_heads{trigram_model_cfg.n_heads}"
    suffix += f"use_ff{trigram_model_cfg.use_ff}"
    suffix += f"_use_pos{trigram_model_cfg.use_pos_encoding}"
    if trigram_model_cfg.use_ff:
        suffix += f"_ff{trigram_model_cfg.ff_fanout}"

    exp_name = f"trigram{trigram_model_cfg.vocab_size}-{trigram_model_cfg.d_model}_"

    return config.wandb_run_name_prefix + exp_name + suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    trigram_model: TrigramTransformer,
    trigram_model_train_config_dict: dict[str, Any],
) -> None:
    save_file(trigram_model.state_dict(), out_dir / "trigram.pth")
    save_file(trigram_model_train_config_dict, out_dir / "trigram_train_config.yaml")

    if save_to_wandb:
        wandb.save(str(out_dir / "trigram.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "trigram_train_config.yaml"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["trigram"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    # Get output directory (automatically uses wandb run ID if available)
    out_dir = get_output_dir()

    task_config = config.task_config
    assert isinstance(task_config, TrigramTaskConfig)  # check on this

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict = TrigramTransformer.from_pretrained(
        config.pretrained_model_path,
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, trigram_model_cfg=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    save_file(config.model_dump(mode="json"), out_dir / "final_config.yaml")
    if sweep_params:
        save_file(sweep_params, out_dir / "sweep_params.yaml")
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")
        if sweep_params:
            wandb.save(str(out_dir / "sweep_params.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        trigram_model=target_model,
        trigram_model_train_config_dict=target_model_train_config_dict,
    )

    dataset = TrigramDataset(
        # Check that these are correct params
        vocab_size=target_model.config.vocab_size,
        seq_len=target_model.config.seq_len,
        n_trigrams=task_config.n_trigrams,
        min_skip_distance=task_config.min_skip_distance,
        max_skip_distance=task_config.max_skip_distance,
        device=device,
    )

    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
