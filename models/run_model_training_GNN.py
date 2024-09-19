from __future__ import annotations

import argparse
from fairchem.core.common.utils import (
    new_trainer_context,
)
from pathlib import Path
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
    setup_logging,
)


def evaluation_function(
    data_to_fit="S1_exc",
    config_path="",
    test=False,
):
    setup_logging()
    # checkpoint_path = "/media/mohammed/Work/FORMED_ML/GNN_models/checkpoints/2024-09-16-07-47-44/checkpoint.pt"
    parser: argparse.ArgumentParser = flags.get_parser()
    model_name = config_path.split("/")[-1].split(".")[0]
    result_dir = Path("/media/mohammed/Work/FORMED_ML/models/training/" + model_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args(
        [
            "--mode",
            "train",
            "--config-yml",
            config_path,
            "--run-dir",
            "/media/mohammed/Work/FORMED_ML/models/training/" + model_name,
            # "--checkpoint",
            # checkpoint_path,
            "--identifier",
            data_to_fit + "_" + config_path.split("/")[-1].split(".")[0],
        ]
    )
    config = build_config(args, {})
    config["trainer"] = "energy"
    for dataset_name in ["train", "val", "test"]:
        config["dataset"][dataset_name]["key_mapping"] = {data_to_fit: "energy"}
        config["dataset"][dataset_name]["a2g_args"]["r_data_keys"] = [data_to_fit]
    config["optim"]["lr_initial"] = 0.0001
    if test:
        config["optim"]["max_epochs"] = 1
    with new_trainer_context(config=config) as ctx:
        config = ctx.config
        task = ctx.task
        trainer = ctx.trainer
    task.setup(trainer)
    task.trainer.scheduler.optimizer.param_groups[0]["lr"] = config["optim"][
        "lr_initial"
    ]  # update learning rate
    task.run()
    return task, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN models")
    parser.add_argument(
        "--data_to_fit",
        type=str,
        default="S1_exc",
        help="Target property to fit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to run a test",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/media/mohammed/Work/FORMED_ML/config_files/schnet/schnet.yml",
        help="Path to config file",
    )
    evaluation_function(**vars(parser.parse_args()))
