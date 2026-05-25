import os
import hydra
import lightning as L
import numpy as np
import torch

from omegaconf import DictConfig, ListConfig
from omegaconf.base import Metadata, ContainerMetadata
from lightning.pytorch.loggers import TensorBoardLogger  # , CometLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

from mltau.tools.io import ParT_dataloader as dl
from mltau.models import MultiParTau_module, SingleParTau_module
from mltau.tools.evaluation import inference


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(cfg: DictConfig):
    torch.serialization.add_safe_globals([DictConfig, ListConfig, Metadata, ContainerMetadata, list, dict])
    torch.set_float32_matmul_precision("high")
    datamodule = dl.ParTDataModule(cfg=cfg, debug_run=cfg.training.debug_run)
    model_name = cfg.training.model.name
    if model_name == "MultiParTau":
        model = MultiParTau_module.ParTauModule(cfg=cfg, input_dim=17, num_dm_classes=6)
    elif model_name == "SingleParTau":
        model = SingleParTau_module.ParTauModule(
            cfg=cfg, input_dim=17, num_dm_classes=6, task=cfg.training.model.task
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose 'MultiParTau' or 'SingleParTau'."
        )
    models_dir = os.path.join(cfg.output_dir, "models")
    log_dir = os.path.join(cfg.output_dir, "logs")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Configure callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=100),  # Reduced refresh rate for CPU
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParT-model_best",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTau_experiment",
                log_graph=False,
                default_hp_metric=False,
            ),
        ],
        accelerator="auto",  # Automatically detect GPU/CPU
        precision="16-mixed",  # fp16 activations: halves GPU memory, ~30% faster
        num_sanity_val_steps=0,  # Skip sanity validation for faster startup
        enable_progress_bar=True,  # Keep enabled for monitoring
    )

    trainer.fit(model=model, datamodule=datamodule)
    # --- Inference on test set using best checkpoint ---
    best_ckpt_path = os.path.join(models_dir, "ParT-model_best.ckpt")
    if os.path.exists(best_ckpt_path):
        print(f"\n[INFO] Running inference on test set using {best_ckpt_path}")
        # Reload the best model
        if model_name == "MultiParTau":
            best_model = MultiParTau_module.ParTauModule.load_from_checkpoint(
                best_ckpt_path, cfg=cfg, input_dim=17, num_dm_classes=6, weights_only=False
            )
        elif model_name == "SingleParTau":
            best_model = SingleParTau_module.ParTauModule.load_from_checkpoint(
                best_ckpt_path,
                cfg=cfg,
                input_dim=17,
                num_dm_classes=6,
                task=cfg.training.model.task,
                weights_only=False,
            )
        else:
            raise ValueError(f"Unknown model '{model_name}' for prediction.")

        inference.create_predictions_files(
            best_model=best_model, model_name=model_name, cfg=cfg
        )

    else:
        print(
            f"[WARNING] Best checkpoint not found at {best_ckpt_path}. Skipping inference."
        )


if __name__ == "__main__":
    train()
