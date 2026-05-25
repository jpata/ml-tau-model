import torch
import awkward as ak
import numpy as np
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools import general as g
from mltau.tools.losses import FocalLoss, SigmoidFocalLoss
from mltau.tools.logging import tagging, kinematics, decay_mode, charge_id
from mltau.models.SingleParTau import ParTau

VALID_TASKS = {"is_tau", "charge", "decay_mode", "kinematics"}


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int, task: str):
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        self.cfg = cfg
        self.task = task
        self.ParTau = ParTau(
            input_dim=input_dim,
            task=task,
            num_dm_classes=num_dm_classes,
            num_layers=2,
            embed_dims=[256, 512, 256],
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="eta-phi",
        )
        if task == "is_tau":
            self.loss_fn = SigmoidFocalLoss(alpha=0.75, gamma=2.0, reduction="none")
        elif task == "charge":
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        elif task == "decay_mode":
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        elif task == "kinematics":
            self.loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    def _loss_key(self):
        return f"{self.task}_loss"

    def _make_accumulator(self):
        # Original aggregate-only accumulator kept for reference.
        # return {key: [] for key in ["loss", self._loss_key()]}
        keys = ["loss", self._loss_key()]
        if self.task == "kinematics":
            keys.extend(
                [
                    "kinematics_log_pt_loss",
                    "kinematics_delta_eta_loss",
                    "kinematics_sin_delta_phi_loss",
                    "kinematics_cos_delta_phi_loss",
                    "kinematics_log_mass_loss",
                ]
            )
        return {key: [] for key in keys}

    def training_step(self, batch, batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        for key, value in metrics.items():
            self.training_loss_accumulator[key].append(value.detach())
        self.log(
            "LR",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return metrics["loss"]

    def predict_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.ParTau.parameters(),
            lr=self.cfg.training.lr,
        )

        # Check if estimated_stepping_batches is available and valid
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if estimated_steps is None or estimated_steps <= 0:
            # Fallback: calculate based on config (will be approximate but functional)
            max_epochs = self.cfg.training.trainer.max_epochs
            # Use a conservative estimate of steps per epoch
            # This will be less precise but the scheduler will still work
            estimated_steps_per_epoch = 500  # Reasonable default for most datasets
            T_max = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated T_max={T_max} (estimated_stepping_batches not available)"
            )
        else:
            T_max = estimated_steps
            print(f"Using calculated T_max={T_max} from estimated_stepping_batches")

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=self.cfg.training.lr * 0.01,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _calculate_baseline_charges(self, inputs):
        """Calculate baseline jet charge using Q*kappa weighting."""
        cand_charges = inputs.cand_features[:, 7, :]
        cand_mask = inputs.cand_mask[:, 0, :]

        px = inputs.cand_kinematics_pxpypze[:, 0, :]
        py = inputs.cand_kinematics_pxpypze[:, 1, :]
        cand_pts = torch.sqrt(px**2 + py**2)

        try:
            reco_jet_p4s_ak = ak.Array(inputs.reco_jet_p4s)
            reco_jet_p4s = g.reinitialize_p4(reco_jet_p4s_ak)

            pt_values = reco_jet_p4s.pt
            if hasattr(pt_values, "to_numpy"):
                pt_numpy = pt_values.to_numpy()
            else:
                pt_numpy = ak.to_numpy(pt_values)

            if pt_numpy.ndim == 0:
                pt_numpy = np.array([pt_numpy])
            elif pt_numpy.ndim > 1:
                pt_numpy = pt_numpy.flatten()[: len(cand_charges)]

            jet_pts = torch.tensor(
                pt_numpy, dtype=torch.float32, device=cand_charges.device
            )

            if len(jet_pts) != len(cand_charges):
                if len(jet_pts) == 1:
                    jet_pts = jet_pts.repeat(len(cand_charges))
                else:
                    jet_pts = jet_pts[: len(cand_charges)]
        except Exception:
            jet_pts = torch.sum(cand_pts * cand_mask, dim=1)

        cand_charges_masked = cand_charges * cand_mask
        cand_pts_masked = cand_pts * cand_mask

        kappa = 0.2
        numer = torch.sum(cand_charges_masked * (cand_pts_masked**kappa), dim=1)
        denom = jet_pts**kappa
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        baseline_charges = numer / denom
        return baseline_charges.detach().cpu().numpy()

    # def configure_optimizers(self):
    #     optimizer = torch.optim.RAdam(
    #         params=self.ParTau.parameters(),
    #         lr=self.cfg.training.lr,
    #         betas=(0.95, 0.999),
    #         eps=1e-5,
    #     )
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=20000 * self.cfg.training.trainer.max_epochs,
    #         eta_min=self.cfg.training.lr * 0.01,
    #     )
    #     return [optimizer], [lr_scheduler]

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        model_output = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        # Keep logging inputs stable while using task-specific tensors for the loss.
        if self.task == "charge":
            charge_logits = model_output[0]
            predictions = {
                self.task: torch.softmax(charge_logits, dim=-1)[:, 1],
                "charge_logits": charge_logits,
            }
        elif self.task == "decay_mode":
            decay_mode_logits = model_output[0]
            predictions = {
                self.task: torch.softmax(decay_mode_logits, dim=-1),
                "decay_mode_logits": decay_mode_logits,
            }
        elif self.task == "is_tau":
            tau_logits = model_output[0]
            predictions = {
                self.task: torch.sigmoid(tau_logits),
                "is_tau_logits": tau_logits,
            }
        else:
            predictions = {self.task: model_output[0]}
        return predictions, inputs.target, inputs.weight

    def calculate_metrics(self, targets, predictions, weights):
        pred = predictions[self.task]
        target = targets[self.task]

        if self.task == "kinematics":
            component_raw_loss = self.loss_fn(pred, target)
            is_tau_mask = targets["is_tau"].bool()
            masked_component_loss = component_raw_loss[is_tau_mask]
            component_loss = masked_component_loss.mean(dim=0)
            l_m = 0.2
            loss = (
                component_loss[0]
                + component_loss[1]
                + component_loss[2]
                + component_loss[3]
                + l_m * component_loss[4]
            ) / (4.0 + l_m)
            metrics = {
                "loss": loss,
                self._loss_key(): loss,
                "kinematics_log_pt_loss": component_loss[0],
                "kinematics_delta_eta_loss": component_loss[1],
                "kinematics_sin_delta_phi_loss": component_loss[2],
                "kinematics_cos_delta_phi_loss": component_loss[3],
                "kinematics_log_mass_loss": component_loss[4],
            }
            return metrics
        elif self.task == "is_tau":
            loss = (self.loss_fn(predictions["is_tau_logits"], target) * weights).mean()
        elif self.task == "charge":
            loss = self.loss_fn(
                predictions["charge_logits"], targets["charge"].long()
            ).mean()
        else:  # "decay_mode" — only meaningful for signal taus
            raw_loss = self.loss_fn(predictions["decay_mode_logits"], target)
            is_tau_mask = targets["is_tau"].bool()
            loss = raw_loss[is_tau_mask].mean()

        return {"loss": loss, self._loss_key(): loss}

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        inputs = BatchInputs(*batch)
        self.validation_outputs.append(
            {
                "predictions": predictions,
                "targets": targets,
                "gen_jet_p4s": inputs.gen_jet_p4s,
                "reco_jet_p4s": inputs.reco_jet_p4s,
                "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
                "inputs": inputs if self.task == "charge" else None,
            }
        )
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())
        return metrics["loss"]

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_loss_accumulator = self._make_accumulator()

    def _log_task_metrics(
        self,
        targets,
        predictions,
        gen_jet_p4s,
        gen_jet_tau_p4s,
        reco_jet_p4s,
        tb_logger,
        current_epoch,
        dataset,
        baseline_charges=None,
    ):
        kwargs = dict(
            targets=targets,
            predictions=predictions,
            tb_logger=tb_logger,
            current_epoch=current_epoch,
        )
        if self.task == "is_tau":
            tagging.log_all_tagging_metrics(
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )
        elif self.task == "charge":
            charge_id.log_charge_id_performance(
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                baseline_charges=baseline_charges,
                **kwargs,
            )
        elif self.task == "decay_mode":
            decay_mode.log_all_decay_mode_metrics(**kwargs)
        elif self.task == "kinematics":
            kinematics.log_all_kinematics_metrics(
                reco_jet_p4s=reco_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = self.validation_outputs if dataset == "val" else []

        if dataset_outputs:
            all_predictions = {}
            all_targets = {}
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}
            all_inputs = []

            for output in dataset_outputs:
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                for key, value in output["gen_jet_p4s"].items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["reco_jet_p4s"].items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["gen_jet_tau_p4s"].items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

                if output.get("inputs") is not None:
                    all_inputs.append(output["inputs"])

            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(
                    all_gen_jet_tau_p4s[key], axis=0
                )

            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            all_baseline_charges = None
            if self.task == "charge" and all_inputs:
                baseline_chunks = [
                    self._calculate_baseline_charges(inputs) for inputs in all_inputs
                ]
                all_baseline_charges = np.concatenate(baseline_chunks, axis=0)

            self._log_task_metrics(
                targets=all_targets,
                predictions=all_predictions,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                tb_logger=self.logger.experiment,
                current_epoch=self.current_epoch,
                dataset=dataset,
                baseline_charges=all_baseline_charges,
            )

            dataset_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            epoch_metrics = {
                k: torch.stack(v).mean()
                for k, v in self.validation_loss_accumulator.items()
                if v
            }
            for k, v in epoch_metrics.items():
                self.log(f"val_losses/{k}", v)
        self._log_at_epoch_end(dataset="val")

    def on_train_epoch_start(self):
        self.training_loss_accumulator = self._make_accumulator()

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
