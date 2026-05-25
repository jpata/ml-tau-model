"""
Inference postprocessor: translates raw ParTau model predictions into physical quantities
and packages them as an awkward array.

Model output dict:
  predictions["is_tau"]      shape (N,)    sigmoid score ∈ [0, 1]
  predictions["charge"]      shape (N,)    sigmoid score ∈ [0, 1]  (1 = positive charge)
  predictions["decay_mode"]  shape (N, 6)  softmax probabilities over DM classes [0,1,2,10,11,15]
  predictions["kinematics"]  shape (N, 5)  raw regression:
      [:,0] = log(pt_gen / pt_reco)              → pred_pt   = exp(pred[:,0]) * reco.pt
      [:,1] = delta_eta (gen - reco)             → pred_eta  = pred[:,1] + reco.eta
      [:,2] = sin(delta_phi) (gen - reco)        ↘
      [:,3] = cos(delta_phi) (gen - reco)        → pred_phi  = reco.phi + atan2(sin, cos)
      [:,4] = log(m_gen / m_reco)                → pred_mass = exp(pred[:,4]) * reco.mass

Output awkward array fields per candidate:
  tagging_score        float   sigmoid score (is_tau)
  charge_score         float   sigmoid score (positive charge)
  decay_mode           int     decoded decay mode class ∈ {0, 1, 2, 10, 11, 15}
  decay_mode_probs     float[6]  softmax probabilities for each DM class
  pred_pt              float   [GeV]
  pred_eta             float
  pred_phi             float   [rad]
  pred_energy          float   [GeV]
  pred_mass            float   [GeV]
"""

import os
import glob
import vector
import numpy as np
import awkward as ak

vector.register_awkward()
import lightning as L
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm.auto import tqdm

from mltau.tools.general import reinitialize_p4, one_hot_decoding
from mltau.tools.io.general import BatchInputs
from mltau.tools.io import ParT_dataloader as pdl
from mltau.tools.io import general as ig
# from mltau.tools.io.preprocessed_ParTau_dataloader import ParticleTransformerDataset


# def softmax(x):
#     # x shape: (N, 6)
#     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return e_x / np.sum(e_x, axis=1, keepdims=True)


def to_np(x):
    # --- helpers to convert any tensor/array to numpy ---
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def decode_kinematic_predictions(predictions: dict, reco_jet_p4s: ak.Array) -> ak.Array:
    # --- reco 4-vector ---
    reco = reinitialize_p4(reco_jet_p4s)
    reco_pt = np.asarray(reco.pt)
    reco_eta = np.asarray(reco.eta)
    reco_phi = np.asarray(reco.phi)
    reco_mass = np.asarray(reco.mass)

    # --- decode kinematics ---
    kin = to_np(predictions)  # (N, 5)

    pred_pt = np.exp(kin[:, 0]) * reco_pt
    pred_eta = kin[:, 1] + reco_eta
    pred_phi = reco_phi + np.arctan2(kin[:, 2], kin[:, 3])  # atan2(sin, cos)
    pred_mass = np.exp(kin[:, 4]) * reco_mass

    # energy:  E = sqrt(pt^2 * cosh^2(eta) + m^2)  [via p = pt*cosh(eta)]
    pred_p = pred_pt * np.cosh(pred_eta)
    pred_energy = np.sqrt(pred_p**2 + pred_mass**2)
    # --- build predicted p4 as a vector awkward array (mirrors reinitialize_p4) ---
    pred_p4 = vector.awk(
        ak.zip(
            {
                "pt": pred_pt,
                "eta": pred_eta,
                "phi": pred_phi,
                "energy": pred_energy,
            }
        )
    )
    return pred_p4


def decode_decay_mode_predictions(predictions):
    # --- decode decay mode ---
    dm_probs = to_np(predictions)  # (N, 6)
    # dm_probs = softmax(to_np(predictions["decay_mode"]))  # (N, 6)
    dm_idx = np.argmax(dm_probs, axis=-1)  # (N,) indices 0-5
    dm_class = one_hot_decoding(dm_idx)  # (N,) e.g. {0,1,2,10,11,15}
    return dm_class, dm_probs


def postprocess_multi_predictions(
    predictions: dict, reco_jet_p4s: ak.Array
) -> ak.Array:
    """Decode raw ParTau predictions into physical quantities.

    Args:
        predictions: dict returned by ParTau.forward() / predict_step.
            Values may be torch.Tensor or numpy-convertible arrays.
        reco_jet_p4s: awkward array with reco-jet 4-momentum fields
            (must be compatible with reinitialize_p4).

    Returns:
        ak.Array with fields:
            tagging_score, charge_score, decay_mode, decay_mode_probs,
            pred_p4 (vector p4 with pt, eta, phi, energy)
    """
    # --- pass-through scores ---
    tagging_score = to_np(predictions["is_tau"])  # (N,)
    charge_score = to_np(predictions["charge"])  # (N,)
    dm_class, dm_probs = decode_decay_mode_predictions(predictions["decay_mode"])
    pred_p4 = decode_kinematic_predictions(predictions["kinematics"], reco_jet_p4s)
    return ak.Array(
        {
            "tau_tagging_score": tagging_score,
            "tau_charge_score": charge_score,
            "tau_decay_mode": dm_class,
            "tau_decay_mode_probs": dm_probs,
            "tau_p4": pred_p4,
        }
    )


def postprocess_single_predictions(predictions, reco_jet_p4s, task):
    if task == "kinematics":
        pred_p4 = decode_kinematic_predictions(
            predictions[task], reco_jet_p4s=reco_jet_p4s
        )
        ret = ak.Array({"tau_p4": pred_p4})
    elif task == "decay_mode":
        dm_class, dm_probs = decode_decay_mode_predictions(predictions[task])
        ret = ak.Array(
            {
                "tau_decay_mode": dm_class,
                "tau_decay_mode_probs": dm_probs,
            }
        )
    elif task == "is_tau":
        tagging_score = to_np(predictions[task])  # (N,)
        ret = ak.Array({"tau_tagging_score": tagging_score})
    elif task == "charge":
        charge_score = to_np(predictions[task])  # (N,)
        ret = ak.Array({"tau_charge_score": charge_score})
    else:
        raise NotImplementedError(f"Task '{task}' is not implemented.")
    return ret


def postprocess_predictions(
    predictions, reco_jet_p4s, model_name: str, cfg: DictConfig
):
    if model_name == "MultiParTau":
        ret = postprocess_multi_predictions(
            predictions=predictions, reco_jet_p4s=reco_jet_p4s
        )
    elif model_name == "SingleParTau":
        ret = postprocess_single_predictions(
            predictions, reco_jet_p4s, cfg.training.model.task
        )
    else:
        raise NotImplementedError(f"No such model as {model_name}")
    return ret


def create_predictions_files(
    best_model, cfg: DictConfig, model_name: str, test_only: bool = True
):
    split = "test" if test_only else "*"
    # Match the training-time sample routing:
    # - MultiParTau and SingleParTau tau-ID use all samples
    # - other SingleParTau tasks use only z samples
    sample_pattern = "*"
    if model_name == "SingleParTau" and cfg.training.model.task != "is_tau":
        sample_pattern = "z"

    paths_to_process = glob.glob(
        os.path.join(cfg.dataset.data_dir, f"{sample_pattern}_{split}.parquet")
    )
    if not paths_to_process:
        print(
            "[WARNING] No input files matched for prediction creation:",
            os.path.join(cfg.dataset.data_dir, f"{sample_pattern}_{split}.parquet"),
        )
        return
    print("[INFO] Prediction inputs:")
    for input_path in paths_to_process:
        print(" -", input_path)
    for input_path in paths_to_process:
        create_predictions_file(best_model, input_path, model_name, cfg)


def create_predictions_file(
    best_model, input_path: str, model_name: str, cfg: DictConfig
):
    # Load your .parquet file and build the dataset
    row_groups = ig.get_row_groups([input_path])
    if cfg.training.debug_run:
        row_groups = row_groups[:2]
    dataset = pdl.ParticleTransformerDataset(
        row_groups=row_groups,
        cfg=cfg,
        batch_size=cfg.training.dataloader.batch_size,
    )
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=None)

    # --- Postprocess and save as {sample}_test.parquet ---

    (
        all_gen_jet_p4,
        all_reco_jet_p4,
        all_gen_jet_tau_p4,
        all_gen_jet_tau_decaymode,
        all_gen_jet_tau_charge,
        all_is_tau,
    ) = ([], [], [], [], [], [])
    all_cand_charges = []
    all_cand_p4 = []
    all_post = []

    def _move_to_device(item, device):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, dict):
            return {k: _move_to_device(v, device) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(_move_to_device(v, device) for v in item)
        return item

    try:
        device = next(best_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    best_model.eval()

    progress_desc = f"{model_name} inference on {os.path.basename(input_path)}"
    with tqdm(dataloader, desc=progress_desc, unit="batch") as progress:
        for i, batch in enumerate(progress):
            batch_on_device = _move_to_device(batch, device)
            with torch.no_grad():
                preds = best_model.predict_step(batch_on_device, i)

            inputs = BatchInputs(*batch)
            # Get ground truth fields
            all_gen_jet_p4.append(ak.Array(inputs.gen_jet_p4s))
            all_reco_jet_p4.append(ak.Array(inputs.reco_jet_p4s))
            all_gen_jet_tau_p4.append(ak.Array(inputs.gen_jet_tau_p4s))
            all_gen_jet_tau_decaymode.append(
                inputs.target["decay_mode"].detach().cpu().numpy()
            )
            all_gen_jet_tau_charge.append(
                inputs.target["charge"].detach().cpu().numpy()
            )
            all_is_tau.append(inputs.target["is_tau"].detach().cpu().numpy())

            # --- Candidate mask: 1 for real, 0 for padded ---
            cand_features = (
                inputs.cand_features.detach().cpu().numpy()
            )  # (n_jets, n_cands, n_features)
            cand_kinematics = (
                inputs.cand_kinematics_pxpypze.detach().cpu().numpy()
            )  # (n_jets, n_cands, 4)
            cand_mask = (
                inputs.cand_mask.detach().cpu().numpy().astype(bool)
            )  # (n_jets, n_cands)

            # --- Feature index for charge ---
            charge_idx = 7  # Feature index 7 is charge
            batch_cand_charges = []
            batch_cand_p4 = []
            n_jets, n_cands, _ = cand_features.shape
            for jet_idx in range(n_jets):
                mask = cand_mask[jet_idx]  # (n_cands,)
                real_idx = np.where(mask)[0]
                if len(real_idx) == 0:
                    batch_cand_charges.append(np.array([]))
                    batch_cand_p4.append(
                        {
                            "px": np.array([]),
                            "py": np.array([]),
                            "pz": np.array([]),
                            "energy": np.array([]),
                        }
                    )
                    continue
                charges = cand_features[jet_idx, real_idx, charge_idx]
                kin = cand_kinematics[jet_idx, real_idx, :]
                batch_cand_charges.append(charges)
                batch_cand_p4.append(
                    {
                        "px": kin[:, 0],
                        "py": kin[:, 1],
                        "pz": kin[:, 2],
                        "energy": kin[:, 3],
                    }
                )
            all_cand_charges.append(ak.Array(batch_cand_charges))
            all_cand_p4.append(ak.Array(batch_cand_p4))

            post = postprocess_predictions(
                preds, ak.Array(inputs.reco_jet_p4s), model_name, cfg
            )
            all_post.append(post)

    # Flatten
    gen_jet_p4 = ak.concatenate(all_gen_jet_p4)
    reco_jet_p4 = ak.concatenate(all_reco_jet_p4)
    gen_jet_tau_p4 = ak.concatenate(all_gen_jet_tau_p4)
    decay_mode_target = np.concatenate(all_gen_jet_tau_decaymode)
    charge_target = np.concatenate(all_gen_jet_tau_charge)
    is_tau_target = np.concatenate(all_is_tau).astype(bool)

    # Convert stored training targets back into physical truth labels for output.
    decay_mode_indices = np.argmax(decay_mode_target, axis=-1)
    gen_jet_tau_decaymode = np.where(
        is_tau_target, one_hot_decoding(decay_mode_indices), -1
    )
    gen_jet_tau_charge = np.where(
        is_tau_target,
        np.where(charge_target >= 0.5, 1.0, -1.0),
        np.nan,
    )
    cand_charges = ak.concatenate(all_cand_charges)
    cand_p4 = ak.concatenate(all_cand_p4)
    post = ak.concatenate(all_post)

    # Build output awkward array
    out = ak.zip(
        {
            "gen_jet_p4": gen_jet_p4,
            "reco_jet_p4": reco_jet_p4,
            "gen_jet_tau_p4": gen_jet_tau_p4,
            "gen_jet_tau_decaymode": gen_jet_tau_decaymode,
            "gen_jet_tau_charge": gen_jet_tau_charge,
            "cand_charges": cand_charges,
            "cand_p4": cand_p4,
            **{k: post[k] for k in post.fields},
        },
        depth_limit=1,
    )

    output_file = input_path.split("/")[-1].replace(".pt", ".parquet")
    output_dir = os.path.join(cfg.output_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    ak.to_parquet(out, output_path)
    print(f"[INFO] Saved predictions to {output_path}")
