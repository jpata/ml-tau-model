import os
import glob
import math
import torch
import numpy as np
import awkward as ak

from collections.abc import Sequence
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig
from lightning import LightningDataModule

from mltau.tools.io import general as ig  # RowGroupDataset
from mltau.tools import general as g
from mltau.tools import features as f

np.random.seed(42)


class ParticleTransformerDataset(IterableDataset):
    def __init__(
        self, row_groups: Sequence[ig.RowGroup], cfg: DictConfig, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.")

    def __len__(self):
        return math.ceil(self.num_rows / self.batch_size)

    def _get_p4(self, data, prefix):
        """Helper to extract p4 fields regardless of coordinate system (cylindrical or Cartesian)"""
        if prefix in data.fields:
            p4_data = data[prefix]
        elif f"{prefix}s" in data.fields:
            p4_data = data[f"{prefix}s"]
        else:
            raise KeyError(f"Could not find p4 field with prefix {prefix}")

        fields = p4_data.fields
        if "rho" in fields:
            # Already cylindrical
            return (
                p4_data["rho"],
                p4_data["eta"],
                p4_data["phi"],
                p4_data["t"],
            )
        elif "x" in fields:
            # Cartesian (x, y, z, tau=mass)
            # Use awkward operations for conversion to avoid vector/multiprocessing issues
            x = p4_data["x"]
            y = p4_data["y"]
            z = p4_data["z"]
            m = p4_data["tau"]
            
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            # eta = arcsinh(z/rho)
            eta = np.arcsinh(z / ak.where(rho > 0, rho, 1e-6))
            energy = np.sqrt(x**2 + y**2 + z**2 + m**2)
            
            return (
                rho,
                eta,
                phi,
                energy,
            )
        else:
            raise ValueError(f"Unknown p4 coordinate system in {prefix}: {fields}")

    def build_tensors(self, data: ak.Array):
        max_cands = self.cfg.dataset.max_cands
        eps = 1e-6

        # ------------------------------------------------------------------
        # Helper: pad jagged awkward array → dense float32 [N, max_cands]
        # ------------------------------------------------------------------
        def pad_cand(arr, fill=0.0):
            return ak.to_numpy(
                ak.fill_none(ak.pad_none(arr, max_cands, clip=True), fill)
            ).astype(np.float32)

        # ------------------------------------------------------------------
        # Candidate p4 components
        # ------------------------------------------------------------------
        cand_pt_jagged, cand_eta_jagged, cand_phi_jagged, cand_en_jagged = self._get_p4(
            data, "reco_cand_p4"
        )
        cand_pt = pad_cand(cand_pt_jagged)
        cand_eta = pad_cand(cand_eta_jagged)
        cand_phi = pad_cand(cand_phi_jagged)
        cand_en = pad_cand(cand_en_jagged)

        # Handle field name variations
        def get_field(data, names):
            for name in names:
                if name in data.fields:
                    return data[name]
            return None

        reco_cand_charges = get_field(data, ["reco_cand_charges", "reco_cand_charge"])
        reco_cand_pdgs = get_field(data, ["reco_cand_pdgs", "reco_cand_pdg"])
        reco_cand_dz = get_field(data, ["reco_cand_dz"])
        reco_cand_dz_error = get_field(
            data, ["reco_cand_dz_error", "reco_cand_dz_err"]
        )
        reco_cand_dxy = get_field(data, ["reco_cand_dxy"])
        reco_cand_dxy_error = get_field(
            data, ["reco_cand_dxy_error", "reco_cand_dxy_err"]
        )

        cand_charge = pad_cand(reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(reco_cand_pdgs))
        cand_dz = pad_cand(reco_cand_dz)
        cand_dz_err = pad_cand(reco_cand_dz_error)
        cand_dxy = pad_cand(reco_cand_dxy)
        cand_dxy_err = pad_cand(reco_cand_dxy_error)

        # Mask: True = real particle, False = padding  [N, max_cands]
        lengths = np.minimum(ak.to_numpy(ak.num(reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Scalar jet p4s
        jet_pt, jet_eta, jet_phi, jet_en = [
            ak.to_numpy(x).astype(np.float32)
            for x in self._get_p4(data, "reco_jet_p4")
        ]

        _pt_gen, _eta_gen, _phi_gen, _energy_gen = [
            ak.to_numpy(x).astype(np.float32)
            for x in self._get_p4(data, "gen_jet_tau_p4")
        ]

        _pt_gen_jet, _eta_gen_jet, _phi_gen_jet, _energy_gen_jet = [
            ak.to_numpy(x).astype(np.float32)
            for x in self._get_p4(data, "gen_jet_p4")
        ]

        # ------------------------------------------------------------------
        # Compute 17 ParticleTransformer features in numpy (zero awkward)
        # ParticleTransformer features from https://arxiv.org/pdf/2202.03772, table 2
        # Broadcast jet scalars [N] → [N, 1] against candidates [N, max_cands]
        # ------------------------------------------------------------------
        jpt = jet_pt[:, None]
        jeta = jet_eta[:, None]
        jphi = jet_phi[:, None]
        jen = jet_en[:, None]

        cand_deta = np.abs(cand_eta - jeta)
        dphi_raw = cand_phi - jphi
        cand_dphi = np.abs(np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw)))
        cand_logpt = np.log(np.maximum(cand_pt, eps))
        cand_loge = np.log(np.maximum(cand_en, eps))
        cand_logptrel = np.log(np.maximum(cand_pt / np.maximum(jpt, eps), eps))
        cand_logerel = np.log(np.maximum(cand_en / np.maximum(jen, eps), eps))
        cand_dR = np.sqrt(cand_deta**2 + cand_dphi**2)

        isElectron = (cand_pdg_abs == 11).astype(np.float32)
        isMuon = (cand_pdg_abs == 13).astype(np.float32)
        isPhoton = (cand_pdg_abs == 22).astype(np.float32)
        isChargedHadron = (cand_pdg_abs == 211).astype(np.float32)
        isNeutralHadron = (cand_pdg_abs == 130).astype(np.float32)

        # Stack → [N, 17, max_cands], zero padded slots, fix nan/inf
        cand_features_np = np.stack(
            [
                cand_deta,
                cand_dphi,
                cand_logpt,
                cand_loge,
                cand_logptrel,
                cand_logerel,
                cand_dR,
                cand_charge,
                isElectron,
                isMuon,
                isPhoton,
                isChargedHadron,
                isNeutralHadron,
                cand_dz,
                cand_dz_err,
                cand_dxy,
                cand_dxy_err,
            ],
            axis=1,
        )  # [N, 17, max_cands]
        cand_features_np *= mask_np[:, None, :]
        np.nan_to_num(cand_features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Cand kinematics: (px, py, pz, energy) → [N, 4, max_cands]
        cand_px = cand_pt * np.cos(cand_phi)
        cand_py = cand_pt * np.sin(cand_phi)
        cand_pz = cand_pt * np.sinh(cand_eta)
        cand_kinematics_np = np.stack([cand_px, cand_py, cand_pz, cand_en], axis=1)
        cand_kinematics_np *= mask_np[:, None, :]

        # ------------------------------------------------------------------
        # Weights, decay mode, charge
        # ------------------------------------------------------------------
        if "cls_weight" not in data.fields:
            weight_tensors = torch.ones(len(data), dtype=torch.float32)
        else:
            weight_tensors = torch.from_numpy(
                ak.to_numpy(data.cls_weight).astype(np.float32)
            )

        gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
        reduced_gen_decay_modes = g.get_reduced_decaymodes(gen_jet_tau_decaymode)
        ohe_prepared_decay_modes = g.prepare_one_hot_encoding(reduced_gen_decay_modes)
        gen_jet_tau_decaymode_reduced = torch.from_numpy(
            ohe_prepared_decay_modes.astype(np.int64)
        )
        gen_jet_tau_decaymode_ohe = torch.nn.functional.one_hot(
            gen_jet_tau_decaymode_reduced, 6
        ).float()
        gen_jet_tau_decaymode_exists = torch.from_numpy(
            (gen_jet_tau_decaymode != -1).astype(np.int64)
        )
        if "gen_jet_tau_charge" in data.fields:
            charge_tensor = torch.from_numpy(
                (ak.to_numpy(data.gen_jet_tau_charge).astype(np.int32) == 1).astype(
                    np.float32
                )
            )
        else:
            charge_tensor = torch.zeros(len(data), dtype=torch.float32)

        # ------------------------------------------------------------------
        # Kinematics regression targets (pure numpy, no reinitialize_p4)
        # ------------------------------------------------------------------
        _deta = _eta_gen - jet_eta
        _dphi_raw = _phi_gen - jet_phi
        _dphi = np.arctan2(np.sin(_dphi_raw), np.cos(_dphi_raw))
        _vis_pt_ratio = np.maximum(_pt_gen / np.maximum(jet_pt, eps), eps)
        # m^2 = E^2 - pt^2 * cosh^2(eta)
        _mass_gen = np.sqrt(
            np.maximum(_energy_gen**2 - (_pt_gen * np.cosh(_eta_gen)) ** 2, 0.0)
        )
        _mass_reco = np.sqrt(
            np.maximum(jet_en**2 - (jet_pt * np.cosh(jet_eta)) ** 2, 0.0)
        )
        _vis_m_ratio = np.maximum(_mass_gen / np.maximum(_mass_reco, eps), eps)
        kinematics_tensor = torch.from_numpy(
            np.stack(
                [
                    np.log(_vis_pt_ratio),
                    _deta,
                    np.sin(_dphi),
                    np.cos(_dphi),
                    np.log(_vis_m_ratio),
                ],
                axis=-1,
            )
        )

        return (
            torch.from_numpy(cand_features_np),
            torch.from_numpy(cand_kinematics_np),
            {
                "kinematics": kinematics_tensor.float(),
                "decay_mode": gen_jet_tau_decaymode_ohe.float(),
                "charge": charge_tensor.float(),
                "is_tau": gen_jet_tau_decaymode_exists.long(),
            },
            torch.from_numpy(mask_np).unsqueeze(1),  # [N, 1, max_cands]
            weight_tensors.float(),
            {
                "pt": torch.from_numpy(_pt_gen),
                "eta": torch.from_numpy(_eta_gen),
                "phi": torch.from_numpy(_phi_gen),
                "energy": torch.from_numpy(_energy_gen),
            },
            {
                "pt": torch.from_numpy(jet_pt),
                "eta": torch.from_numpy(jet_eta),
                "phi": torch.from_numpy(jet_phi),
                "energy": torch.from_numpy(jet_en),
            },
            {
                "pt": torch.from_numpy(_pt_gen_jet),
                "eta": torch.from_numpy(_eta_gen_jet),
                "phi": torch.from_numpy(_phi_gen_jet),
                "energy": torch.from_numpy(_energy_gen_jet),
            },
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(
                math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            row_groups_start = worker_id * per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]

        # Define all possible columns we might need across different dataset versions
        _POTENTIAL_COLUMNS = [
            "reco_cand_p4", "reco_cand_p4s",
            "reco_cand_charge", "reco_cand_charges",
            "reco_cand_pdg", "reco_cand_pdgs",
            "reco_cand_dz",
            "reco_cand_dz_err", "reco_cand_dz_error",
            "reco_cand_dxy",
            "reco_cand_dxy_err", "reco_cand_dxy_error",
            "reco_jet_p4", "reco_jet_p4s",
            "gen_jet_tau_p4", "gen_jet_tau_p4s",
            "gen_jet_p4", "gen_jet_p4s",
            "gen_jet_tau_decaymode",
            "gen_jet_tau_charge",
            "cls_weight",
        ]

        # Cache fields for each filename to avoid repeated heavy I/O
        file_fields_cache = {}

        import gc

        for row_group in row_groups_to_process:
            if row_group.filename not in file_fields_cache:
                # Use ak.from_parquet with first row_group to just get the fields
                tmp_data = ak.from_parquet(row_group.filename, row_groups=[0])
                file_fields_cache[row_group.filename] = tmp_data.fields
            
            file_fields = file_fields_cache[row_group.filename]
            columns_to_load = [c for c in _POTENTIAL_COLUMNS if c in file_fields]

            data = ak.from_parquet(
                row_group.filename,
                row_groups=[row_group.row_group],
                columns=columns_to_load,
            )
            tensors = self.build_tensors(data)
            N = tensors[0].shape[0]

            # Yield pre-batched slices — bypasses PyTorch per-sample collation entirely
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                yield (
                    tensors[0][start:end],  # cand_features
                    tensors[1][start:end],  # cand_kinematics
                    {k: v[start:end] for k, v in tensors[2].items()},  # targets
                    tensors[3][start:end],  # mask
                    tensors[4][start:end],  # weights
                    {k: v[start:end] for k, v in tensors[5].items()},  # gen_jet_tau_p4s
                    {k: v[start:end] for k, v in tensors[6].items()},  # reco_jet_p4s
                    {k: v[start:end] for k, v in tensors[7].items()},  # gen_jet_p4s
                )
            
            # Explicitly delete to help GC
            del data
            del tensors
            gc.collect()


class ParTDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        debug_run: bool = False,
    ):
        """Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.

        """
        self.cfg = cfg
        use_bkg = (cfg.training.model.task == "is_tau") or (
            cfg.training.model.name == "MultiParTau"
        )
        self.debug_run = debug_run
        self.sample = "z" if not use_bkg else "*"
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_row_groups = 10 if debug_run else None
        self.save_hyperparameters()
        super().__init__()

    def get_dataset_rowgroups(self, dataset_type: str):
        if dataset_type == "test":
            test_paths_wcp = os.path.join(
                self.cfg.dataset.data_dir, f"{self.sample}_test.parquet"
            )
            test_paths = list(glob.glob(test_paths_wcp))
            test_rowgroups = ig.get_row_groups(input_paths=test_paths)
            np.random.shuffle(test_rowgroups)
            if self.num_row_groups:
                test_rowgroups = test_rowgroups[: self.num_row_groups]
            return test_rowgroups
        elif dataset_type == "train":
            total = sum(
                [
                    self.cfg.dataset.relative_sizes[dataset]
                    for dataset in ["train", "val"]
                ]
            )
            fractions = {
                dataset: self.cfg.dataset.relative_sizes[dataset] / total
                for dataset in ["train", "val"]
            }
            train_paths_wcp = os.path.join(
                self.cfg.dataset.data_dir, f"{self.sample}_train.parquet"
            )
            train_paths = list(glob.glob(train_paths_wcp))
            all_train_rowgroups = ig.get_row_groups(input_paths=train_paths)
            np.random.shuffle(all_train_rowgroups)
            if self.num_row_groups:
                all_train_rowgroups = all_train_rowgroups[: self.num_row_groups]
            n_train_rowgroups = int(len(all_train_rowgroups) * fractions["train"])
            train_rowgroups = all_train_rowgroups[:n_train_rowgroups]
            val_rowgroups = all_train_rowgroups[n_train_rowgroups:]
            return train_rowgroups, val_rowgroups
        else:
            return []

    def setup(self, stage: str) -> None:
        # For debug runs, use smaller but reasonable batch size for speed
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        if stage == "fit":
            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )
            self.train_dataset = ParticleTransformerDataset(
                row_groups=train_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            self.val_dataset = ParticleTransformerDataset(
                row_groups=val_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            # batch_size=None: dataset yields pre-batched slices, skip collation entirely
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=None,
                persistent_workers=(
                    False
                    if (
                        self.debug_run
                        or self.cfg.training.dataloader.num_dataloader_workers == 0
                    )
                    else True
                ),
                num_workers=(
                    0
                    if self.debug_run
                    else self.cfg.training.dataloader.num_dataloader_workers
                ),
                multiprocessing_context=(
                    None
                    if (
                        self.debug_run
                        or self.cfg.training.dataloader.num_dataloader_workers <= 1
                    )
                    else None  # Using None to default to 'fork' on Linux
                ),
                prefetch_factor=(
                    self.cfg.training.dataloader.prefetch_factor
                    if (
                        not self.debug_run
                        and self.cfg.training.dataloader.num_dataloader_workers > 0
                    )
                    else None
                ),
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=None,
                persistent_workers=(
                    False
                    if (
                        self.debug_run
                        or self.cfg.training.dataloader.num_dataloader_workers == 0
                    )
                    else True
                ),
                num_workers=(
                    0
                    if self.debug_run
                    else self.cfg.training.dataloader.num_dataloader_workers
                ),
                multiprocessing_context=(
                    None
                    if (
                        self.debug_run
                        or self.cfg.training.dataloader.num_dataloader_workers <= 1
                    )
                    else None  # Using None to default to 'fork' on Linux
                ),
                prefetch_factor=(
                    self.cfg.training.dataloader.prefetch_factor
                    if (
                        not self.debug_run
                        and self.cfg.training.dataloader.num_dataloader_workers > 0
                    )
                    else None
                ),
                pin_memory=True,
            )
        elif stage == "test" or stage == "predict":
            test_row_groups = self.get_dataset_rowgroups(dataset_type="test")
            self.test_dataset = ParticleTransformerDataset(
                row_groups=test_row_groups, cfg=self.cfg, batch_size=batch_size
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=None,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                prefetch_factor=(
                    self.cfg.training.dataloader.prefetch_factor
                    if self.cfg.training.dataloader.num_dataloader_workers > 0
                    else None
                ),
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader
