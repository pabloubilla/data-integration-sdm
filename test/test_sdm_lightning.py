import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L


# -----------------------------
# 1) Model kind (no string names)
# -----------------------------
class ModelKind(Enum):
    PO_ONLY = auto()
    PO_ONLY_ABN = auto()
    PO_ONLY_COV_BIAS = auto()


@dataclass
class TrainConfig:
    kind: ModelKind
    # data sizes
    n_train: int = 20000
    n_val: int = 4000
    n_cov: int = 20          # covariates for species model
    n_bias_cov: int = 2      # covariates for bias model (Z)
    n_species: int = 10

    # model/training
    hidden_size: int = 128
    hidden_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    num_workers: int = 2
    max_epochs: int = 3

    # ABN knobs (toy)
    abn_lambda: float = 0.1  # weight of auxiliary/theta regularizer

    # device / lightning
    accelerator: str = "auto"  # "mps" is great on Apple Silicon
    devices: int = 1
    logger: bool = False       # keep it simple (no lightning_logs)


# -----------------------------
# 2) Synthetic dataset (dict batch like your SDMDataset)
# -----------------------------
class SDMDictDataset(Dataset):
    """
    Returns dicts with:
      - "X": (n_cov,)
      - "Z": (n_bias_cov,)  optional (only used for cov-bias model)
      - "Y": (n_species,)
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, Z: Optional[torch.Tensor] = None):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"X": self.X[idx], "Y": self.Y[idx]}
        if self.Z is not None:
            item["Z"] = self.Z[idx]
        return item


def make_synthetic_po_pa(
    n: int,
    n_cov: int,
    n_bias_cov: int,
    n_species: int,
    seed: int = 0,
    include_bias: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Synthetic multi-label problem. We simulate logits from a linear model plus noise.
    Y is multi-label Bernoulli.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n_cov, generator=g)

    # "true" species weights
    W = torch.randn(n_cov, n_species, generator=g) * 0.7
    base_logits = X @ W + 0.2 * torch.randn(n, n_species, generator=g)

    if include_bias:
        Z = torch.randn(n, n_bias_cov, generator=g)
        # bias affects prevalence (adds a shared bias term per species)
        B = torch.randn(n_bias_cov, n_species, generator=g) * 0.5
        logits = base_logits + Z @ B
    else:
        Z = None
        logits = base_logits

    probs = torch.sigmoid(logits)
    Y = torch.bernoulli(probs)

    return X.float(), Y.float(), (Z.float() if Z is not None else None)


# -----------------------------
# 3) Toy models (swap with your real ones later)
# -----------------------------
def mlp(in_dim: int, out_dim: int, hidden_size: int, hidden_layers: int) -> nn.Module:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, hidden_size), nn.ReLU()]
        d = hidden_size
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class POOnlyModel(nn.Module):
    def __init__(self, n_cov: int, n_species: int, hidden_size: int, hidden_layers: int):
        super().__init__()
        self.net = mlp(n_cov, n_species, hidden_size, hidden_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)  # logits


class ABNModel(nn.Module):
    """
    Toy ABN-style: returns (logit_p, logit_theta)
    where logit_theta is an auxiliary head.
    """
    def __init__(self, n_cov: int, n_species: int, hidden_size: int, hidden_layers: int):
        super().__init__()
        self.shared = mlp(n_cov, hidden_size, hidden_size, max(1, hidden_layers - 1))
        self.head_p = nn.Linear(hidden_size, n_species)
        self.head_theta = nn.Linear(hidden_size, n_species)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(X)
        return self.head_p(h), self.head_theta(h)


class CovBiasModel(nn.Module):
    """
    Species logits from X, plus a bias logits term from Z.
    Returns (logits, bias_logits).
    """
    def __init__(self, n_cov: int, n_bias_cov: int, n_species: int, hidden_size: int, hidden_layers: int):
        super().__init__()
        self.species_net = mlp(n_cov, n_species, hidden_size, hidden_layers)
        self.bias_net = mlp(n_bias_cov, n_species, hidden_size, max(1, hidden_layers // 2))

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.species_net(X)
        bias_logits = self.bias_net(Z)
        return logits, bias_logits


# -----------------------------
# 4) Losses (toy versions; swap with your imports)
# -----------------------------
def bce_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


def abn_loss(logit_p: torch.Tensor, logit_theta: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Simple ABN-like objective:
      - primary BCE on p
      - regularize theta to stay small (toy stand-in for your q/theta logic)
    """
    return bce_logits(logit_p, y) + lam * torch.mean(torch.sigmoid(logit_theta) ** 2)


def cov_bias_loss(logits: torch.Tensor, bias_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Combine logits + bias_logits then BCE.
    In your real code you may use a special bias-aware loss.
    """
    return bce_logits(logits + bias_logits, y)


# -----------------------------
# 5) AUC helper (no sklearn dependency)
# -----------------------------
@torch.no_grad()
def per_species_auc_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[int, float]:
    """
    Computes a simple AUC per species using a rank-based approximation.
    Works when each column has both positives and negatives.
    """
    # y_true,y_score: [N, S]
    N, S = y_true.shape
    aucs: Dict[int, float] = {}

    for s in range(S):
        yt = y_true[:, s]
        ys = y_score[:, s]
        pos = yt > 0.5
        neg = ~pos
        n_pos = int(pos.sum().item())
        n_neg = int(neg.sum().item())
        if n_pos == 0 or n_neg == 0:
            aucs[s] = float("nan")
            continue

        # rank-based AUC: (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        # implement via argsort ranks
        order = torch.argsort(ys)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, N + 1, device=ys.device, dtype=torch.float32)
        sum_pos_ranks = ranks[pos].sum()
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs[s] = float(auc.item())

    return aucs


# -----------------------------
# 6) LightningDataModule
# -----------------------------
class SDMDataModule(L.LightningDataModule):
    def __init__(self, cfg: TrainConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        include_bias = self.cfg.kind == ModelKind.PO_ONLY_COV_BIAS

        Xtr, Ytr, Ztr = make_synthetic_po_pa(
            n=self.cfg.n_train,
            n_cov=self.cfg.n_cov,
            n_bias_cov=self.cfg.n_bias_cov,
            n_species=self.cfg.n_species,
            seed=self.seed,
            include_bias=include_bias,
        )
        Xva, Yva, Zva = make_synthetic_po_pa(
            n=self.cfg.n_val,
            n_cov=self.cfg.n_cov,
            n_bias_cov=self.cfg.n_bias_cov,
            n_species=self.cfg.n_species,
            seed=self.seed + 999,
            include_bias=include_bias,
        )

        self.train_ds = SDMDictDataset(Xtr, Ytr, Ztr)
        self.val_ds = SDMDictDataset(Xva, Yva, Zva)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=(self.cfg.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=(self.cfg.num_workers > 0),
        )


# -----------------------------
# 7) LightningModule with routing by ModelKind
# -----------------------------
class SDMLit(L.LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Factory for model creation
        if cfg.kind == ModelKind.PO_ONLY:
            self.model = POOnlyModel(cfg.n_cov, cfg.n_species, cfg.hidden_size, cfg.hidden_layers)
        elif cfg.kind == ModelKind.PO_ONLY_ABN:
            self.model = ABNModel(cfg.n_cov, cfg.n_species, cfg.hidden_size, cfg.hidden_layers)
        elif cfg.kind == ModelKind.PO_ONLY_COV_BIAS:
            self.model = CovBiasModel(cfg.n_cov, cfg.n_bias_cov, cfg.n_species, cfg.hidden_size, cfg.hidden_layers)
        else:
            raise ValueError(f"Unsupported kind: {cfg.kind}")

        self._val_scores = []
        self._val_targets = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, raw_logits_for_auc).
        raw_logits_for_auc should be the thing you want to score with sigmoid.
        """
        X = batch["X"]
        Y = batch["Y"]

        if self.cfg.kind == ModelKind.PO_ONLY:
            logits = self.model(X)
            loss = bce_logits(logits, Y)
            return loss, logits

        if self.cfg.kind == ModelKind.PO_ONLY_ABN:
            logit_p, logit_theta = self.model(X)
            loss = abn_loss(logit_p, logit_theta, Y, lam=self.cfg.abn_lambda)
            return loss, logit_p

        if self.cfg.kind == ModelKind.PO_ONLY_COV_BIAS:
            Z = batch["Z"]
            logits, bias_logits = self.model(X, Z)
            loss = cov_bias_loss(logits, bias_logits, Y)
            return loss, (logits + bias_logits)

        raise RuntimeError("Unhandled kind.")

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self._val_scores.append(logits.detach().cpu())
        self._val_targets.append(batch["Y"].detach().cpu())

    def on_validation_epoch_end(self):
        scores = torch.cat(self._val_scores, dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        probs = torch.sigmoid(scores)
        aucs = per_species_auc_torch(targets, probs)
        avg_auc = float(np.nanmean(list(aucs.values())))

        self.log("val_avg_auc", avg_auc, prog_bar=True)

        # print a small summary so you can see it immediately
        # (optional, remove if you prefer silent)
        if self.trainer.is_global_zero:
            first3 = list(aucs.items())[:3]
            print(f"\nval_avg_auc={avg_auc:.4f} | first species aucs={first3}\n")

        self._val_scores.clear()
        self._val_targets.clear()


# -----------------------------
# 8) Run helper
# -----------------------------
def run(cfg: TrainConfig, seed: int = 0):
    L.seed_everything(seed, workers=True)

    dm = SDMDataModule(cfg, seed=seed)
    lit = SDMLit(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=cfg.logger,              # False -> no lightning_logs/version_*
        enable_checkpointing=False,     # keep it minimal
        log_every_n_steps=20,
    )
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    # Pick ONE to test quickly:
    # kind = ModelKind.PO_ONLY
    # kind = ModelKind.PO_ONLY_ABN
    kind = ModelKind.PO_ONLY_COV_BIAS

    cfg = TrainConfig(
        kind=kind,
        max_epochs=50,
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
        batch_size=512,
        num_workers=2,
        logger=True,
    )

    run(cfg, seed=42)
