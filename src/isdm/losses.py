"""
Loss functions for PO / PA / integrated species distribution models.

Includes a small registry so tuning scripts can select losses by name
(e.g. for grid search over loss variants) instead of importing classes
directly.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


## Balanced Binary Cross-Entropy Loss: automatically balances to be 50-50
class BalancedBCELoss(nn.Module):
    def __init__(self, eps: float = 1e-8, clamp=None, log_ratio: bool = False):
        super().__init__()
        self.eps = eps
        self.clamp = clamp
        self.log_ratio = log_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  raw model outputs (before sigmoid)
        targets: binary labels {0,1}
        """
        targets = targets.float()

        n_pos = targets.sum()
        n_neg = targets.numel() - n_pos

        pos_weight = n_neg / (n_pos + self.eps)
        pos_weight = pos_weight.to(logits.device, logits.dtype)

        if self.log_ratio:
            pos_weight = 1 + torch.log(pos_weight + 1)

        if self.clamp is not None:
            pos_weight = pos_weight.clamp(max=self.clamp)

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )


## DeepMaxEntLoss: Based on Ryckewaert
class DeepMaxEntLoss(nn.Module):
    def __init__(self, eps=1e-8, pos_weight=None):
        super().__init__()
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, input, target):
        # input:  (B,C)
        # target: (B,C) multi-hot
        normalized_target = target / target.sum(dim=0).clamp_min(self.eps)

        logp = input.log_softmax(dim=0)  # softmax over batch, per class
        loss_num = -(normalized_target * logp).sum()
        loss_den = len(target)
        return loss_num / loss_den


class DeepMaxentLossBias(nn.Module):
    def __init__(self, bias_l2: float = 1e-4):
        super().__init__()
        self.bias_l2 = bias_l2
        self.nll = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")

    def forward(self, input1, input2, target):
        # input1: log-base-rate (or a linear predictor for it)
        # input2: log-bias from covariates
        log_lam = input1 + input2

        poisson = self.nll(log_lam, target)

        # Regularize bias towards 0 => bias factor exp(input2) towards 1
        reg = (input2 ** 2).mean()

        return poisson + self.bias_l2 * reg


class BernoulliFromLogRateLoss(nn.Module):
    """
    Bernoulli likelihood induced by Poisson intensity:

        lambda = exp(log_lambda)
        P(y=1) = 1 - exp(-lambda)
        P(y=0) = exp(-lambda)

    Useful for presence/absence when model outputs log-rate.

    Args:
        positive_only:  If True, only penalise false negatives (ignore true
                        negatives).  Mutually exclusive with balance_pos.
        eps:            Small constant for numerical stability.
        balance_pos:    If True, up-weight positive samples so that positive
                        and negative classes contribute equally to the loss,
                        regardless of class imbalance.  Mutually exclusive
                        with positive_only.
    """

    def __init__(
        self,
        positive_only: bool = False,
        eps: float = 1e-12,
        balance_pos: bool = False,
    ):
        super().__init__()
        if positive_only and balance_pos:
            raise ValueError(
                "`positive_only` and `balance_pos` are mutually exclusive: "
                "balancing requires both positive and negative loss terms."
            )
        self.positive_only = positive_only
        self.eps = eps
        self.balance_pos = balance_pos

    def forward(self, log_lambda: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(log_lambda.dtype)

        lambda_ = torch.exp(log_lambda)
        log_p0 = -lambda_
        log_p1 = torch.log(-torch.expm1(-lambda_) + self.eps)

        if self.positive_only:
            loss = -(target * log_p1)
            return loss.mean()

        loss = -(target * log_p1 + (1.0 - target) * log_p0)

        if self.balance_pos:
            n_pos = target.sum()
            n_neg = target.numel() - n_pos

            pos_weight = (n_neg / (n_pos + self.eps)).to(
                device=log_lambda.device, dtype=log_lambda.dtype
            )
            weight = target * pos_weight + (1.0 - target)
            return (loss * weight).sum() / weight.sum()

        return loss.mean()


class PoissonLogRateLoss(nn.Module):
    """Poisson NLL where model output is log(lambda)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")

    def forward(self, log_lambda, target):
        target = target.to(log_lambda.dtype)
        return self.loss(log_lambda, target)


class BiasL2Penalty(nn.Module):
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight

    def forward(self, bias_raw):
        if bias_raw is None or self.weight <= 0:
            return torch.tensor(0.0)
        return self.weight * (bias_raw ** 2).mean()


class IntegratedLoss(nn.Module):
    """
    Generic two-source loss.

        total = w_source1 * loss_source1(pred_source1, y_source1)
              + w_source2 * loss_source2(pred_source2, y_source2)
              + optional bias regularization

    Does not assume source1 is PO or source2 is PA — that's decided by
    the caller (e.g. run_one_split_popa).
    """

    def __init__(
        self,
        source1_loss: nn.Module,
        source2_loss: nn.Module,
        source1_weight: float = 1.0,
        source2_weight: float = 1.0,
        bias_weight: float = 0.0,
        clamp_source1: Optional[tuple] = None,
        clamp_source2: Optional[tuple] = None,
        concat_sources: bool = False,  # if True uses only source1_loss
    ):
        super().__init__()
        self.source1_loss = source1_loss
        self.source2_loss = source2_loss
        self.source1_weight = source1_weight
        self.source2_weight = source2_weight
        self.bias_weight = bias_weight
        self.clamp_source1 = clamp_source1
        self.clamp_source2 = clamp_source2
        self.concat_sources = concat_sources

    def forward(
        self,
        pred_source1,
        pred_source2,
        target_source1,
        target_source2,
        bias_raw=None,
        extra_source1=None,
        extra_source2=None,
    ):
        if self.clamp_source1 is not None:
            pred_source1 = pred_source1.clamp(*self.clamp_source1)
        if self.clamp_source2 is not None:
            pred_source2 = pred_source2.clamp(*self.clamp_source2)

        if self.concat_sources:
            pred_concat = torch.cat([pred_source1, pred_source2], dim=0)
            target_concat = torch.cat([target_source1, target_source2], dim=0)
            loss = self.source1_loss(pred_concat, target_concat)
            logs = {
                "loss/source1": loss.detach(),
                "loss/source2": loss.detach(),
                "loss/total": loss.detach(),
            }
            return loss, logs

        loss_source1 = self.source1_loss(pred_source1, target_source1)
        loss_source2 = self.source2_loss(pred_source2, target_source2)

        total = self.source1_weight * loss_source1 + self.source2_weight * loss_source2

        loss_bias = None
        if bias_raw is not None and self.bias_weight > 0:
            loss_bias = self.bias_weight * (bias_raw ** 2).mean()
            total = total + loss_bias

        logs = {
            "loss/source1": loss_source1.detach(),
            "loss/source2": loss_source2.detach(),
            "loss/total": total.detach(),
        }
        if loss_bias is not None:
            logs["loss/bias_reg"] = loss_bias.detach()

        return total, logs


## ABN Loss: combines likelihood of observed data under noise model with a noise parameter
class ABNLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, probs, theta, yobs, q):
        eps = self.eps
        probs = probs.clamp(eps, 1 - eps)
        theta_b = theta.unsqueeze(0).expand_as(probs).clamp(eps, 1 - eps)

        ll_noise = 5 * q * (
            (1 - yobs) * torch.log(theta_b) + yobs * torch.log(1 - theta_b)
        )
        ll_prior = q * torch.log(probs) + (1 - q) * torch.log(1 - probs)

        return -(ll_noise + ll_prior).mean()


# ─────────────────────────────────────────────
#  Registries: name → constructor, used by tuning scripts to select
#  losses via a plain string (keeps grid rows/CSVs printable).
# ─────────────────────────────────────────────
LOSS_REGISTRY = {
    "balanced_bce": BalancedBCELoss,
    "deep_maxent": DeepMaxEntLoss,
    "deep_maxent_bias": DeepMaxentLossBias,
    "bernoulli_from_log_rate": BernoulliFromLogRateLoss,
    "poisson_log_rate": PoissonLogRateLoss,
    "abn": ABNLoss,
}