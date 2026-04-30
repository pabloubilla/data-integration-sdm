import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


import torch
import torch.nn.functional as F

## Balanced Binary Cross-Entropy Loss: automatically balances to be 50-50
class BalancedBCELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

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

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight
        )
        return loss

## DeepMaxEntLoss: Based on Ryckewaert
class DeepMaxEntLoss(nn.Module):
    def __init__(self, eps=1e-8, pos_weight=None):
        super().__init__()
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, input, target):
        # input:  (B,C)
        # target: (B,C) multi-hot

        w = 1.0
        if self.pos_weight is not None:
            w = self.pos_weight.unsqueeze(0)  # (1,C)

        # Only count positive entries in the normalization
        weighted_target = target * w

        logp = input.log_softmax(dim=0)  # softmax over batch, per class
        loss_num = -(weighted_target * logp).sum()
        loss_den = weighted_target.sum().clamp_min(self.eps)
        return loss_num / loss_den



class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int):
        super().__init__()

        # --- backbone (feature extractor) --
        self.fc1 = nn.Linear(input_size, hidden_size)

        # residual hidden blocks: Linear -> ReLU -> add residual
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )

        # expose a handle called "feature_extractor" so code outside can freeze it
        # (just keep references to same modules)
        self.feature_extractor = nn.ModuleDict({
            "fc1": self.fc1,
            "hidden_layers": self.hidden_layers
        })

        # --- head (output layer) ---
        # keep bias=False like your original
        self.output_layer = nn.Linear(hidden_size, output_size, bias=False)


    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute backbone features (before final linear head)."""
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            h = F.relu(layer(x))
            x = x + h              # residual connection
        return x                   # feature vector z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_features(x)
        logits = self.output_layer(z)  # logits; apply sigmoid outside if needed
        return logits



class SDMWithBias(nn.Module):
    """    species_head: B x C -> B x K
    bias_head:    B x D -> B x 1
    Combines them to produce lambda: B x K
    """
    def __init__(self, n_species_covariates, n_species, n_bias_covariates,
                 hidden_species=(128,64), hidden_bias=(64,64), output_bias=1):
        super().__init__()
        self.species_head = MLP(n_species_covariates, n_species, hidden_species)
        self.bias_head    = MLP(n_bias_covariates, output_bias, hidden_bias)
        # assert link in {"logadd","multiply"}
        # self.link = link

        # Optional learnable global intercept per species
        # self.species_intercept = nn.Parameter(torch.zeros(1, n_species))

    def forward(self, X, Z = None):
        S = self.species_head(X) #+ self.species_intercept  # B x K
        if Z is not None:
            b = self.bias_head(Z) 
        else:
            b = None
        
        return S, b                              # B x 1

        if self.link == "logadd":
            # λ = exp(S + b), broadcasting b over species dimension
            log_lambda = S + b
            lam = torch.exp(log_lambda)
            return lam, log_lambda
        else:
            # λ = softplus(S) * softplus(b)
            lam = F.softplus(S) * F.softplus(b)            # broadcast
            # define a pseudo log for convenience (not used in multiply mode)
            log_lambda = torch.log(lam + 1e-8)
            return lam, log_lambda


class DeepMaxentLossBias(nn.Module):
    def __init__(self, bias_l2=1e-4):
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
    """

    def __init__(self, positive_only: bool = False, eps: float = 1e-12):
        super().__init__()
        self.positive_only = positive_only
        self.eps = eps

    def forward(self, log_lambda, target):
        target = target.to(log_lambda.dtype)

        lambda_ = torch.exp(log_lambda)
        log_p0 = -lambda_
        log_p1 = torch.log(-torch.expm1(-lambda_) + self.eps)

        if self.positive_only:
            loss = -(target * log_p1)
        else:
            loss = -(target * log_p1 + (1.0 - target) * log_p0)

        return loss.mean()


class PoissonLogRateLoss(nn.Module):
    """
    Poisson NLL where model output is log(lambda).
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.PoissonNLLLoss(
            log_input=True,
            full=False,
            reduction="mean",
        )

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

    Example:
        total = w_source1 * loss_source1(pred_source1, y_source1)
              + w_source2 * loss_source2(pred_source2, y_source2)
              + optional bias regularization

    It does not assume that source1 is PO or source2 is PA.
    You decide that in the experiment script.
    """

    def __init__(
        self,
        source1_loss: nn.Module,
        source2_loss: nn.Module,
        source1_weight: float = 1.0,
        source2_weight: float = 1.0,
        bias_weight: float = 0.0,
        clamp_source1: tuple[float, float] | None = None,
        clamp_source2: tuple[float, float] | None = None,
    ):
        super().__init__()

        self.source1_loss = source1_loss
        self.source2_loss = source2_loss

        self.source1_weight = source1_weight
        self.source2_weight = source2_weight

        self.bias_weight = bias_weight

        self.clamp_source1 = clamp_source1
        self.clamp_source2 = clamp_source2

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

        loss_source1 = self.source1_loss(pred_source1, target_source1)
        loss_source2 = self.source2_loss(pred_source2, target_source2)

        total = (
            self.source1_weight * loss_source1
            + self.source2_weight * loss_source2
        )

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




# Asymmetric binomial noise model, probability of detection depends on species (Not used for now)
class ABNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr):
        super().__init__()
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)]
        )
        self.fc3_lambda = nn.Linear(hidden_size, output_size, bias=False)  

        # theta parameter
        self.logit_theta = nn.Parameter(torch.zeros(output_size))  # one per species
        # initialize with closer to 1 probability
        # nn.init.constant_(self.logit_theta, 2.0)  # logit(0.88) ~ 
    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            # scaled residual to avoid blow-up
            x = x + layer(x).relu()
        logits = self.fc3_lambda(x)           # [batch, output_size]
        return logits, self.logit_theta


## ABN Loss: combines likelihood of observed data under noise model with a noise parameter
class ABNLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, probs, theta, yobs, q):
        eps = self.eps

        # clamp both before log to avoid log(0)
        probs = probs.clamp(eps, 1 - eps)
        theta_b = theta.unsqueeze(0).expand_as(probs).clamp(eps, 1 - eps)

        # noise term
        ll_noise =  5 * q * (
            (1 - yobs) * torch.log(theta_b) +
            (yobs) * torch.log(1 - theta_b)
        )
        ll_prior = q * torch.log(probs) + (1 - q) * torch.log(1 - probs)

        return -(ll_noise + ll_prior).mean()



### Version with per-plot bias ###
class MLPWithPlotBias(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr, num_plots,
                 separate = True):
        super().__init__()
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)]
        )
        # keep no bias here; we’ll add a per-plot bias after this layer
        self.fc3_lambda = nn.Linear(hidden_size, output_size, bias=False)

        # one learnable scalar per plot (shared across all Y)
        self.plot_bias = nn.Embedding(num_plots, 1)
        nn.init.zeros_(self.plot_bias.weight)  

        self.separate = separate

    def forward(self, xinput, plot_idx):
        """
        xinput:  [batch, input_size]
        plot_idx: [batch] LongTensor with the plot ID for each row in xinput
        """
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu() + x
        logits = self.fc3_lambda(x)                     # [batch, output_size]
        if plot_idx is None:
            return logits  # no bias if no indices provided
        b = self.plot_bias(plot_idx)# .squeeze(-1)        # [batch]

        if self.separate:
            return logits, b

        logits = logits + b# .unsqueeze(1)                # broadcast to [batch, output_size]
        return logits
    

    


