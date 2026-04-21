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
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layer: int):
        super().__init__()

        # --- backbone (feature extractor) --
        self.fc1 = nn.Linear(input_size, hidden_size)

        # residual hidden blocks: Linear -> ReLU -> add residual
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer)]
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




class IntegratedLoss(nn.Module):
    ## TODO: Check these parameters
    def __init__(self, bias_l2=1e-1, w_pa=2.5, 
                #  clamp_log_rate=(-10, 10) # numerical stability
                clamp_log_rate=None, po_loss='poisson',
                add_area=False
                 ):
        super().__init__()
        self.bias_l2 = bias_l2
        self.pa_w = w_pa
        self.clamp_log_rate = clamp_log_rate
        self.poisson = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
        self.deepmaxent = DeepMaxEntLoss(normalize_target=True)
        self.po_loss = po_loss
        self.add_area = add_area

    def forward(self, base_raw_po, base_raw_pa, bias_raw, yb_po, yb_pa, area_pa):
        # log-rates (raw NN outputs interpreted as log-rate components)
        log_lambda_po = base_raw_po #+ bias_raw          # biased rate for Poisson counts
        log_lambda_pa = base_raw_pa                     # separate rate for Bernoulli PA

        # check shapes of base_raw_pa and area_pa
        # print('base_raw_pa shape:', base_raw_pa.shape)
        # print('area_pa shape:', area_pa.shape)

        if self.add_area:
            # expand area_pa second dimension to match base_raw_pa shape, then add log(area) to log_lambda_pa
            # TODO: Check how to add area and base_raw_pa, taking into account area is just a scaling for the loss
            log_area = np.log(area_pa + 1e-8)  # add small
            log_lambda_pa = log_lambda_pa + torch.tensor(log_area, dtype=log_lambda_pa.dtype, device=log_lambda_pa.device).unsqueeze(1)  # broadcast to [batch, species]


        # # print average raw po and bias, with random chance
        # if np.random.rand() < 0.01:
        #     print('avg base_raw_po:', base_raw_po.mean().item())
        #     print('avg bias_raw:', bias_raw.mean().item())

        if self.clamp_log_rate is not None:
            log_lambda_po = log_lambda_po.clamp(*self.clamp_log_rate)
            log_lambda_pa = log_lambda_pa.clamp(*self.clamp_log_rate)

        # 1) Poisson count NLL using λ_po (TODO: maybe I should try only summing the ones, like in dorazio)
        if self.po_loss == 'bernoulli':
            # treat PO counts as Bernoulli presence/absence
            lambda_po = log_lambda_po.exp()
            log_p0 = -lambda_po                                        # log P(po=0)
            log_p1 = torch.log(-torch.expm1(-lambda_po) + 1e-12)       # log P(po=1), stable

            po = (yb_po > 0).to(log_lambda_po.dtype)
            # loss_poisson = -(po * log_p1 + (1.0 - po) * log_p0).mean()
            loss_po = -(po * log_p1).mean() # only consider presences
        elif self.po_loss == 'poisson':   
            # loss_poisson = self.poisson(log_lambda_po, yb_po)
            lambda_po = log_lambda_po.exp()
            bias_soft = torch.nn.functional.softplus(bias_raw)  # convert bias to positive values
            lambda_po_biased = lambda_po - bias_soft
            #  lambda_po_biased = lambda_po * bias_prob  # apply bias to rate
            # loss_poisson = -((yb_po)*(lambda_po_biased.log_softmax(0))).mean(0).mean()
            loss_po = self.poisson(lambda_po_biased, yb_po)
        elif self.po_loss == 'deepmaxent':
            # softmax_lambda = log_lambda_po.log_softmax(0)  # log_softmax over batch dimension to get probabilities
            # softmax_bias = bias_raw.log_softmax(0)  # bias as log-probabilities, broadcast to species dimension
            # # print('Shapes softmax_lambda:', softmax_lambda.shape, 'softmax_bias:', softmax_bias.shape)
            # print('Sample softmax_lambda:', softmax_lambda[:3])
            # print('Sample softmax_bias:', softmax_bias[:3])
            # # exit(0)

            # lambda_po = log_lambda_po.exp()
            # bias_soft = torch.nn.functional.softplus(bias_raw)  # convert bias to positive values
            # log_lambda_po_biased = log_lambda_po - bias_soft
            # # loss_po = -((yb_po)*(log_lambda_po_biased.log_softmax(0))).mean()
            # loss_po = self.deepmaxent(log_lambda_po_biased, yb_po)

            loss_po = self.deepmaxent(log_lambda_po, yb_po)

            # print('Sample bias:', bias_soft[:3])


            # loss_po = -((yb_po)*(softmax_lambda + softmax_bias)).mean()     


        # 2) Bernoulli PA NLL using p = 1 - exp(-λ_pa)
        lambda_pa = log_lambda_pa.exp()
        log_p0 = -lambda_pa                                        # log P(pa=0)
        log_p1 = torch.log(-torch.expm1(-lambda_pa) + 1e-12)       # log P(pa=1), stable

        pa = yb_pa.to(log_lambda_pa.dtype)
        loss_pa = -(pa * log_p1 + (1.0 - pa) * log_p0).mean()

        # 3) regularize bias toward 0 => exp(bias) toward 1
        loss_reg = (bias_raw ** 2).mean()

        return loss_po + self.pa_w * loss_pa #+ self.bias_l2 * loss_reg



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
    

    


