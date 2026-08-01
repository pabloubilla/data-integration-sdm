import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from typing import Optional

import torch
import torch.nn.functional as F






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
    

    


