import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class DeepMaxEntLoss(nn.Module):
    def __init__(self):
        super(DeepMaxEntLoss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        # # apply softmax to target as well
        # softmax_target = F.softmax(target, dim=0)
        # loss = -((softmax_target)*(input.log_softmax(0))).mean(0).mean()
        return loss

class DeepMaxEntModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_nbr: int):
        super().__init__()

        # --- backbone (feature extractor) --
        self.fc1 = nn.Linear(input_size, hidden_size)

        # residual hidden blocks: Linear -> ReLU -> add residual
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)]
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




class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 64), dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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

# class deepmaxent_loss_w_bias(nn.Module):
#     def __init__(self):
#         super(deepmaxent_loss_w_bias, self).__init__()

#     def forward(self, input1, input2, target):

#         # Ensure positivity (softplus > 0)
#         rate1 = F.softplus(input1)
#         rate2 = F.softplus(input2)

#         lam = rate1 * rate2 # + self.eps  # avoid log(0)

#         # Negative log-likelihood (dropping log(y!))
#         loss = (lam - target * torch.log(lam)).mean()
#         return loss

#         # loss = -((target)*(input1.log_softmax(0)+input2.log_softmax(0))).mean(0).mean()

#         # loss = -((target)*(input1.log()+input2.log())).mean(0).mean()

#         # loss = -((target)*(input1.softmax(0)*input2.softmax(0)).log()).mean(0).mean()
#         return loss


class deepmaxent_loss_w_bias(nn.Module):
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




class PoissonCountAndPresenceLoss(nn.Module):
    ## TODO: Check these parameters
    def __init__(self, bias_l2=1e-1, pa_w=2.5, 
                #  clamp_log_rate=(-10, 10) # numerical stability
                clamp_log_rate=None
                 ):
        super().__init__()
        self.bias_l2 = bias_l2
        self.pa_w = pa_w
        self.clamp_log_rate = clamp_log_rate
        self.poisson = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")

    def forward(self, base_raw_po, base_raw_pa, bias_raw, yb_po, yb_pa):
        # log-rates (raw NN outputs interpreted as log-rate components)
        log_lambda_po = base_raw_po + bias_raw          # biased rate for Poisson counts
        log_lambda_pa = base_raw_pa                     # separate rate for Bernoulli PA

        if self.clamp_log_rate is not None:
            log_lambda_po = log_lambda_po.clamp(*self.clamp_log_rate)
            log_lambda_pa = log_lambda_pa.clamp(*self.clamp_log_rate)

        # 1) Poisson count NLL using λ_po
        loss_poisson = self.poisson(log_lambda_po, yb_po)

        # 2) Bernoulli PA NLL using p = 1 - exp(-λ_pa)
        lambda_pa = log_lambda_pa.exp()
        log_p0 = -lambda_pa                                        # log P(pa=0)
        log_p1 = torch.log(-torch.expm1(-lambda_pa) + 1e-12)       # log P(pa=1), stable

        pa = yb_pa.to(log_lambda_pa.dtype)
        loss_pa = -(pa * log_p1 + (1.0 - pa) * log_p0).mean()

        # 3) regularize bias toward 0 => exp(bias) toward 1
        loss_reg = (bias_raw ** 2).mean()

        return loss_poisson + self.pa_w * loss_pa #+ #self.bias_l2 * loss_reg


    
# por ahora hay un sesgo por especie por deafult en nn.Linear

# Asymmetric binomial noise model, probability of detection depends on species
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
        # nn.init.constant_(self.logit_theta, 2.0)  # logit(0.88) ~ 2.0

    # def forward(self, xinput):
    #     x = self.fc1_lambda(xinput).relu()
    #     print('x',xinput)
    #     for layer in self.hidden_layers_lambda:
    #         x = layer(x).relu() + x
    #     logits = self.fc3_lambda(x)                     # [batch, output_size]
    #     # apply sigmoid to logits to get probabilities
    #     probs = torch.sigmoid(logits)                  # [batch, output_size]
    #     theta = torch.sigmoid(self.logit_theta)        # [output_size]

    #     return logits, self.logit_theta

    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            # scaled residual to avoid blow-up
            x = x + layer(x).relu()
        logits = self.fc3_lambda(x)           # [batch, output_size]
        probs = torch.sigmoid(logits)         # [batch, output_size]
        theta = torch.sigmoid(self.logit_theta)  # [output_size]
        return logits, self.logit_theta


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
        # print('ll_noise',ll_noise)

        # prior term
        ll_prior = q * torch.log(probs) + (1 - q) * torch.log(1 - probs)
        # print('ll_prior',ll_prior)
        # check if any nan in ll_prior
        # check if any negative in q, 
        # if torch.isnan(ll_prior).any():
        #     print('NaN detected in ll_prior')
        #     # identify indexes where nan occurs
        #     nan_indexes = torch.isnan(ll_prior).nonzero(as_tuple=True)
        #     #print for q
        #     print('q at NaN indexes:', q[nan_indexes])
        #     #print for probs
        #     print('probs at NaN indexes:', probs[nan_indexes])
        #     #print for (1 - q)
        #     print('1 - q at NaN indexes:', (1 - q)[nan_indexes])
        #     print(nan_indexes)

        return -(ll_noise + ll_prior).mean()



### Version with per-plot bias ###
class DeepMaxEntPlotBias(nn.Module):
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
    

    



######### To try domain adaptation #########

class DeepMaxEntDomain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(hidden_nbr):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_size)
    
    def get_features(self, x):
        return self.feature_extractor(x)
    
    def forward(self, x):
        z = self.feature_extractor(x)
        out = torch.sigmoid(self.output_layer(z))
        return out


# --- Gradient reversal layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# --- Domain discriminator (tiny MLP)
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
    




# --- NEW: Two-head shared-trunk model ----------------------------------------
class DeepMaxentTwoHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_nbr: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(hidden_nbr):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        # Heads
        self.pa_head = nn.Linear(hidden_size, output_size)  # used for eval/inference
        
        self.po_head = nn.Linear(hidden_size, output_size)  # used to absorb PO-specific calibration
        # for compatibility with your predict() which expects model.output_layer
        self.output_layer = self.pa_head

    def get_features(self, x):
        return self.trunk(x)

    def forward(self, x, head: str = "pa"):
        z = self.get_features(x)
        if head == "pa":
            return self.pa_head(z)
        elif head == "po":
            return self.po_head(z)
        else:
            raise ValueError("head must be 'pa' or 'po'")
        


