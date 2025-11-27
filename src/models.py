import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class deepmaxent_loss(nn.Module):
    def __init__(self):
        super(deepmaxent_loss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        return loss




class deepmaxent_model(nn.Module):
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

        # # (Optional) small init to keep things stable
        # nn.init.kaiming_uniform_(self.fc1.weight, a=0.0)
        # for l in self.hidden_layers:
        #     nn.init.kaiming_uniform_(l.weight, a=0.0)
        # nn.init.xavier_uniform_(self.output_layer.weight)

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
                 hidden_species=(128,64), hidden_bias=(64,), link="logadd"):
        super().__init__()
        self.species_head = MLP(n_species_covariates, n_species, hidden_species)
        self.bias_head    = MLP(n_bias_covariates, 1, hidden_bias)
        assert link in {"logadd","multiply"}
        self.link = link

        # Optional learnable global intercept per species
        # self.species_intercept = nn.Parameter(torch.zeros(1, n_species))

    def forward(self, X, Z):
        S = self.species_head(X) #+ self.species_intercept  # B x K
        b = self.bias_head(Z) 
        
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

class deepmaxent_loss_w_bias(nn.Module):
    def __init__(self):
        super(deepmaxent_loss_w_bias, self).__init__()
    def forward(self, input1, input2, target):
        loss = -((target)*(input1.log_softmax(0)+input2.log_softmax(0))).mean(0).mean()
        # loss = -((target)*(input1.softmax(0)*input2.softmax(0)).log()).mean(0).mean()
        return loss



# class deepmaxent_model(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size,hidden_nbr):
#         super(deepmaxent_model, self).__init__()
        
#         self.fc1_lambda = nn.Linear(input_size, hidden_size)
#         self.hidden_layers_lambda = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)])
#         self.fc3_lambda = nn.Linear(hidden_size, output_size, bias = False)
        
#     def forward(self, xinput):
#         x = self.fc1_lambda(xinput).relu()
#         for layer in self.hidden_layers_lambda:
#             x = layer(x).relu()+x
#         x = self.fc3_lambda(x) # agregar (revisar), se le suma el vector de sesgo (que se asocia a los indices)
        
#         return x
    
    
# por ahora hay un sesgo por especie por deafult en nn.Linear

# Asymmetric binomial noise model, probability of detection depends on species
class sdm_model_abn(nn.Module):
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
        nn.init.constant_(self.logit_theta, 2.0)  # logit(0.88) ~ 2.0

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


class sdm_loss_abn(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, probs, theta, yobs, q):
        eps = self.eps

        # clamp both before log to avoid log(0)
        probs = probs.clamp(eps, 1 - eps)
        theta_b = theta.unsqueeze(0).expand_as(probs).clamp(eps, 1 - eps)

        # noise term
        ll_noise =  10 * q * (
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


# class loss abn
# class sdm_loss_abn(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, probs, theta, yobs, q):
#         """
#         probs: probability of real y=1 from model, shape [batch, output_size]
#         theta: probability of detection given y=1, shape [output_size]
#         yobs: observed data (0/1), shape [batch, output_size]
#         q: posterior probability (E-step output), shape [batch, output_size]
#         """
#         ll = q * (yobs * torch.log(theta) + (1 - yobs) * torch.log(1 - theta))
#         ll += q * torch.log(probs) + (1 - q) * torch.log(1 - probs)
#         return -ll.mean()


### Version with per-plot bias ###
class deepmaxent_model_w_bias(nn.Module):
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

class deepmaxent_domain(nn.Module):
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
        


