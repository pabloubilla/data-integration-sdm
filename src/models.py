import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class deepmaxent_loss(nn.Module):
    def __init__(self):
        super(deepmaxent_loss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        return loss




class deepmaxent_model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_nbr: int):
        super().__init__()

        # --- backbone (feature extractor) ---
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


# para cada batch tengo los indices de cada plot


### Version with per-plot bias ###
class deepmaxent_model_w_bias(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr, num_plots):
        super().__init__()
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)]
        )
        # keep no bias here; we’ll add a per-plot bias after this layer
        self.fc3_lambda = nn.Linear(hidden_size, output_size, bias=False)

        # one learnable scalar per plot (shared across all Y)
        self.plot_bias = nn.Embedding(num_plots, 1)
        nn.init.zeros_(self.plot_bias.weight)  # start with no offset

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

        b = self.plot_bias(plot_idx).squeeze(-1)        # [batch]
        logits = logits + b.unsqueeze(1)                # broadcast to [batch, output_size]
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
        


