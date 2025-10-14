import torch.nn as nn


class deepmaxent_loss(nn.Module):
    def __init__(self):
        super(deepmaxent_loss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        return loss
    
class deepmaxent_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,hidden_nbr):
        super(deepmaxent_model, self).__init__()
        
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)])
        self.fc3_lambda = nn.Linear(hidden_size, output_size, bias = False)
        
    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu()+x
        x = self.fc3_lambda(x) # agregar (revisar), se le suma el vector de sesgo (que se asocia a los indices)
        
        return x
    
    
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