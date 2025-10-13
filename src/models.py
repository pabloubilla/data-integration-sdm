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
        self.fc3_lambda = nn.Linear(hidden_size, output_size)
        
    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu()+x
        x = self.fc3_lambda(x) # agregar (revisar), se le suma el vector de sesgo (que se asocia a los indices)
        
        return x
    
    
# por ahora hay un sesgo por especie por deafult en nn.Linear


# para cada batch tengo los indices de cada plot

