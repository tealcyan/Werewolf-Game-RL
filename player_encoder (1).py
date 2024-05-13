import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        # Activation and initialization settings
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        # Define layers
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim))
        self.fc2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim)
            ) for _ in range(layer_N - 1)
        ])
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), active_func, nn.LayerNorm(output_dim))

        # Apply initialization
        self.apply(lambda m: self.init_(m, init_method, gain))

    def init_(self, m, init_method, gain):
        if isinstance(m, nn.Linear):
            init_method(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.fc2:
            x = layer(x)
        x = self.fc3(x)
        return x
    
player_encoder = MLPLayer(input_dim=253, output_dim=1536, hidden_dim=512, layer_N=3, use_orthogonal=True, use_ReLU=True)


