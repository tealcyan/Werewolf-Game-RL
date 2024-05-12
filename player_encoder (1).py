import numpy as np
import torch.nn as nn

from onpolicy.algorithms.utils.util import init


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)), active_func, nn.LayerNorm(hidden_dim))
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_dim, hidden_dim)), active_func, nn.LayerNorm(hidden_dim)) for i in range(self._layer_N - 1)])
        self.fc3 = nn.Sequential(
            init_(nn.Linear(hidden_dim, output_dim)), active_func, nn.LayerNorm(output_dim))

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N - 1):
            x = self.fc2[i](x)
        x = self.fc3(x)
        return x
    
player_encoder = MLPLayer(input_dim=253, output_dim=1536, hidden_dim=512, layer_N=3, use_orthogonal=True, use_ReLU=True)


