import torch.nn as nn

class ComplexOverfit(nn.Module):
    def __init__(self, input_dim):
        super(ComplexOverfit, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): 
        return self.net(x)

class ComplexRegularized(nn.Module):
    def __init__(self, input_dim):
        super(ComplexRegularized, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): 
        return self.net(x)

def get_simple_model(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 1))
