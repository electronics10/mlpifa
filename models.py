import torch.nn as nn
    
class AntennaMLP(nn.Module):
    def __init__(self):
        super(AntennaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)