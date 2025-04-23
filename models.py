import torch.nn as nn

# === Define simple MLP ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.fc1 = nn.Linear(10, 45)
        self.fc2 = nn.Linear(45, 120)
        self.fc3 = nn.Linear(120, 64)
        self.fc4 = nn.Linear(64, 45)
        self.fc5 = nn.Linear(45, 45)
        self.fc6 = nn.Linear(45, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.net(x)
        x = self.relu(self.fc1(x))
        nn.Dropout(0.7)
        x = self.relu(self.fc2(x))
        nn.Dropout(0.7)
        x = self.relu(self.fc3(x))
        nn.Dropout(0.5)
        x = self.relu(self.fc4(x))
        nn.Dropout(0.5)
        # x = self.relu(self.fc5(x))
        # nn.Dropout(0.5)
        x = self.fc6(x)
        return x
