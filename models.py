import torch
import torch.nn as nn
from settings import BLOCKS_NUM, OUTPUT_LABELS

class AntennaMLP(nn.Module):
    def __init__(self):
        super(AntennaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(BLOCKS_NUM+1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(OUTPUT_LABELS)),
        )

    def forward(self, x):
        return self.net(x)

class FeedbackGRUModel(nn.Module):
    def __init__(self, input_dim=BLOCKS_NUM+1, output_dim=len(OUTPUT_LABELS), hidden_dim=64, num_iterations=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.gru = nn.GRUCell(hidden_dim + output_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_init = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        y = self.output_init.unsqueeze(0).expand(batch_size, -1)
        h = encoded

        for _ in range(self.num_iterations):
            gru_input = torch.cat([encoded, y], dim=1)
            h = self.gru(gru_input, h)
            y = self.decoder(h)

        return y
