import torch.nn as nn

class PoseGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=4):
        super(PoseGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)        # h_n: [num_layers, batch, hidden_dim]
        out = self.fc(h_n[-1])      # last layer hidden
        return out