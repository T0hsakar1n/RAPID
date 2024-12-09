import torch.nn as nn

__all__ = [
    "ColumnFC",
]

class ColumnFC(nn.Module):
    def __init__(self, input_dim=100, output_dim=100, dropout=0.1):
        super(ColumnFC, self).__init__()
        self.num_classes = output_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
