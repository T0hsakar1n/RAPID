import torch
import torch.nn.functional as F
from torch import nn


class MIAFC(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2):
        super(MIAFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MIATransformer(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=64, num_layers=3, nhead=4, dropout=0.2):
        super(MIATransformer, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.length = input_dim // 3
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(3, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=128,
                                                   dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_dim * self.length, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        x1, x2, x3 = x[:, :self.length].unsqueeze(2), x[:, self.length:self.length*2].unsqueeze(2), \
                     x[:, self.length*2:].unsqueeze(2)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.gelu(self.fc1(x).permute(1, 0, 2))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(-1, self.hidden_dim * self.length)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
