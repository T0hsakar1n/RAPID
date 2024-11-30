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
'''
class MIATransformer(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, hidden_dim=64, num_layers=3, nhead=4, dropout=0.2):
        super(MIATransformer, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.length = input_dim // 2
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=128,
                                                   dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_dim * self.length, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        x1, x2 = x[:, :self.length].unsqueeze(2), x[:, self.length:self.length*2].unsqueeze(2)
        x = torch.cat([x1, x2], dim=-1)
        x = F.gelu(self.fc1(x).permute(1, 0, 2))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(-1, self.hidden_dim * self.length)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

class MIARnn(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(MIARnn, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state
    
class ExtFC(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super(ExtFC, self).__init__()
        self.fcn = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    # nn.Dropout(dropout),
                    nn.Linear(128, output_dim)
                )

    def forward(self, x):
        x = self.fcn(x)
        return x


class ClsFC(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super(ClsFC, self).__init__()
        self.fcn = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    # nn.Dropout(dropout),
                    nn.Linear(128, output_dim)
                )

    def forward(self, x):
        cat_x = torch.cat(x, dim=1)
        x = self.fcn(cat_x)
        return x


class MEMIA(nn.Module):
    def __init__(self, num_submodule=2, input_dim=1, hidden_dim=1, output_dim=1, dropout=0.2):
        super(MEMIA, self).__init__()
        self.num_submodule = num_submodule
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self._create_attack_components()

    def _create_attack_components(self):
        submodule_list = []
        total_hidden_dim = 0
        for i in range(self.num_submodule):
            submodule = ExtFC(self.input_dim, output_dim=self.hidden_dim, dropout=self.dropout)
            submodule_list.append(submodule)
            total_hidden_dim += self.hidden_dim
        self.extractors = nn.ModuleList(submodule_list)
        self.classifier = ClsFC(total_hidden_dim, output_dim=self.output_dim, dropout=self.dropout)

    def forward(self, x_list):
        r"""
        NOTE: The sequence of x_list must be consistent 
              with the extractors.
        """
        enc_x_list = []
        for x, extractor in zip(x_list, self.extractors):
            enc_x_list.append(extractor(x))

        attack_outputs = self.classifier(enc_x_list)
        return attack_outputs