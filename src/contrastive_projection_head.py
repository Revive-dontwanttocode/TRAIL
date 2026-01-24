import torch
import torch.nn as nn


class ContrastiveProjectionHead(nn.Module):
    """
    A SimCLR style projection head with two MLP layers.
    """
    def __init__(self, in_dim, hid_dim=1024, out_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, out_dim, bias=False)
        self.ln = nn.LayerNorm(out_dim)

        nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)


    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        z = self.fc2(x)
        z = self.ln(z.to(torch.float32)).to(x.dtype)
        return z