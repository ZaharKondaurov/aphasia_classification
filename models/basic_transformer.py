import torch
from torch import nn

class BasicTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, sequence_length, num_heads=8, num_layers=6, hidden_dim=256, dropout=0.1):
        super(BasicTransformer, self).__init__()

        self.sequence_length = sequence_length

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[..., :self.sequence_length, :]
        encoder_output = self.transformer_encoder(x)

        x = encoder_output.mean(dim=1)
        return self.fc_out(x)
