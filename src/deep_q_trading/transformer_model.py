import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, state_dim, embedding_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(state_dim, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.squeeze(1))
