import torch.nn as nn

class TransformerModel(nn.Module):
    """
    A Transformer-based neural network model for processing state representations
    and producing action outputs.
    """

    def __init__(self, state_dim, embedding_dim, num_heads, num_layers, output_dim):
        """
        Initializes the TransformerModel with embedding, transformer, and fully connected layers.

        Args:
            state_dim (int): Dimension of the input state space.
            embedding_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads in the Transformer layers.
            num_layers (int): Number of Transformer encoder layers.
            output_dim (int): Dimension of the output (e.g., number of actions).
        """

        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(state_dim, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the TransformerModel.

        Args:
            x (Tensor): Input tensor of shape (batch_size, state_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """

        x = self.embedding(x)  # Embedding layer
        x = self.transformer(x)  # Transformer encoder layers
        return self.fc(x.squeeze(1))  # Fully connected layer to produce outputs
