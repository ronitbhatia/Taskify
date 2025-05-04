import torch
import torch.nn as nn

class TaskMatchModel(nn.Module):
    """
    A hybrid neural network model for task-to-member matching using:
    - Transformer-based encoding for task and team embeddings
    - Cross-attention to align team understanding with task context
    - Structured feature processing via a deep MLP
    - Final classification via a fully connected head
    """

    def __init__(self, embed_dim=384, struct_dim=6):
        """
        Initialize the TaskMatchModel.
        
        Args:
            embed_dim (int): Dimensionality of the task and team embeddings.
            struct_dim (int): Number of structured numerical features.
        """
        super().__init__()

        # Normalize input embeddings
        self.task_norm = nn.LayerNorm(embed_dim)
        self.team_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder for task and team
        self.task_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=512),
            num_layers=2
        )
        self.team_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, dim_feedforward=512),
            num_layers=2
        )

        # Multi-head cross-attention: team attends to task context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=6,
            batch_first=True
        )

        # MLP for structured feature processing
        self.struct_net = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output is a logit; use with BCEWithLogitsLoss
        )

    def forward(self, task_embed, team_embed, struct_feats):
        """
        Forward pass through the model.

        Args:
            task_embed (Tensor): Task embedding tensor of shape (B, 1, embed_dim)
            team_embed (Tensor): Team embedding tensor of shape (B, 1, embed_dim)
            struct_feats (Tensor): Structured feature tensor of shape (B, struct_dim)

        Returns:
            Tensor: Predicted logits for binary classification (B,)
        """
        # Normalize and encode task and team embeddings
        task_embed = self.task_norm(task_embed)
        team_embed = self.team_norm(team_embed)

        task_encoded = self.task_transformer(task_embed.transpose(0, 1)).transpose(0, 1)
        team_encoded = self.team_transformer(team_embed.transpose(0, 1)).transpose(0, 1)

        # Apply cross-attention: team attends to task
        attn_output, _ = self.cross_attention(team_encoded, task_encoded, task_encoded)

        # Flatten encoded outputs
        task_vec = task_encoded.squeeze(1)
        team_vec = attn_output.squeeze(1)

        # Process structured features
        struct_vec = self.struct_net(struct_feats)

        # Concatenate all features and classify
        combined = torch.cat([task_vec, team_vec, struct_vec], dim=1)
        return self.classifier(combined).squeeze(1)
