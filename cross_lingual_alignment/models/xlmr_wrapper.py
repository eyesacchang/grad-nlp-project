import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive loss.

    Sits on top of the encoder's final-layer pooled rep. Separates the geometry
    the encoder stores (measured by alignment scorers) from the space optimized
    by InfoNCE, preventing contrastive loss from directly distorting encoder layers.
    """

    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class XLMRWrapper(torch.nn.Module):
    """Wraps XLM-RoBERTa to expose per-layer mean-pooled representations."""

    def __init__(self, model_name="xlm-roberta-base", proj_hidden=2048, proj_out=256):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.num_layers = self.model.config.num_hidden_layers  # 12 for base
        hidden_dim = self.model.config.hidden_size  # 768 for base
        self.projection = ProjectionHead(hidden_dim, proj_hidden, proj_out)

    def forward(self, input_ids, attention_mask):
        """Returns projected embedding for contrastive loss (not raw encoder rep)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return self.projection(pooled)

    def get_all_layer_reps(self, input_ids, attention_mask):
        """Return list of mean-pooled reps for all 12 encoder layers (index 0 = layer 1).

        Uses output_hidden_states=True. hidden_states[0] is the embedding layer;
        hidden_states[i] for i in 1..12 are the transformer layers.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Skip index 0 (embedding layer), return layers 1-12
        layer_reps = [
            self._mean_pool(hidden, attention_mask)
            for hidden in outputs.hidden_states[1:]
        ]
        return layer_reps  # list of 12 tensors, each (B, d)

    def get_layer_rep(self, input_ids, attention_mask, layer_idx):
        """Return mean-pooled rep for a specific layer (1-indexed)."""
        all_reps = self.get_all_layer_reps(input_ids, attention_mask)
        return all_reps[layer_idx - 1]

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        """Mean pool token representations, ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(summed / counts, dim=-1)
