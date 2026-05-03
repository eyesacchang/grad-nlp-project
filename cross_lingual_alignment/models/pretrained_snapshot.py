import copy
import torch
from models.xlmr_wrapper import XLMRWrapper


class PretrainedSnapshot:
    """Frozen copy of the pretrained model for regularization.

    Stores reference representations so we can penalize drift from the pretrained geometry.
    All parameters are frozen (no_grad).
    """

    def __init__(self, model: XLMRWrapper, device):
        self.model = copy.deepcopy(model)
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def get_all_layer_reps(self, input_ids, attention_mask):
        """Return frozen mean-pooled reps for all 12 layers."""
        return self.model.get_all_layer_reps(input_ids, attention_mask)

    @torch.no_grad()
    def get_layer_rep(self, input_ids, attention_mask, layer_idx):
        return self.model.get_layer_rep(input_ids, attention_mask, layer_idx)
