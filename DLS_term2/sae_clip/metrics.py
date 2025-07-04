import torch
import torch.nn.functional as F


def compute_mse_loss(representation: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    mse_loss = F.mse_loss(representation, reconstruction, reduction="none")
    return mse_loss.reshape(representation.shape[0], -1).mean(1)

def compute_sparsity(activations: torch.Tensor) -> torch.Tensor:
    activations = activations.reshape(activations.shape[0],-1)
    return activations.count_nonzero(1) / activations.size(1)

def compute_mean_activation_value(activations: torch.Tensor) -> torch.Tensor:
    activations = activations.reshape(activations.shape[0],-1)
    return activations.sum(1) / activations.count_nonzero(1)

def compute_label_entropy(activations: torch.Tensor) -> torch.Tensor:
    prob_c = activations.reshape(activations.shape[0],-1).sum(1) / activations.sum()
    return -prob_c * prob_c.log()