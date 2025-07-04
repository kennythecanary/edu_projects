import numpy as np
import torch
import torch.nn.functional as F
import datasets
import torchvision



def im_process(processor, batch: datasets.Dataset) -> torch.Tensor:
    batch = processor(text=batch["label"], images=batch["image"], return_tensors="pt", do_normalize=False)
    return batch.pixel_values


def extract_patches(batch: torch.Tensor, size: int = 16) -> torch.Tensor:
    return batch.unfold(2, size, size).unfold(3, size, size)


def merge_patches(batch: torch.Tensor) -> torch.Tensor:
    B, C, N, _, size, _ = batch.shape
    batch = batch.reshape(B,C,-1, size*size).permute(0,1,3,2).view(B*C, size*size, -1)
    batch = F.fold(batch, output_size=(N*size, N*size), kernel_size=size, stride=size)
    batch = batch.reshape(B, C, N*size, N*size)
    return batch


def generate_mask(unfold_batch: torch.Tensor, activations: torch.Tensor, k: int, neuron: int) -> torch.Tensor:
    activation_mask = activations[1:,k-1] == neuron
    channel_mask = unfold_batch[:,:,:1,0,0].bool()
    channel_mask = torch.cat([channel_mask if i!=k-1 else ~channel_mask for i in range(3)], 2)
    mask = torch.logical_and(activation_mask[:,None,None], channel_mask)
    return mask


def mask_image(model, batch: torch.Tensor, activations: torch.Tensor, k: int, neuron: int, patch_size: int = 16) -> torch.Tensor:
    B, C, H, W = batch.shape
    unfold_batch = extract_patches(batch).permute(2,3,0,1,4,5).reshape(-1, B, C, patch_size, patch_size)
    mask = generate_mask(unfold_batch, activations, k, neuron)
    unfold_batch[mask] = torch.zeros(unfold_batch[mask].shape)
    unfold_batch = unfold_batch.reshape(H//patch_size, W//patch_size, B, C, patch_size, patch_size).permute(2,3,0,1,4,5)
    reconstruction = merge_patches(unfold_batch)
    return reconstruction


def make_grid(batch: datasets.Dataset, nrow: int) -> np.array:
    batch_tensor = torch.tensor(np.stack([np.array(img) for img in batch["image"]])).permute(0,3,1,2)
    grid = torchvision.utils.make_grid(batch_tensor, nrow=nrow)
    return grid.permute(1,2,0).numpy()