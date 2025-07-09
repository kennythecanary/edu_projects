import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch.nn.init import _calculate_fan_in_and_fan_out
import math



class SAEonCLIP(nn.Module):
    def __init__(self, clip_model: str, hook_layer: int, hook_module: str, expansion_factor: int, 
                 geom_dec_bias: bool = False, k: int = None, device: str = "cuda"
        ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model).to(device).requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(clip_model, use_fast=False)

        self.clip.vision_model.encoder.layers[hook_layer]._modules[hook_module].register_forward_hook(self._hook)
        self.stream_out = {}
        self.hook_module = hook_module

        activation_dim = self.clip.visual_projection.in_features
        self.k = k
        activation_fun = nn.ReLU() if k is None else TopK(k)
        self.sae = SparseAutoEncoder(
            activation_dim, activation_dim * expansion_factor, activation_fun, geom_dec_bias).to(device)


    def _hook(self, model, inputs, outputs):
        outputs = outputs[0] if self.hook_module == "self_attn" else outputs
        self.stream_out[self.hook_module] = outputs.detach()
        

    def encode(self, inputs: dict) -> tuple:
        inputs = self.processor(text=inputs["label"], images=inputs["image"], return_tensors="pt", padding=True)
        inputs = inputs.to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip(**inputs)

        representation = self.stream_out[self.hook_module]
        activations, reconstruction =  self.sae(representation)
        
        return representation, activations, reconstruction
            

    def forward(self, inputs: dict) -> tuple:
        representation, activations, reconstruction = self.encode(inputs)
        mse_loss = F.mse_loss(representation, reconstruction)
        l1_loss = F.l1_loss(activations, torch.zeros_like(activations))
        
        with torch.no_grad():
            l0_metric = activations.count_nonzero() / activations.numel()
        
        return mse_loss, l1_loss, l0_metric



class SparseAutoEncoder(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, activation_fun: nn.Module, geom_dec_bias: bool):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.activation_fun = activation_fun

        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        if geom_dec_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.decoder.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.decoder.bias, -bound, bound)

    
    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.activation_fun(self.encoder(activations))
    
    
    def decode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.decoder(activations)

    
    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_representation = self.encode(activations)
        reconstructed_activations = self.decode(encoded_representation)
        return encoded_representation, reconstructed_activations



class TopK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        topk, indices = inputs.topk(10)
        output = torch.zeros(inputs.shape).to(inputs.device)
        return output.scatter_(2, indices, topk)