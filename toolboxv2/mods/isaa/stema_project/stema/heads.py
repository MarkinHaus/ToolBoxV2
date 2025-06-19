import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Any
from .data import MultimodalPatchifier  # To get vocab_size dynamically
class TextHead(nn.Module):
    # --- FIX: Takes config directly, not patchifier ---
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        # Get vocab_size from the main config file
        self.vocab_size = config['data']['vocab_size']
        self.decoder = nn.Linear(dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ImageHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.latent_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 4),
        )
        initial_channels = 256
        self.initial_spatial_dims = 4
        self.initial_proj = nn.Sequential(
            nn.Linear(dim * 4, initial_channels * self.initial_spatial_dims * self.initial_spatial_dims),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(initial_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output range [-1, 1]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_embedding = x.mean(dim=1) if x.dim() > 2 else x
        latent_vector = self.latent_proj(mean_embedding)
        initial_feature_map = self.initial_proj(latent_vector)
        img_latent = rearrange(initial_feature_map, 'b (c h w) -> b c h w', h=self.initial_spatial_dims, w=self.initial_spatial_dims)
        generated_image = self.decoder(img_latent)
        return (generated_image + 1) / 2 # Scale to [0, 1] for loss functions

class AudioHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.n_mels = config['data']['audio_n_mels']
        # This head needs to predict a spectrogram of shape (B, S, Mels), but core gives (B,S,D)
        # We need to project from Dim to Mels for each time step.
        self.decoder = nn.Linear(dim, self.n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (B, S, D). Output should be (B, Mels, S_new)
        # For simplicity, we assume one-to-one mapping from input sequence to output sequence.
        mel_output = self.decoder(x) # (B, S, Mels)
        return rearrange(mel_output, 'b s m -> b m s')


class VideoHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.frame_generator = ImageHead(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = x.shape
        reshaped_x = rearrange(x, 'b s d -> (b s) 1 d')
        generated_frames = self.frame_generator(reshaped_x)
        return rearrange(generated_frames, '(b s) c h w -> b s c h w', s=seq_len)


# --- NEW HEAD FOR MPP ---
class EmbeddingPredictionHead(nn.Module):
    """
    A generic head to predict embeddings, typically for Masked Patch Prediction (MPP).
    Input: (Batch, Seq, Dim_in) -> Output: (Batch, Seq, Dim_out)
    Usually Dim_in == Dim_out == model_embedding_dim.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        # Simple linear projection. Could be more complex (e.g., a small MLP).
        self.predictor = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        return self.predictor(x)  # (batch, seq_len, dim)


def get_all_heads(config: Dict[str, Any], *a) -> nn.ModuleDict:
    heads = {
        "text": TextHead(config),
        "image": ImageHead(config),
        "audio": AudioHead(config),
        "video": VideoHead(config),
        # --- ADDED MPP HEAD ---
        "embedding_prediction": EmbeddingPredictionHead(config),
    }
    # Conditionally add heads if specific config options are present, e.g.
    # if config['model'].get('use_mpp_heads', False):
    # heads["video_mpp"] = EmbeddingPredictionHead(config)
    # heads["audio_mpp"] = EmbeddingPredictionHead(config)
    return nn.ModuleDict(heads)
