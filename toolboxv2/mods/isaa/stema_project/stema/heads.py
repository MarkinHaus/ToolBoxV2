import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Any

class TextHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
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
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_embedding = x.mean(dim=1) if x.dim() > 2 else x
        latent_vector = self.latent_proj(mean_embedding)
        initial_feature_map = self.initial_proj(latent_vector)
        img_latent = rearrange(initial_feature_map, 'b (c h w) -> b c h w', h=self.initial_spatial_dims, w=self.initial_spatial_dims)
        generated_image = self.decoder(img_latent)
        return (generated_image + 1) / 2


class AudioHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.n_mels = config['data']['audio_n_mels']
        self.decoder = nn.Linear(dim, self.n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel_output = self.decoder(x)
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


class EmbeddingPredictionHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.predictor = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class ClassificationHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        num_classes = config['training']['num_classes']
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_features = x.mean(dim=1)
        return self.classifier(pooled_features)


def get_all_heads(config: Dict[str, Any]) -> nn.ModuleDict:
    """Factory function to create all available heads based on the config."""
    heads = {
        "text": TextHead(config),
        "image": ImageHead(config),
        "audio": AudioHead(config),
        "video": VideoHead(config),
        "embedding_prediction": EmbeddingPredictionHead(config),
        # The classification head is now used for the linear classifier
        # in the fast FF inference path.
        "classification": ClassificationHead(config),
    }
    return nn.ModuleDict(heads)
