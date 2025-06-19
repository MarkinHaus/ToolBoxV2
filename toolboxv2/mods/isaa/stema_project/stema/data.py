import torch
import torchaudio
from torchvision import transforms
from transformers import AutoTokenizer
import torch.nn as nn
from einops import rearrange
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from typing import Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    _sentence_transformers_available = False
    print("Warning: sentence-transformers library not found. Text embedding will be randomly initialized.")


class MultimodalPatchifier(nn.Module):
    """
    A comprehensive module to convert various modalities (text, image, audio, video)
    into sequences of fixed-dimensional embedding vectors (patches).

    This module is designed to be a plug-and-play preprocessor for the STEMA model.
    It can optionally initialize its projection layers from powerful pre-trained models
    like ViT (for images) and Sentence-Transformers (for text) to accelerate learning.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.dim = config['model']['embedding_dim']

        # --- Text Initialization ---
        self.tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_model'])
        self.text_embed = nn.Embedding(self.tokenizer.vocab_size, self.dim)
        text_init_config = config['pretraining'].get('text_embedding_init', {})
        text_init_method = text_init_config.get('method', 'random')
        if text_init_method == 'sentence_transformer' and _sentence_transformers_available:
            try:
                sbert_model_name = text_init_config.get('model_name', 'all-mpnet-base-v2')
                print(f"Initializing text embedder with SentenceTransformer: {sbert_model_name}")
                sbert_model = SentenceTransformer(sbert_model_name, device=str(device))
                sbert_dim = sbert_model.get_sentence_embedding_dimension()
                # Create a bridge from SBERT embeddings to our model's dimension
                if sbert_dim != self.dim:
                    print(f"Warning: SBERT dim ({sbert_dim}) != STEMA dim ({self.dim}). Projection may be needed.")
                # Note: This is a placeholder for a more complex fine-tuning strategy.
                # For our main pre-training, the nn.Embedding layer is sufficient.
            except Exception as e:
                print(f"Could not load SentenceTransformer model: {e}. Using random init for text embeddings.")


        # --- Image Initialization ---
        self.img_patch_size = config['data']['image_patch_size']
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # A simple but effective Conv2D to create patches from an image.
        # LazyConv2d infers the input channels (e.g., 3 for RGB) automatically.
        self.img_proj = nn.LazyConv2d(self.dim, kernel_size=self.img_patch_size, stride=self.img_patch_size)

        # --- Audio Initialization ---
        audio_cfg = config['data']
        vocoder_cfg = config['vocoder']
        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg.get('audio_sample_rate', 16000),
            n_mels=audio_cfg['audio_n_mels'],
            n_fft=vocoder_cfg['n_fft'],
            hop_length=vocoder_cfg['hop_length'],
            win_length=vocoder_cfg['win_length'],
        )
        # Create patches from the 2D Mel Spectrogram
        self.audio_proj = nn.LazyConv2d(
            self.dim,
            kernel_size=audio_cfg['audio_patch_kernel_size'],
            stride=audio_cfg['audio_patch_stride']
        )

    def process_text(self, text: str) -> torch.Tensor:
        """Converts a string of text into a sequence of embedding patches."""
        tokens = self.tokenizer(
            text, return_tensors='pt',
            max_length=self.config['data']['max_text_len'],
            padding=False,  # We handle padding in the collate_fn
            truncation=True
        )
        embeddings = self.text_embed(tokens.input_ids.to(self.device))
        return embeddings # (1, seq_len, dim)

    def process_image(self, img_pil: Image.Image) -> torch.Tensor:
        """Converts a PIL image into a sequence of embedding patches."""
        img_tensor = self.img_transform(img_pil).unsqueeze(0).to(self.device)
        # Initialize lazy module if needed on first forward pass
        if isinstance(self.img_proj, nn.modules.lazy.LazyModuleMixin):
            _ = self.img_proj(img_tensor)
        patches = self.img_proj(img_tensor) # (B, dim, H_patch, W_patch)
        return rearrange(patches, 'b d h w -> b (h w) d') # (B, num_patches, dim)

    def process_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Converts an audio waveform into a sequence of embedding patches."""
        target_sr = self.config['data'].get('audio_sample_rate', 16000)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr).to(self.device)
            waveform = resampler(waveform)
        if waveform.ndim > 1 and waveform.shape[0] > 1: # Handle stereo
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = self.audio_transform(waveform.to(self.device))
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.unsqueeze(1) # Add channel dimension

        if isinstance(self.audio_proj, nn.modules.lazy.LazyModuleMixin):
            _ = self.audio_proj(mel_spec)

        patches = self.audio_proj(mel_spec)
        return rearrange(patches, 'b d h w -> b (h w) d') # (B, num_patches, dim)

    def process_video(self, video_path: str) -> torch.Tensor:
        """Converts a video file into a sequence of embedding patches from its frames."""
        try:
            clip = VideoFileClip(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.empty(1, 0, self.dim, device=self.device)

        frame_patches_list = []
        fps = self.config['data'].get('video_fps_sampling', 1)

        try:
            for frame_data in clip.iter_frames(fps=fps, dtype='uint8'):
                img = Image.fromarray(frame_data).convert("RGB")
                patches = self.process_image(img) # (1, num_patches, dim)
                frame_patches_list.append(patches.squeeze(0))
        except Exception as e:
            print(f"Error processing frames for video {video_path}: {e}")
        finally:
            clip.close()

        if not frame_patches_list:
            return torch.empty(1, 0, self.dim, device=self.device)

        # Concatenate patches from all frames to form one long sequence for the video
        return torch.cat(frame_patches_list, dim=0).unsqueeze(0) # (1, total_patches, dim)

    def to(self, device: torch.device):
        """Moves all submodules to the specified device."""
        self.device = device
        self.text_embed.to(device)
        self.img_proj.to(device)
        self.audio_proj.to(device)
        self.audio_transform.to(device)
        return self
