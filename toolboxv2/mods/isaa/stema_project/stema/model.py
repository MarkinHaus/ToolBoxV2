# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from typing import Dict, Any, Optional, Tuple


# --- Primitives (Unverändert) ---
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        threshold = x.abs().mean() * 0.7
        return (x.abs() > threshold).float() * torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, use_ternary: bool = False):
        super().__init__()
        self.use_ternary = use_ternary
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_fp = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_fp', None)
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_fp, a=5 ** 0.5)
        if self.bias_fp is not None: nn.init.zeros_(self.bias_fp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = TernarySTE.apply(self.weight_fp) if self.use_ternary else self.weight_fp
        return F.linear(x, weight, self.bias_fp)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem):
        spk = (mem > 0).float()
        ctx.save_for_backward(mem)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        grad = grad_output * (1 - torch.tanh(mem) ** 2) * 0.5
        return grad


class BioInspiredLIFNeuron(nn.Module):
    def __init__(self, reservoir_size: int, config: Dict[str, Any]):
        super().__init__()
        self.use_spikes = config['model'].get('use_spikes', True)
        self.spike_fn = SurrogateSpike.apply
        self.initial_threshold = 1.0
        self.use_sfa = config['model'].get('use_sfa', True) and self.use_spikes
        beta = torch.full((reservoir_size,), config['model'].get('lif_beta_mean', 0.95))
        self.register_buffer('beta', beta)
        if self.use_sfa:
            sfa_increment = torch.full((reservoir_size,), config['model'].get('sfa_increment_mean', 0.05))
            tau_sfa = torch.full((reservoir_size,), config['model'].get('lif_tau_sfa_mean', 20.0))
            self.register_buffer('sfa_increment', sfa_increment)
            self.register_buffer('sfa_decay', torch.exp(-1.0 / tau_sfa))

    def forward(self, total_current, mem, adaptive_th):
        if not self.use_spikes: return F.gelu(total_current), mem, adaptive_th
        mem = self.beta * mem + total_current
        current_threshold = self.initial_threshold + adaptive_th if self.use_sfa else self.initial_threshold
        spike = self.spike_fn(mem - current_threshold)
        mem = mem * (1.0 - spike.detach())
        if self.use_sfa: adaptive_th = self.sfa_decay * adaptive_th + self.sfa_increment * spike
        return spike, mem, adaptive_th


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# --- Bausteine der neuen, stabilen Architektur ---

class GatedAttention(nn.Module):
    def __init__(self, dim, num_heads, use_ternary):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = TernaryLinear(dim, dim * 3, bias=False, use_ternary=use_ternary)
        self.to_out = TernaryLinear(dim, dim, bias=False, use_ternary=use_ternary)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # Hocheffizient
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, use_ternary):
        super().__init__()
        self.w1 = TernaryLinear(dim, hidden_dim, bias=False, use_ternary=use_ternary)
        self.w2 = TernaryLinear(hidden_dim, dim, bias=False, use_ternary=use_ternary)
        self.w3 = TernaryLinear(dim, hidden_dim, bias=False, use_ternary=use_ternary)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- NEU: Stabiler, skalierbarer Standard-Transformer-Block ---
class STEMA_FFN_Block(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        num_heads = config['model']['num_attn_heads']
        use_ternary = config['model'].get('use_ternary', False)
        ffn_hidden_dim = int(dim * 2.66)  # Standard-Faktor für FFNs

        self.attn_norm = RMSNorm(dim)
        self.attn = GatedAttention(dim, num_heads, use_ternary)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = GatedFeedForward(dim, ffn_hidden_dim, use_ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --- Verbessertes GatedLSM (jetzt ein paralleler Co-Prozessor) ---
class GatedLSM(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        reservoir_size = config['model']['lsm_reservoir_size']
        use_ternary = config['model'].get('use_ternary', False)

        self.input_proj = TernaryLinear(dim, reservoir_size, use_ternary=use_ternary)
        self.recurrent_proj = TernaryLinear(reservoir_size, reservoir_size, bias=False, use_ternary=use_ternary)
        nn.init.orthogonal_(self.recurrent_proj.weight_fp)

        self.lif_reservoir = BioInspiredLIFNeuron(reservoir_size, config)
        self.lsm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # --- VERBESSERUNG: Leistungsstärkerer MLP-Readout ---
        self.readout_norm = RMSNorm(reservoir_size)
        self.readout_mlp = nn.Sequential(
            TernaryLinear(reservoir_size, dim * 2, use_ternary=use_ternary),
            nn.GELU(),
            TernaryLinear(dim * 2, dim, use_ternary=use_ternary)
        )

    def forward(self, x, lsm_states):
        B, T, _ = x.shape
        mem, adaptive_th = lsm_states if lsm_states is not None else (
            torch.zeros(B, self.input_proj.out_features, device=x.device),
            torch.zeros(B, self.input_proj.out_features, device=x.device)
        )
        last_activity = torch.zeros_like(mem)

        output_sequence = []
        for t in range(T):
            xt = x[:, t, :]
            total_current = self.input_proj(xt) + self.recurrent_proj(last_activity)
            activity, mem, adaptive_th = self.lif_reservoir(total_current, mem, adaptive_th)
            output_sequence.append(activity)
            last_activity = activity

        output_sequence = torch.stack(output_sequence, dim=1)

        # Stabiler und effektiverer Readout-Pfad
        normalized_activity = self.readout_norm(output_sequence)
        readout_output = self.readout_mlp(normalized_activity)

        return readout_output, (mem.detach(), adaptive_th.detach())

    def reset_state(self):
        self.lsm_states = None


# --- Encoders (Unverändert) ---
class ImageEncoderBW(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2)
        )
        self.projection = nn.Linear(32 * 7 * 7, dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.projection(x).unsqueeze(1)  # Gibt eine Sequenz der Länge 1 aus

class ImageEncoderRGB(nn.Module):
    """Encodes a SINGLE image (B, C, H, W)"""
    def __init__(self, config):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),      # 224 -> 112
            nn.GELU(), nn.MaxPool2d(2),     # 112 -> 56
            nn.Conv2d(16, 32, 3, 2, 1),     # 56 -> 28
            nn.GELU(), nn.MaxPool2d(2),     # 28 -> 14
            nn.Conv2d(32, 64, 3, 2, 1),     # 14 -> 7
            nn.GELU()
        )
        # --- FIX: Recalculate flattened size for 224x224 input ---
        # Final feature map size is 7x7 with 64 channels
        self.projection = nn.Linear(64 * 7 * 7, dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.projection(x).unsqueeze(1)


class TextEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.token_embedding = nn.Embedding(config['data']['vocab_size'], config['model']['embedding_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(x)


class AudioEncoder(nn.Module):
    """
    Efficient CNN-based feature extractor for audio Mel-spectrograms.
    It processes a 2D spectrogram and converts it into a sequence of embedding vectors,
    one for each time step in the reduced-resolution feature map.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        n_mels = config['data']['audio_n_mels']  # e.g., 128

        # A small 2D CNN to extract features from the spectrogram.
        # It learns to detect patterns across both frequency (n_mels) and time.
        self.encoder = nn.Sequential(
            # Input: (B, 1, n_mels, time_steps)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Downsamples both frequency and time
            # -> (B, 32, n_mels/2, time_steps/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # -> (B, 64, n_mels/4, time_steps/4)
        )

        # After the CNN, we need to calculate the size of the feature vector
        # for each time step. It's the number of output channels times the
        # final, reduced frequency dimension.
        # Example: n_mels=128 -> final_freq_dim = 128 / 4 = 32
        # Feature size = 64 (channels) * 32 (final_freq_dim) = 2048
        final_freq_dim = n_mels // 4
        cnn_output_feature_size = 64 * final_freq_dim

        # A linear layer to project the extracted features for each time step
        # into the main model's embedding dimension.
        self.projection = nn.Linear(cnn_output_feature_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input Mel-spectrogram of shape (B, 1, n_mels, time_steps).
                              The input should be a 4D tensor with a channel dimension.

        Returns:
            torch.Tensor: A sequence of embeddings of shape (B, sequence_length, embedding_dim).
        """
        # 1. Pass spectrogram through the CNN feature extractor
        # x: (B, 1, n_mels, time_steps) -> (B, 64, n_mels/4, time_steps/4)
        x = self.encoder(x)

        # 2. Reshape for sequence processing. We treat the time dimension as our
        #    sequence length and merge the channel and frequency dimensions
        #    to form the features for each time step.
        #
        #    (B, Channels, Freq_dim, Time_dim) -> (B, Time_dim, Channels * Freq_dim)
        #    'b c f t -> b t (c f)' is the einops equivalent of permute and reshape.
        x = rearrange(x, 'b c f t -> b t (c f)')

        # 3. Project the features of each time step to the target embedding dimension.
        # x: (B, time_steps/4, cnn_output_feature_size) -> (B, time_steps/4, embedding_dim)
        x = self.projection(x)

        return x


class VideoEncoder(nn.Module):
    """Encodes a video tensor of frames (B, T, C, H, W)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        in_channels = 3  # RGB frames

        self.encoder = nn.Sequential(
            # Input: (B, C, T, H, W)
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # -> (B, 16, T/2, H/4, W/4)
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # -> (B, 32, T/4, H/8, W/8)
        )
        # For a 224x224 input, final H/W is 224/8=28
        final_h = 224 // 8
        final_w = 224 // 8
        cnn_output_feature_size = 32 * final_h * final_w
        self.projection = nn.Linear(cnn_output_feature_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape is (B, T, C, H, W)
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.encoder(x)
        x = rearrange(x, 'b c t h w -> b t (c h w)')
        x = self.projection(x)
        return x


# --- NEU: Hauptmodell mit entkoppelter, paralleler Architektur ---
class STEMA_Core(nn.Module):
    def __init__(self, config: Dict[str, Any], all_heads: nn.ModuleDict):
        super().__init__()
        dim = config['model']['embedding_dim']

        self.encoders = nn.ModuleDict({
            "imageBW": ImageEncoderBW(config),
            "imageRGB": ImageEncoderRGB(config),
            "text": TextEncoder(config),
            "audio": AudioEncoder(config),
            "video": VideoEncoder(config),
        })
        self.input_norm = RMSNorm(dim)
        self.output_norm = RMSNorm(dim)

        # Pfad 1: Der stabile, tiefe Transformer-Stapel
        self.ffn_layers = nn.ModuleList([STEMA_FFN_Block(config) for _ in range(config['model']['num_layers'])])

        # Pfad 2: Der parallele, rekurrente Co-Prozessor
        self.parallel_lsm = GatedLSM(config)

        # Pfad 3: Der Gate, der die beiden Pfade intelligent fusioniert
        self.fusion_gate = nn.Sequential(
            TernaryLinear(dim, 1, bias=False, use_ternary=config['model'].get('use_ternary', False)),
            nn.Sigmoid()
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, config['model']['max_seq_len'], dim))
        nn.init.normal_(self.pos_embedding, std=.02)

        self.modality_type_embeddings = nn.Embedding(len(self.encoders), dim)
        self.register_buffer('modality_ids', torch.arange(len(self.encoders)))

        self.heads = all_heads
        self.current_task = config['inference'].get('default_task', "classification_head")

    def forward(self, inputs: Dict[str, torch.Tensor], task: Optional[str] = None) -> torch.Tensor:
        # Hier vereinfacht für den Klassifikations-Fall mit einer Sequenzlänge von 1

        processed_sequences = []
        modality_map = {name: i for i, name in enumerate(self.encoders.keys())}

        # Step 1: Encode each modality in parallel using its specific encoder
        for modality_name, input_tensor in inputs.items():
            if modality_name in self.encoders:
                # Get embeddings from the specific encoder
                sequence = self.encoders[modality_name](input_tensor)  # (B, S_mod, D)

                # Add modality-specific type embedding
                modality_id = modality_map[modality_name]
                type_embedding = self.modality_type_embeddings(
                    self.modality_ids.new_full((1,), modality_id)
                )  # (1, D)
                sequence += type_embedding.unsqueeze(1)  # Broadcast across sequence

                processed_sequences.append(sequence)

        if not processed_sequences:
            raise ValueError("No valid modalities found in input dictionary.")

        # Step 2: Combine all sequences into a single sequence
        x = torch.cat(processed_sequences, dim=1)  # (B, S_total, D)

        # Step 3: Add positional embedding and apply input normalization
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.pos_embedding.size(1)}).")
        x = x + self.pos_embedding[:, :seq_len]
        x = self.input_norm(x)

        # Verarbeite den Input in beiden Pfaden parallel
        ffn_out = x
        for layer in self.ffn_layers:
            ffn_out = layer(ffn_out)

        lsm_out, _ = self.parallel_lsm(x, self.parallel_lsm.lsm_states)

        # Berechne den Fusions-Gate
        gate = self.fusion_gate(x)

        # Kombiniere die Ausgaben der beiden Experten
        fused_output = gate * ffn_out + (1.0 - gate) * lsm_out

        x = self.output_norm(fused_output)

        task_to_run = task or self.current_task
        if task_to_run not in self.heads:
            task_to_run = list(self.heads.keys())[0]
        return self.heads[task_to_run](x)

    def reset_all_states(self):
        self.parallel_lsm.reset_state()

    def set_task(self, task: str):
        self.current_task = task

    def get_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Wrapper für Klassifikation ---
class STEMAClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any], num_classes: int):
        super().__init__()
        dim = config['model']['embedding_dim']
        classification_head = nn.Linear(dim, num_classes)
        all_heads = nn.ModuleDict({'classification_head': classification_head})

        self.stema_model = STEMA_Core(config, all_heads)
        self.reset_all_states = self.stema_model.reset_all_states

    def forward(self, x: torch.Tensor, task_type: str = 'imageBW') -> torch.Tensor:
        inputs = {task_type: x}
        # Der Output hat jetzt die Form (B, 1, num_classes), daher squeeze(1)
        return self.stema_model(inputs).squeeze(1)

