# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from typing import Dict, Any, Optional, Tuple

from .heads import get_all_heads


# --- Primitives (Unchanged) ---
# TernarySTE, TernaryLinear, SurrogateSpike, BioInspiredLIFNeuron, RMSNorm...
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        threshold = x.abs().mean() * 0.7
        return (x.abs() > threshold).float() * torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output): return grad_output


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_ternary=False):
        super().__init__();
        self.use_ternary = use_ternary;
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features));
        if bias:
            self.bias_fp = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_fp', None)
        self.out_features = out_features;
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_fp, a=5 ** 0.5);
        if self.bias_fp is not None: nn.init.zeros_(self.bias_fp)

    def forward(self, x):
        weight = TernarySTE.apply(self.weight_fp) if self.use_ternary else self.weight_fp
        return F.linear(x, weight, self.bias_fp)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem): spk = (mem > 0).float(); ctx.save_for_backward(mem); return spk

    @staticmethod
    def backward(ctx, grad_output): (mem,) = ctx.saved_tensors; grad = grad_output * (
            1 - torch.tanh(mem) ** 2) * 0.5; return grad


class BioInspiredLIFNeuron(nn.Module):
    def __init__(self, reservoir_size, config):
        super().__init__();
        self.use_spikes = config['model'].get('use_spikes', True);
        self.spike_fn = SurrogateSpike.apply;
        self.initial_threshold = 1.0
        self.use_sfa = config['model'].get('use_sfa', True) and self.use_spikes
        beta = torch.full((reservoir_size,), config['model'].get('lif_beta_mean', 0.95));
        self.register_buffer('beta', beta)
        if self.use_sfa:
            sfa_increment = torch.full((reservoir_size,), config['model'].get('sfa_increment_mean', 0.05));
            tau_sfa = torch.full((reservoir_size,), config['model'].get('lif_tau_sfa_mean', 20.0))
            self.register_buffer('sfa_increment', sfa_increment);
            self.register_buffer('sfa_decay', torch.exp(-1.0 / tau_sfa))

    def forward(self, total_current, mem, adaptive_th):
        if not self.use_spikes: return F.gelu(total_current), mem, adaptive_th
        mem = self.beta * mem + total_current;
        current_threshold = self.initial_threshold + adaptive_th if self.use_sfa else self.initial_threshold
        spike = self.spike_fn(mem - current_threshold);
        mem = mem * (1.0 - spike.detach())
        if self.use_sfa: adaptive_th = self.sfa_decay * adaptive_th + self.sfa_increment * spike
        return spike, mem, adaptive_th


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6): super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# --- Core FFN and Attention Primitives (Unchanged) ---
class GatedAttention(nn.Module):
    def __init__(self, dim, num_heads, use_ternary):
        super().__init__();
        self.num_heads = num_heads;
        self.head_dim = dim // num_heads;
        self.scale = self.head_dim ** -0.5
        self.to_qkv = TernaryLinear(dim, dim * 3, bias=False, use_ternary=use_ternary)
        self.to_out = TernaryLinear(dim, dim, bias=False, use_ternary=use_ternary)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1);
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v);
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, use_ternary):
        super().__init__();
        self.w1 = TernaryLinear(dim, hidden_dim, bias=False, use_ternary=use_ternary)
        self.w2 = TernaryLinear(hidden_dim, dim, bias=False, use_ternary=use_ternary)
        self.w3 = TernaryLinear(dim, hidden_dim, bias=False, use_ternary=use_ternary)

    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- NEW: Hybrid Block for Unified Backprop + FF Training ---
class HybridForwardBlock(nn.Module):
    """
    A unified block that supports backpropagation while calculating an
    auxiliary Forward-Forward loss for regularization.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        num_heads = config['model']['num_attn_heads']
        use_ternary = config['model'].get('use_ternary', False)
        ffn_hidden_dim = int(dim * 2.66)
        dropout_rate = config['model'].get('dropout_rate', 0.1)
        self.threshold = config['model'].get('ff_goodness_threshold', 2.0)

        self.attn = GatedAttention(dim, num_heads, use_ternary)
        self.attn_norm = RMSNorm(dim)
        self.ffn = GatedFeedForward(dim, ffn_hidden_dim, use_ternary)
        self.ffn_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the main data path (x_pos) with full gradient flow
        and computes an auxiliary FF loss using x_neg.
        """
        # --- Main Path (Positive Data) with Backpropagation ---
        # Note: No .detach() is used here. Gradients flow freely.
        x_res = x_pos + self.attn(self.attn_norm(x_pos))
        x_out = x_res + self.ffn(self.ffn_norm(x_res))
        x_out = self.dropout(x_out)

        # --- Auxiliary Path (Negative Data) for FF Loss ---
        # This part runs in parallel but only for loss calculation.
        # We use torch.no_grad() to ensure this path doesn't contribute
        # to the main backprop graph, saving computation.
        with torch.no_grad():
            x_neg_res = x_neg + self.attn(self.attn_norm(x_neg))
            x_neg_out = x_neg_res + self.ffn(self.ffn_norm(x_neg_res))

        # --- Local Auxiliary FF Loss Calculation ---
        g_pos = x_out.pow(2).mean([1, 2])
        g_neg = x_neg_out.pow(2).mean([1, 2])
        ff_loss = F.softplus(torch.cat([-g_pos + self.threshold, g_neg - self.threshold])).mean()

        return x_out, ff_loss


# --- Parallel LSM Co-processor (Now with Dropout for stability) ---
class GatedLSM(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        reservoir_size = config['model']['lsm_reservoir_size']
        use_ternary = config['model'].get('use_ternary', False)
        dropout_rate = config['model'].get('dropout_rate', 0.1)

        self.input_proj = TernaryLinear(dim, reservoir_size, use_ternary=use_ternary)
        self.recurrent_proj = TernaryLinear(reservoir_size, reservoir_size, bias=False, use_ternary=use_ternary)
        nn.init.orthogonal_(self.recurrent_proj.weight_fp)
        self.lif_reservoir = BioInspiredLIFNeuron(reservoir_size, config)
        self.lsm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.readout_norm = RMSNorm(reservoir_size)
        self.readout_mlp = nn.Sequential(
            TernaryLinear(reservoir_size, dim * 2, use_ternary=use_ternary),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Added dropout
            TernaryLinear(dim * 2, dim, use_ternary=use_ternary)
        )

    def forward(self, x, lsm_states):
        # ... forward logic remains the same ...
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
        normalized_activity = self.readout_norm(output_sequence)
        readout_output = self.readout_mlp(normalized_activity)
        return readout_output, (mem.detach(), adaptive_th.detach())

    def reset_state(self): self.lsm_states = None


# --- Encoders (Unchanged) ---
class ImageEncoder(nn.Module):  # ... Unchanged
    def __init__(self, config):
        super().__init__();
        dim = config['model']['embedding_dim'];
        in_channels = config['data'].get('image_in_channels', 3)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 16, 3, 2, 1), nn.GELU(), nn.MaxPool2d(2),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 2, 1),
                                     nn.GELU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.projection = nn.Linear(64, dim)

    def forward(self, x): x = self.encoder(x); x = x.flatten(1); x = self.projection(x); return x.unsqueeze(1)


class TextEncoder(nn.Module):  # ... Unchanged
    def __init__(self, config): super().__init__(); self.token_embedding = nn.Embedding(config['data']['vocab_size'],
                                                                                        config['model'][
                                                                                            'embedding_dim'])

    def forward(self, x): return self.token_embedding(x)


class AudioEncoder(nn.Module):  # ... Unchanged
    def __init__(self, config):
        super().__init__();
        dim = config['model']['embedding_dim'];
        n_mels = config['data']['audio_n_mels']
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.GELU(), nn.MaxPool2d(2))
        final_freq_dim = n_mels // 4;
        cnn_output_feature_size = 64 * final_freq_dim
        self.projection = nn.Linear(cnn_output_feature_size, dim)

    def forward(self, x): x = self.encoder(x); x = rearrange(x, 'b c f t -> b t (c f)'); x = self.projection(
        x); return x


class VideoEncoder(nn.Module):  # ... Unchanged
    def __init__(self, config):
        super().__init__();
        dim = config['model']['embedding_dim'];
        in_channels = 3
        self.encoder = nn.Sequential(nn.Conv3d(in_channels, 16, (3, 3, 3), (1, 2, 2), (1, 1, 1)), nn.GELU(),
                                     nn.MaxPool3d((2, 2, 2)), nn.Conv3d(16, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                                     nn.GELU(), nn.MaxPool3d((2, 2, 2)))
        final_h, final_w = 224 // 8, 224 // 8;
        cnn_output_feature_size = 32 * final_h * final_w
        self.projection = nn.Linear(cnn_output_feature_size, dim)

    def forward(self, x): x = rearrange(x, 'b t c h w -> b c t h w'); x = self.encoder(x); x = rearrange(x,
                                                                                                         'b c t h w -> b t (c h w)'); x = self.projection(
        x); return x


class STEMA_Core(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dim = config['model']['embedding_dim']
        self.config = config
        self.ff_loss_weight = config['training'].get('ff_loss_weight', 0.5)

        self.encoders = nn.ModuleDict(
            {"image": ImageEncoder(config), "text": TextEncoder(config), "audio": AudioEncoder(config),
             "video": VideoEncoder(config)})
        self.input_norm = RMSNorm(dim)
        self.output_norm = RMSNorm(dim)

        # Unified blocks for hybrid training
        self.hybrid_layers = nn.ModuleList([HybridForwardBlock(config) for _ in range(config['model']['num_layers'])])
        self.parallel_lsm = GatedLSM(config)
        self.fusion_gate = nn.Sequential(TernaryLinear(dim, 1, bias=False), nn.Sigmoid())

        # Embeddings - No longer need class embeddings for input
        self.pos_embedding = nn.Parameter(torch.empty(1, config['model']['max_seq_len'], dim))
        nn.init.normal_(self.pos_embedding, std=.02)

        # Use generic heads for output
        self.heads = get_all_heads(config)

    def _encode_and_prepare(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes multiple modalities into a single sequence."""
        processed_sequences = []
        for modality_name, input_tensor in inputs.items():
            if modality_name in self.encoders:
                sequence = self.encoders[modality_name](input_tensor)
                processed_sequences.append(sequence)

        x = torch.cat(processed_sequences, dim=1)
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Input seq len ({seq_len}) > max_seq_len ({self.pos_embedding.size(1)}).")
        x = x + self.pos_embedding[:, :seq_len]
        return self.input_norm(x)

    def forward(self, inputs: Dict[str, torch.Tensor], y: Optional[torch.Tensor] = None,
                task: str = 'classification') -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        A single, unified forward pass for hybrid training or inference.

        If `y` is provided (training), returns (logits, total_loss).
        If `y` is None (inference), returns logits.
        """
        x = self._encode_and_prepare(inputs)
        total_ff_loss = 0.0

        # --- Generate Negative Data for FF Loss (only if training) ---
        if self.training and y is not None:
            # Create negative data by pairing with shuffled inputs from the same batch
            # This is a simple but effective self-supervised contrastive method
            rand_indices = torch.randperm(x.size(0))
            x_neg = x[rand_indices]
        else:
            # For inference, negative path is not needed for loss, so we can feed dummy data
            x_neg = torch.zeros_like(x)

        # --- Deep Recurrent Data Flow ---
        lsm_state = None  # Reset LSM state for each batch
        for layer in self.hybrid_layers:
            # 1. Fuse FFN output with LSM output
            lsm_out, lsm_state = self.parallel_lsm(x, lsm_state)
            gate = self.fusion_gate(x)
            x_fused = gate * x + (1.0 - gate) * lsm_out

            # 2. Pass fused data through the hybrid FFN block
            x, ff_loss = layer(x_fused, x_neg)  # x_fused is positive data, x_neg is negative
            total_ff_loss += ff_loss

        # Final output processing
        x = self.output_norm(x)
        logits = self.heads[task](x)

        # Return based on mode
        if self.training and y is not None:
            # Polymorphic loss calculation based on task
            if task == 'classification':
                primary_loss = F.cross_entropy(logits, y)
            elif task == 'text':
                # For text generation, y contains target token IDs with -100 for ignored positions.
                primary_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100
                )
            else:
                raise ValueError(f"Loss calculation for task '{task}' is not implemented.")

            total_loss = primary_loss + self.ff_loss_weight * total_ff_loss
            return logits, total_loss
        else:  # Inference
            return logits

    def reset_all_states(self):
        # This method is less critical now as LSM state is reset per batch,
        # but can be kept for compatibility or explicit resets.
        self.parallel_lsm.reset_state()
