# config.py
import yaml
from typing import Dict, Any

def load_config(path: str = 'configs/default_config.yaml') -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_default_config() -> Dict[str, Any]:
    """
    Returns the default STEMA configuration, serving as a single source of truth for all parameters.
    This structure is crucial for modularity and easy experimentation.
    """
    return {
        "model": {
            "max_seq_len": 128,
            "embedding_dim": 256,
            "num_layers": 4,
            "num_attn_heads": 8,  # Köpfe für die Spiking Attention
            "lsm_reservoir_size": 512,  # Größe des LSM Reservoirs
            "lsm_sparsity": 0.9,  # Sparsity der rekurrenten LSM-Verbindungen
            "lif_beta": 0.95,  # Membran-Decay-Rate für LIF-Neuronen
            "lif_sfa_beta": 0.99,  # Decay-Rate für den adaptiven Threshold (SFA)
            "lif_delta_threshold": 0.02,  # Schwellen-Erhöhung bei einem Spike (SFA)
            "lif_initial_threshold": 1.0,  # Initiale Feuerschwelle
            "reasoning_steps": 2,
            "use_spikes": False,
            "use_ternary": False,
            'lsm_e_i_ratio': 0.8,  # 80% exzitatorische, 20% inhibitorische Neuronen
            'recurrent_weight_clamp': 1.5,  # Optional: Klammert rekurrente Gewichte in [-1.5, 1.5]

            # --- BioInspiredLIFNeuron Konfiguration (NEU) ---
            # Heterogenitätsparameter
            'lif_beta_mean': 0.95,  # Mittlerer Zerfallsfaktor des Membranpotentials
            'lif_beta_std': 0.05,  # Standardabweichung für beta

            # SFA (Spike-Frequency Adaptation) Parameter
            'use_sfa': True,
            'sfa_increment': 0.05,  # Wie stark der Schwellenwert pro Spike steigt
            'lif_tau_sfa_mean': 150.0,  # Mittlere Zeitkonstante (in steps) für den SFA-Zerfall
            'lif_tau_sfa_std': 25.0,  # Standardabweichung für tau_sfa

        },
        "training": {
            "device": "cuda",
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 1e-4,
            "clip_grad_norm": 1.0,
            "num_dataloader_workers": 2,
            "lambda_pixel_loss": 1.0,
            "lambda_perceptual_loss": 0.05,
            "steps_per_epoch": 5000,
            "tts_batch_size": 4
        },
        "pretraining": {
            "epochs": 20,
            "batch_size": 4,  # This is the micro-batch size
            "gradient_accumulation_steps": 2, # Effective batch size = batch_size * grad_accum_steps
            "chunks_per_sequence_step": 2, # How many chunks to group into one STEMA sequence
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "use_amp": True,
            "use_torch_compile": True, # For PyTorch 2.x
            "torch_compile_mode": "default", # Or "reduce-overhead"
            "clip_grad_norm": 1.0,
            "log_interval": 50,
            "num_dataloader_workers": 2,
            "min_chunk_duration_sec": 1.0,
            "mask_probability": 0.15,
            "no_text_chunk_ratio": 0.1,
            "loss_weights": {
                "mlm": 1.0,
                "v_mpp": 0.7,
                "a_mpp": 0.7,
                "img_recon": 0.8
            },
            "lambda_pixel_loss": 1.0,
            "lambda_perceptual_loss": 0.1,
        },
        "data": {
            "image_patch_size": 16,
            "audio_n_mels": 80, # Reduced from 128 for better compatibility with smaller n_fft
            "max_text_len": 256,
            "vocab_size": 30522, # From BERT tokenizer
            "tokenizer_model": "bert-base-uncased",
            "chunk_duration_sec": 5,
            "video_fps_sampling": 1,
            "audio_patch_kernel_size": [10, 4], # (80/8, 4)
            "audio_patch_stride": [10, 4],
            "audio_f_min": 0.0,
            "audio_f_max": 8000.0,
        },
        "inference": {
            "default_task": "text",
            "text_head": {}, # vocab_size will be set dynamically
            "text_max_new_tokens": 128,
            "text_temperature": 0.2,
            "text_top_k": 40,
            "text_repetition_penalty": 1.25,
            "text_no_repeat_ngram_size": 3,
        },
        "live_feedback": {
            "enabled": True,
            "log_interval": 10,
            "speak_feedback": False,
        },
        "paths": {
            "checkpoints": "checkpoints",
            "temp_data": "temp_data",
            "log_dir": "logs",
            "generated_video_frames_dir": "generated_video_frames"
        },
        "vocoder": {
            # Parameters for Griffin-Lim and Mel-Spectrogram, must be consistent
            "n_fft": 1024, # Increased for better frequency resolution
            "hop_length": 256,
            "win_length": 1024,
        }
    }

if __name__ == '__main__':
    # This script can be run to generate a fresh default config file
    import os
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    default_config_path = os.path.join(config_dir, 'default_config.yaml')
    current_config_data = get_default_config()
    with open(default_config_path, 'w') as f:
        yaml.dump(current_config_data, f, sort_keys=False, indent=2)
    print(f"Generated/Updated default config at {default_config_path}")
