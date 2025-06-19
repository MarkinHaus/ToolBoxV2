import os
import sys
import argparse
import yaml
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

import datasets
import transformers
import torchaudio
import torchvision.transforms as T
from torchvision.transforms._presets import VideoClassification
from einops import rearrange

from stema.config import load_config


from stema.model import STEMA_Core
from stema.heads import get_all_heads

# Suppress verbose tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For reproducibility
torch.manual_seed(42)


# --- CONFIGURATION ---

def get_config() -> Dict[str, Any]:
    """
    Centralized configuration for the pre-training run.
    This would typically be a YAML or JSON file.
    """
    config = {
        "run_name": "stema_pretrain_run_01",
        "output_dir": "./stema_checkpoints",

        "dist": {
            "backend": "nccl",
        },

        "data": {
            "vocab_size": 30522,  # BERT-base-uncased
            "tokenizer_name": "bert-base-uncased",
            "image_size": 224,
            "audio_n_mels": 128,
            "audio_sample_rate": 16000,
            "video_num_frames": 16,  # Number of frames to sample from a video
            "max_text_len": 77,
            # Datasets to use (Hugging Face Hub names)
            "image_text_dataset": "conceptual_captions",
            "video_text_dataset": "Maysee/webvid-2m-best-5-frames-per-video",
            # A smaller, pre-processed version of webvid
            # NOTE: Add a public audio-text dataset here, e.g., 'ashraq/esc50' or 'google/AudioSet' (if available)
            # For this example, we'll simulate it being absent.
            "audio_text_dataset": 'ashraq/esc50',
        },

        "model": {
            "embedding_dim": 768,
            "num_layers": 12,
            "num_attn_heads": 12,
            "lsm_reservoir_size": 2048,
            "max_seq_len": 256,  # Max combined sequence length of all modalities
            "use_ternary": False,
            "use_spikes": True,
            "use_sfa": True,
            "lif_beta_mean": 0.95,
            "sfa_increment_mean": 0.05,
            "lif_tau_sfa_mean": 20.0,
        },

        "training": {
            "epochs": 10,
            "batch_size": 32,  # Per-GPU batch size
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "lr_warmup_steps": 2000,
            "grad_accum_steps": 1,  # Increase for larger effective batch size
            "log_interval": 50,  # Log every N steps
            "eval_interval": 1000,  # Evaluate every N steps
            "mask_prob": 0.15,  # Probability of masking a patch/token
        },
        "wandb": {
            "project": "STEMA_Pretraining",
            "entity": None,  # Your wandb entity
        }
    }
    return config


# --- DISTRIBUTED TRAINING UTILS ---

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend=get_config()['dist']['backend'])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# --- DATA PREPARATION AND MASKING ---

class MaskingCollator:
    """
    Custom collator for Masked Multimodal Modeling.
    - Tokenizes text and pads sequences.
    - Processes images and videos.
    - Randomly masks a fraction of the input tokens/patches.
    - The "label" for a masked input is its original, unmasked embedding.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_len = config['data']['max_text_len']
        self.mask_prob = config['training']['mask_prob']

        # This will be initialized with the model's embedding layers
        self.token_embed_layer = None
        self.mask_patch_embedding = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # This collator is complex because it handles multiple optional modalities.
        # We will build the final batch dictionary step-by-step.

        final_batch = {}
        text_inputs = [item.get('text') for item in batch if item.get('text')]
        image_inputs = [item.get('image') for item in batch if item.get('image')]
        video_inputs = [item.get('video') for item in batch if item.get('video')]

        # We must have the model's embeddings to create labels
        assert self.token_embed_layer is not None, "Collator needs model.token_embedding layer."
        assert self.mask_patch_embedding is not None, "Collator needs a mask_patch_embedding parameter."

        device = self.mask_patch_embedding.device

        # --- Process Text ---
        if text_inputs:
            tokenized = self.tokenizer(
                text_inputs, return_tensors='pt', padding='max_length',
                truncation=True, max_length=self.max_len
            )
            input_ids = tokenized.input_ids
            # Create labels before masking
            with torch.no_grad():
                text_labels = self.token_embed_layer(input_ids)

            # Masking logic for text
            input_ids, text_mask_indices = self.mask_tokens(input_ids)
            final_batch['text'] = input_ids
            final_batch['text_labels'] = text_labels
            final_batch['text_mask'] = text_mask_indices

        # --- Process Images ---
        if image_inputs:
            images = torch.stack(image_inputs, dim=0)  # (B, C, H, W)
            # Create labels (original embeddings) before masking
            with torch.no_grad():
                # The model's image encoder expects (B,C,H,W) and returns (B, 1, D)
                # We are creating labels here, so we will use the encoder to get the "true" embedding
                # NOTE: This implies the collator needs access to the encoder.
                # A simpler approach for MPP is to predict the raw patches, but predicting embeddings is more common.
                # Let's stick to predicting the embedding of the masked patch.
                # For now, we will create a placeholder label. The loss function will handle this.
                # The "label" is the image itself, conceptually.
                image_labels = images.clone()  # Placeholder

            # For simplicity, we assume image is a single patch. We mask it by replacing with mask embedding.
            # A more advanced version would use a patchifier.
            # Here, we just mask the whole image embedding.
            final_batch['image'] = images
            final_batch['image_labels'] = image_labels
            # 1D mask for sequence of length 1
            final_batch['image_mask'] = (torch.rand(images.size(0), 1) < self.mask_prob)

        # --- Process Videos ---
        if video_inputs:
            videos = torch.stack(video_inputs, dim=0)  # (B, T, C, H, W)
            video_labels = videos.clone()  # Placeholder

            # Masking logic for video frames (treated as a sequence)
            B, T, C, H, W = videos.shape
            # Create a mask for the time dimension
            video_mask = torch.rand(B, T) < self.mask_prob

            final_batch['video'] = videos
            final_batch['video_labels'] = video_labels
            final_batch['video_mask'] = video_mask

        return final_batch

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepares masked input and labels for text."""
        labels = inputs.clone()

        # Probability matrix for masking
        prob_matrix = torch.full(labels.shape, self.mask_prob)

        # Avoid masking special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        prob_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # We only compute loss on masked tokens
        labels[~masked_indices] = -100  # -100 is the ignore index for CrossEntropyLoss

        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with a random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest (10%) we leave as is
        return inputs, masked_indices


class MultimodalDatasetLoader:
    """Handles loading and preprocessing of all datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])

        # --- Define Transforms ---
        self.image_transform = T.Compose([
            T.Resize((config['data']['image_size'], config['data']['image_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.video_transform = VideoClassification(
            crop_size = (config['data']['image_size'], config['data']['image_size']),
            resize_size = (config['data']['image_size'] + 32, config['data']['image_size'] + 32),  # Commonly slightly larger for random crop

        )

        self.audio_transform = T.Compose([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config['data']['audio_sample_rate'],
                n_mels=config['data']['audio_n_mels']
            ),
            torchaudio.transforms.AmplitudeToDB()
        ])

    def get_unified_dataset(self, split: str = 'train') -> datasets.IterableDataset:
        """
        Loads, preprocesses, and combines multiple datasets into a single stream.

        This method identifies the datasets specified in the configuration,
        applies the appropriate preprocessing pipeline to each, and then uses
        `datasets.interleave_datasets` to create a final, mixed-modality
        iterable dataset. This is highly efficient for large-scale training.

        Args:
            split (str): The dataset split to load (e.g., 'train', 'validation').

        Returns:
            datasets.IterableDataset: An interleaved dataset yielding dictionaries
                                      that can contain 'image', 'video', 'audio',
                                      and 'text' keys.
        """
        datasets_to_interleave = []

        # --- 1. Load Image-Text Dataset ---
        if self.config['data'].get('image_text_dataset'):
            # Use streaming to avoid downloading the entire dataset at once
            ds = datasets.load_dataset(self.config['data']['image_text_dataset'], split=split, streaming=True)
            ds = ds.map(_preprocess_image_text, batched=True, batch_size=16)
            # Standardize columns for merging
            ds = ds.remove_columns([c for c in ds.column_names if c not in ['image', 'text']])
            datasets_to_interleave.append(ds)

        # --- 2. Load Video-Text Dataset ---
        if self.config['data'].get('video_text_dataset'):
            # The specified dataset has no validation split, so use 'test' for validation
            data_split = "train" if split == "train" else "test"
            ds = datasets.load_dataset(self.config['data']['video_text_dataset'], split=data_split, streaming=True)
            ds = ds.map(_preprocess_video_text, batched=True, batch_size=4)  # Smaller batch for video
            ds = ds.remove_columns([c for c in ds.column_names if c not in ['video', 'text']])
            datasets_to_interleave.append(ds)

        # --- 3. Load Audio-Text Dataset (if specified) ---
        if self.config['data'].get('audio_text_dataset'):
            # This part is a placeholder for adding an audio-text dataset
            # ds = datasets.load_dataset(...)
            # ds = ds.map(self._preprocess_audio_text, ...)
            # datasets_to_interleave.append(ds)
            pass

        if not datasets_to_interleave:
            raise ValueError("No datasets were specified in the configuration.")

        # --- 4. Interleave Datasets ---
        # This function creates a single stream by sampling from each dataset.
        # The `probabilities` argument controls the sampling ratio. Here, we
        # sample equally from all available sources.
        probabilities = [1.0 / len(datasets_to_interleave)] * len(datasets_to_interleave)
        unified_dataset = datasets.interleave_datasets(
            datasets_to_interleave,
            probabilities=probabilities,
            seed=42  # for reproducibility
        )

        return unified_dataset


def _preprocess_image_text(self, examples):
    # From Conceptual Captions: 'image_url', 'caption'
    # Note: 'image' is pre-loaded by HF datasets
    images = [self.image_transform(img.convert("RGB")) for img in examples['image']]
    texts = examples['caption']
    return {"image": images, "text": texts}


def _preprocess_video_text(self, examples):
    # From WebVid: 'video', 'caption'
    # The dataset is pre-processed to have 5 frames. We select all of them.
    video_tensors = []
    for video_bytes in examples['content']:
        # This is a hack for the specific Maysee/webvid dataset.
        # A real implementation would use decord or torchvision.io.read_video
        try:
            from io import BytesIO
            import av
            container = av.open(BytesIO(video_bytes), mode="r")
            frames = [frame.to_image() for frame in container.decode(video=0)]
            # Sample or pad to fixed number of frames
            indices = torch.linspace(0, len(frames) - 1, self.config['data']['video_num_frames']).long()
            sampled_frames = torch.stack([T.ToTensor()(frames[i]) for i in indices])  # (T, C, H, W)
            video_tensors.append(sampled_frames)
        except Exception:
            # Append a dummy tensor if video is corrupt
            video_tensors.append(torch.zeros(
                self.config['data']['video_num_frames'], 3,
                self.config['data']['image_size'], self.config['data']['image_size']
            ))

    texts = examples['caption']
    return {"video": video_tensors, "text": texts}


def _preprocess_audio_text(self, examples):
    # Placeholder for an audio-text dataset
    # e.g., for AudioCaps: 'audio', 'caption'
    # audios = [self.audio_transform(audio['array']) for audio in examples['audio']]
    # texts = examples['caption']
    # return {"audio": audios, "text": texts}
    return {}  # Not implemented


def get_unified_dataset(self, split='train'):
    datasets_to_interleave = []

    # Image-Text Dataset
    if self.config['data']['image_text_dataset']:
        ds = datasets.load_dataset(self.config['data']['image_text_dataset'], split=split, streaming=True)
        ds = ds.map(self._preprocess_image_text, batched=True, batch_size=16)
        ds = ds.remove_columns([c for c in ds.column_names if c not in ['image', 'text']])
        datasets_to_interleave.append(ds)

    # Video-Text Dataset
    if self.config['data']['video_text_dataset']:
        # Use a smaller part of train for validation
        data_split = "train" if split == "train" else "test"
        ds = datasets.load_dataset(self.config['data']['video_text_dataset'], split=data_split, streaming=True)
        ds = ds.map(self._preprocess_video_text, batched=True, batch_size=4)
        ds = ds.remove_columns([c for c in ds.column_names if c not in ['video', 'text']])
        datasets_to_interleave.append(ds)

    # Audio-Text Dataset (if specified)
    if self.config['data']['audio_text_dataset']:
        # ... loading logic here ...
        pass

    if not datasets_to_interleave:
        raise ValueError("No datasets specified for pre-training.")

    # Interleave datasets to get a mixed stream of multimodal data
    # Probabilities ensure we sample roughly equally, adjust as needed
    probabilities = [1.0 / len(datasets_to_interleave)] * len(datasets_to_interleave)
    unified_dataset = datasets.interleave_datasets(datasets_to_interleave, probabilities=probabilities)

    return unified_dataset


# --- TRAINER CLASS ---

class PreTrainer:
    def __init__(self, config: Dict[str, Any], resume_from: Optional[str] = None):
        self.config = config
        self.is_main = is_main_process()
        self.device = int(os.environ.get("LOCAL_RANK", 0))

        # --- Model Setup ---
        all_heads = get_all_heads(config)
        self.model = STEMA_Core(config, all_heads)
        self.model.to(self.device)

        if dist.is_initialized():
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

        # --- Optimizer and Scheduler ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scaler = GradScaler()

        # --- Dataloaders and Collator ---
        self.collator = MaskingCollator(config)
        # Share model embeddings with collator
        model_ref = self.model.module if dist.is_initialized() else self.model
        self.collator.token_embed_layer = model_ref.encoders['text'].token_embedding
        # Create a learnable mask patch embedding
        self.mask_patch_embedding = nn.Parameter(torch.randn(config['model']['embedding_dim'])).to(self.device)
        self.collator.mask_patch_embedding = self.mask_patch_embedding

        self.train_loader, self.val_loader = self._setup_dataloaders()

        # Infer total steps for LR scheduler
        # Note: For streaming datasets, len() is not available. We estimate.
        estimated_train_steps_per_epoch = 1_000_000 // (config['training']['batch_size'] * dist.get_world_size())
        total_steps = estimated_train_steps_per_epoch * config['training']['epochs']

        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['lr_warmup_steps'],
            num_training_steps=total_steps
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if resume_from:
            self.load_checkpoint(resume_from)

        if self.is_main:
            print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            if config['wandb']['project']:
                import wandb
                wandb.init(
                    project=config['wandb']['project'],
                    entity=config['wandb']['entity'],
                    name=config['run_name'],
                    config=config
                )
                wandb.watch(self.model, log="all", log_freq=500)

    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        loader = MultimodalDatasetLoader(self.config)

        train_ds = loader.get_unified_dataset(split='train').with_format("torch")
        val_ds = loader.get_unified_dataset(split='validation').with_format(
            "torch")  # Conceptual captions has a validation split

        # For streaming, we can't use a standard sampler. The dataset is already shuffled.
        # DDP requires a sampler, but for IterableDataset, it's a no-op.
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.collator,
            num_workers=4,  # Increase if CPU allows
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.collator,
            num_workers=4
        )
        return train_loader, val_loader

    def _calculate_loss(self, batch: Dict[str, torch.Tensor], model_output: torch.Tensor) -> torch.Tensor:
        """Calculates the Masked Prediction Loss."""
        # The model outputs a sequence of predicted embeddings.
        # We need to align this with our input modalities and compute loss on masked parts.

        total_loss = 0.0
        loss_count = 0

        current_seq_pos = 0
        model_ref = self.model.module if dist.is_initialized() else self.model

        # --- Text Loss ---
        if 'text' in batch:
            seq_len = batch['text'].size(1)
            pred_text_embeds = model_output[:, current_seq_pos: current_seq_pos + seq_len]

            mask = batch['text_mask'].to(self.device)
            target_embeds = batch['text_labels'].to(self.device)

            masked_preds = pred_text_embeds[mask]
            masked_targets = target_embeds[mask]

            if masked_preds.numel() > 0:
                loss = F.mse_loss(masked_preds, masked_targets)
                total_loss += loss
                loss_count += 1
            current_seq_pos += seq_len

        # --- Image Loss (Masked Patch Prediction) ---
        if 'image' in batch:
            seq_len = 1  # Image is treated as a single token
            pred_img_embeds = model_output[:, current_seq_pos: current_seq_pos + seq_len]

            mask = batch['image_mask'].to(self.device).view(-1)
            # Label is the embedding of the original image
            with torch.no_grad():
                target_img_embeds = model_ref.encoders['image'](batch['image'].to(self.device))

            masked_preds = pred_img_embeds[mask]
            masked_targets = target_img_embeds[mask]

            if masked_preds.numel() > 0:
                loss = F.mse_loss(masked_preds, masked_targets)
                total_loss += loss
                loss_count += 1
            current_seq_pos += seq_len

        # --- Video Loss ---
        if 'video' in batch:
            seq_len = self.config['data']['video_num_frames']
            pred_video_embeds = model_output[:, current_seq_pos: current_seq_pos + seq_len]

            mask = batch['video_mask'].to(self.device)
            with torch.no_grad():
                target_video_embeds = model_ref.encoders['video'](batch['video'].to(self.device))

            masked_preds = pred_video_embeds[mask]
            masked_targets = target_video_embeds[mask]

            if masked_preds.numel() > 0:
                loss = F.mse_loss(masked_preds, masked_targets)
                total_loss += loss
                loss_count += 1
            current_seq_pos += seq_len

        return total_loss / (loss_count + 1e-8)

    def _run_one_step(self, batch: Dict[str, Any]):
        # The model's forward pass expects a dictionary of modalities.
        # We need to construct this from the collator's output.
        # A key challenge: the model encoders need the raw data (image tensors, text ids)
        # but the masking happens on the embeddings.
        # We solve this by passing the raw data to the model, and then using the masks
        # during the loss calculation.

        # The collator's masking logic needs to be updated. It shouldn't replace inputs,
        # but just provide a mask. The model itself will handle the replacement.
        # For now, we'll stick to the simpler approach where the loss function handles it.

        inputs_for_model = {}
        if 'text' in batch: inputs_for_model['text'] = batch['text'].to(self.device)
        if 'image' in batch: inputs_for_model['image'] = batch['image'].to(self.device)
        if 'video' in batch: inputs_for_model['video'] = batch['video'].to(self.device)

        with autocast():
            # Run forward pass through the core model to get embeddings
            predicted_embeddings = self.model(inputs_for_model, task="embedding_prediction")
            loss = self._calculate_loss(batch, predicted_embeddings)

        return loss

    def train(self):
        if self.is_main:
            print("--- Starting Pre-training ---")

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.model.train()
            # For IterableDataset, we cannot set epoch on sampler

            for i, batch in enumerate(self.train_loader):
                if self.global_step >= self.config['training']['lr_warmup_steps'] + self.scheduler.num_training_steps:
                    if self.is_main: print("Training finished.")
                    return

                loss = self._run_one_step(batch)

                # Gradient accumulation
                loss = loss / self.config['training']['grad_accum_steps']
                self.scaler.scale(loss).backward()

                if (i + 1) % self.config['training']['grad_accum_steps'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.global_step += 1

                # --- Logging ---
                if self.global_step % self.config['training']['log_interval'] == 0 and self.is_main:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Epoch: {epoch + 1} | Step: {self.global_step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
                    if self.config['wandb']['project']:
                        import wandb
                        wandb.log({"train/loss": loss.item(), "learning_rate": lr}, step=self.global_step)

                # --- Evaluation and Checkpointing ---
                if self.global_step % self.config['training']['eval_interval'] == 0:
                    val_loss = self.validate()
                    if self.is_main:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            print(f"** New best validation loss: {val_loss:.4f}. Saving model. **")
                            self.save_checkpoint("best_model.pt")

                        self.save_checkpoint("latest_model.pt")

                    self.model.train()  # Switch back to train mode

            if self.is_main:
                print(f"--- End of Epoch {epoch + 1} ---")

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        count = 0
        if self.is_main: print("--- Running Validation ---")

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Run for a fixed number of validation steps for speed
                if i > 100: break

                loss = self._run_one_step(batch)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count

        if self.is_main and self.config['wandb']['project']:
            import wandb
            wandb.log({"val/loss": avg_loss}, step=self.global_step)

        if dist.is_initialized():
            # Collect loss from all processes
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        if self.is_main:
            print(f"--- Validation Complete | Avg Loss: {avg_loss:.4f} ---")

        return avg_loss

    def save_checkpoint(self, filename: str):
        if not self.is_main: return

        os.makedirs(self.config['output_dir'], exist_ok=True)
        path = os.path.join(self.config['output_dir'], filename)

        model_state = self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict()

        checkpoint = {
            "epoch": self.start_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=f"cuda:{self.device}")

        model_ref = self.model.module if dist.is_initialized() else self.model
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed training from checkpoint: {path} at epoch {self.start_epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="STEMA Multimodal Pre-trainer")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()

    config = {**load_config(), **get_config()}

    # Check for DDP environment variables
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        setup_ddp()
        print(f"DDP Initialized on Rank {os.environ['RANK']}")

    trainer = PreTrainer(config, resume_from=args.resume)
    trainer.train()

    if dist.is_initialized():
        cleanup_ddp()


if __name__ == "__main__":
    main()
