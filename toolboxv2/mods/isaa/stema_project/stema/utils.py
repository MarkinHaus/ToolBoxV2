import torch
import os
import platform
import subprocess
import re
from typing import Dict, Any, Optional, List


def get_device(config: Dict[str, Any]) -> torch.device:
    """Gets the device (CPU or GPU) from config, ensuring CUDA is available if requested."""
    device_str = config['training']['device']
    if device_str == 'cuda':
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.")
            return torch.device('cuda')
        else:
            print("CUDA not available. Falling back to CPU.")
            return torch.device('cpu')
    else:
        print(f"Using {device_str.upper()}.")
        return torch.device(device_str)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: Dict[str, Any],
                    is_best: bool = False, filename_prefix: str = "stema") -> None:
    """Saves the model and optimizer states to a file."""
    checkpoint_dir = config['paths']['checkpoints']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config  # Save config for reproducibility
    }

    filename = os.path.join(checkpoint_dir, f"{filename_prefix}_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"{filename_prefix}_best.pth")
        torch.save(state, best_filename)
        print(f"New best model saved to {best_filename}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> int:
    """Loads a checkpoint into the model and optimizer.

    Args:
        filepath: Path to the checkpoint file.
        model: The model instance.
        optimizer: The optimizer instance (optional).
        device: The device to map the loaded checkpoint to (optional).

    Returns:
        The epoch number from the checkpoint.
    """
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return 0

    map_location = device if device else lambda storage, loc: storage
    checkpoint = torch.load(filepath, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(
                f"Warning: Could not load optimizer state dict. This might be fine if changing optimizers or LR. Error: {e}")
            print("Optimizer state will be reinitialized.")

    loaded_epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded successfully from {filepath}. Resuming from epoch {loaded_epoch + 1}.")
    # config_in_checkpoint = checkpoint.get('config') # Optionally use/compare
    return loaded_epoch


def show_image(image_path: str) -> None:
    """Opens the generated image using the default system viewer."""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', image_path))
        elif platform.system() == 'Windows':  # Windows
            os.startfile(image_path)
        else:  # linux
            subprocess.call(('xdg-open', image_path))
    except Exception as e:
        print(f"Could not open image automatically. Please find it at: {image_path}\nError: {e}")


def clean_transcript_text(text: str) -> str:
    """Cleans transcript text by removing timestamps, HTML-like tags, and normalizing spaces."""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Ignore index numbers and timestamps
        if re.match(r"^\d+$", line.strip()):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}", line):
            continue
        if not line.strip():
            continue
        cleaned_lines.append(line.strip())

    text = " ".join(cleaned_lines)

    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove special characters like [Music], [Applause], etc.
    text = re.sub(r'\[[^\]]+\]', ' ', text)

    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
