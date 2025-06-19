# compare.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
from copy import deepcopy
import gc
from tqdm import tqdm
import argparse

# --- New STEMA Imports ---
from stema.config import load_config
from stema.model import STEMA_Core
from stema.trainer import STEMATrainer
from stema.utils import get_device


class SimpleCNN(nn.Module):
    """ Ein einfaches, aber effektives CNN für MNIST als Referenzmodell. """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits


# --- 1. Definition der Modellgrößen und des Basismodells ---

def get_model_configs(base_config):
    """ Erstellt Konfigurationen für verschiedene STEMA-Modellgrößen. """
    configs = {}

    # Pico: The smallest possible model for debugging/testing
    configs['stema_pico'] = deepcopy(base_config)
    configs['stema_pico']['model'].update(
        {"embedding_dim": 32, "num_layers": 1, "num_attn_heads": 1, "lsm_reservoir_size": 64})

    # Micro: A very small model
    configs['stema_micro'] = deepcopy(base_config)
    configs['stema_micro']['model'].update(
        {"embedding_dim": 64, "num_layers": 2, "num_attn_heads": 2, "lsm_reservoir_size": 128})

    # Nano: A small, more capable model
    configs['stema_nano'] = deepcopy(base_config)
    configs['stema_nano']['model'].update(
        {"embedding_dim": 128, "num_layers": 3, "num_attn_heads": 4, "lsm_reservoir_size": 256})

    # Smol: The "largest" of the small test models
    configs['stema_mimi'] = deepcopy(base_config)
    configs['stema_mimi']['model'].update(
        {"embedding_dim": 256, "num_layers": 4, "num_attn_heads": 8, "lsm_reservoir_size": 512})

    # Config for the CNN baseline model
    configs['cnn'] = deepcopy(base_config)
    return configs


# --- NEW: Custom Dataset to format MNIST for the STEMA model ---
class MNISTDictDataset(Dataset):
    """
    Wraps the standard MNIST dataset to return a dictionary of modalities,
    as expected by the STEMATrainer.
    """

    def __init__(self, mnist_dataset, image_in_channels=3):
        self.mnist_dataset = mnist_dataset
        self.image_in_channels = image_in_channels

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Original data is a tuple (image_tensor, label)
        image, label = self.mnist_dataset[idx]

        # STEMA's ImageEncoder expects C channels, MNIST is grayscale (1 channel)
        if self.image_in_channels > 1 and image.shape[0] == 1:
            image = image.repeat(self.image_in_channels, 1, 1)

        # Return in the format: (inputs_dict, target)
        return {'image': image}, label


# --- 2. Trainings- und Validierungslogik (NUR FÜR DAS CNN-BASELINE) ---
def train_one_epoch_cnn(model, dataloader, optimizer, criterion, device):
    """ Standalone training function ONLY for the SimpleCNN baseline. """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training CNN", leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def validate_cnn(model, dataloader, criterion, device):
    """ Standalone validation function ONLY for the SimpleCNN baseline. """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


# --- 3. Haupt-Vergleichs-Skript ---
def main(args):
    base_config = load_config(args.config_path)
    base_config['training']['device'] = args.device if args.device else base_config['training']['device']
    base_config['training']['epochs'] = args.epochs
    base_config['training']['batch_size'] = args.batch_size
    device = get_device(base_config)

    # Standard MNIST transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load raw MNIST datasets
    mnist_train_raw = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test_raw = MNIST(root='./data', train=False, download=True, transform=transform)

    # --- Setup DataLoaders for STEMA and CNN ---
    # STEMA needs the dict format
    stema_train_dataset = MNISTDictDataset(mnist_train_raw, image_in_channels=base_config['data']['image_in_channels'])
    stema_test_dataset = MNISTDictDataset(mnist_test_raw, image_in_channels=base_config['data']['image_in_channels'])
    stema_train_loader = DataLoader(stema_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True)
    stema_test_loader = DataLoader(stema_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True)

    # CNN uses the standard tuple format
    cnn_train_loader = DataLoader(mnist_train_raw, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    cnn_test_loader = DataLoader(mnist_test_raw, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)

    model_configs = get_model_configs(base_config)
    models_to_compare = ['stema_pico', 'stema_micro', 'stema_nano', 'stema_mimi', 'cnn']

    results = {}

    for model_name in models_to_compare:
        print(
            f"\n{'=' * 20} Training: {model_name} {'=' * 20}")

        config = model_configs[model_name]
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        start_time = time.time()

        if 'stema' in model_name:
            config = model_configs[model_name]
            # Directly create the core model
            model = STEMA_Core(config).to(device)

            trainer = STEMATrainer(
                model=model,
                config=config,
                train_loader=stema_train_loader,
                val_loader=stema_test_loader
            )
            trainer.train()

            # Extract history from the trainer
            history['train_loss'] = trainer.history.get('train_loss', [])
            history['val_loss'] = trainer.history.get('val_loss', [])
            # Note: val_accuracy is not computed by default in the new trainer to keep it lean.
            # We can use the final validation loss as a proxy or add accuracy calculation to the trainer.
            # For this comparison, we'll use the final validation loss.
            final_accuracy = -1  # Placeholder, as trainer focuses on loss.

        else:
            # --- CNN Training using old standalone functions ---
            model = SimpleCNN(num_classes=10).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            for epoch in range(args.epochs):
                train_loss = train_one_epoch_cnn(model, cnn_train_loader, optimizer, criterion, device)
                val_loss, val_accuracy = validate_cnn(model, cnn_test_loader, criterion, device)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            final_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else -1

        end_time = time.time()
        total_time = end_time - start_time
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[model_name] = {
            'history': history,
            'total_time': total_time,
            'throughput': len(mnist_train_raw) * args.epochs / total_time if total_time > 0 else 0,
            'num_params': num_params,
            'final_accuracy': final_accuracy,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf')
        }
        print(
            f"Total time: {total_time:.2f}s | Throughput: {results[model_name]['throughput']:.2f} images/s | Num Params: {num_params}")

        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_results(results, args.epochs)


def plot_results(results, epochs):
    """ Erstellt und speichert die Ergebnis-Graphen. """
    os.makedirs('comparison_results', exist_ok=True)
    model_names = list(results.keys())
    colors = plt.cm.viridis(torch.linspace(0, 1, len(model_names)))

    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(results.items()):
        if data['history']['train_loss']:
            plt.plot(range(1, len(data['history']['train_loss']) + 1), data['history']['train_loss'],
                     label=f"{name} Train Loss", color=colors[i], linestyle='--')
        if data['history']['val_loss']:
            plt.plot(range(1, len(data['history']['val_loss']) + 1), data['history']['val_loss'],
                     label=f"{name} Val Loss", color=colors[i])
    plt.title('Trainings- & Validierungsverlust über Epochen')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results/combined_loss.png')
    plt.close()
    print("Graph 'combined_loss.png' gespeichert.")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    val_losses = [data['final_val_loss'] for data in results.values()]
    ax1.bar(model_names, val_losses, color=colors)
    ax1.set_title('Finaler Validierungsverlust')
    ax1.set_ylabel('Verlust')
    fig.tight_layout()
    plt.savefig('comparison_results/final_val_loss_comparison.png')
    plt.close()
    print("Graph 'final_val_loss_comparison.png' gespeichert.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    times = [data['total_time'] for data in results.values()]
    ax1.bar(model_names, times, color=colors)
    ax1.set_title('Gesamte Trainingszeit')
    ax1.set_ylabel('Zeit (Sekunden)')
    ax1.tick_params(axis='x', rotation=45)

    throughputs = [data['throughput'] for data in results.values()]
    ax2.bar(model_names, throughputs, color=colors)
    ax2.set_title('Durchsatz (Samples/Sekunde)')
    ax2.set_ylabel('Samples / Sekunde')
    ax2.tick_params(axis='x', rotation=45)

    fig.suptitle('Modell-Vergleich: Effizienz', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison_results/efficiency_comparison.png')
    plt.close()
    print("Graph 'efficiency_comparison.png' gespeichert.")

    print("\n--- Zusammenfassung ---")
    print(f"{'Modell':<15} | {'Parameter':<12} | {'Val Verlust':<15} | {'Zeit (s)':<10} | {'Durchsatz':<15}")
    print("-" * 75)
    for name, data in results.items():
        print(
            f"{name:<15} | {data['num_params']:<12,d} | {data['final_val_loss']:<15.4f} | {data['total_time']:<10.2f} | {data['throughput']:<15.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vergleichsskript für STEMA-Modellgrößen.")
    parser.add_argument('--epochs', type=int, default=5, help='Anzahl der Trainingsepochen.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch-Größe für das Training.')
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml',
                        help="Pfad zur Konfigurationsdatei.")
    parser.add_argument('--device', type=str, default=None, help="Gerät überschreiben (z.B. 'cpu' oder 'cuda').")
    args = parser.parse_args()
    main(args)
