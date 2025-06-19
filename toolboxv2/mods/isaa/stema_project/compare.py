# compare.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import yaml
from copy import deepcopy
import gc
from tqdm import tqdm
import argparse

# Lokale STEMA-Importe
from stema.config import get_default_config, load_config
from stema.model import STEMA_Core, STEMAClassifier
from stema.heads import get_all_heads
from stema.data import MultimodalPatchifier
from stema.utils import get_device


# --- 1. Definition der Modellgrößen und des Basismodells ---

def get_model_configs(base_config):
    """ Erstellt Konfigurationen für verschiedene STEMA-Modellgrößen. """
    configs = {}

    # Nano: Das kleinste mögliche Modell
    configs['stema_pico'] = deepcopy(base_config)
    configs['stema_pico']['model'].update({
        "embedding_dim": 2,
        "num_layers": 1,
        "num_attn_heads": 1,
        "lsm_reservoir_size": 6,
    })

    configs['stema_micro'] = deepcopy(base_config)
    configs['stema_micro']['model'].update({
        "embedding_dim": 4,
        "num_layers": 1,
        "num_attn_heads": 2,
        "lsm_reservoir_size": 8,
    })

    # Mimi: Ein mittleres, kleines Modell
    configs['stema_nano'] = deepcopy(base_config)
    configs['stema_nano']['model'].update({
        "embedding_dim": 8,
        "num_layers": 2,
        "num_attn_heads": 2,
        "lsm_reservoir_size": 4,
    })

    # Smol: Das "größte" der kleinen Testmodelle
    configs['stema_mimi'] = deepcopy(base_config)
    configs['stema_mimi']['model'].update({
        "embedding_dim": 16,
        "num_layers": 2,
        "num_attn_heads": 4,
        "lsm_reservoir_size": 32,
    })

    # Konfiguration für das CNN-Basisomdell
    configs['cnn'] = deepcopy(base_config)
    return configs


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


class STEMAClassifierL(nn.Module):
    """
    Wrapper für STEMA, der Patchifizierung und Klassifikation für eine bestimmte Aufgabe kapselt.
    Optimiert für die MNIST-Klassifikation.
    """
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.device = get_device(config)
        self.config = config

        # 1. Patchifier for image processing
        self.patchifier = MultimodalPatchifier(config, self.device)

        # 2. Input projection from patch dim to model dim
        # The patchifier now produces embeddings of size config['model']['embedding_dim'] directly
        # if using LazyConv, or after projection if using ViT.
        # Let's assume patchifier output is already at the target dim for simplicity here.
        # If not, a projection would be needed. MultimodalPatchifier now handles this.
        # The patchifier's `process_image` gives (B, NumPatches, Dim).
        self.input_dim = config['model']['embedding_dim']

        # 3. STEMA-AL core model
        all_heads = get_all_heads(config, self.patchifier)
        self.stema_model = STEMA_Core(config, all_heads)

        # 4. Classification head
        self.embedding_dim = config['model']['embedding_dim']
        self.classifier_head = nn.Linear(self.embedding_dim, num_classes)

        # Move the entire module to the correct device once at the end
        self.to(self.device)

    def forward(self, x):
        # x: MNIST image [B, 1, 28, 28] -> already on the GPU
        x_rgb = x.repeat(1, 3, 1, 1) # Patchifier expects 3 channels

        # Note: Loop is inefficient for batching. A batch-wise patchifier would be faster.
        # For this comparison, we keep it for simplicity.
        patches_list = []
        for i in range(x_rgb.shape[0]):
            # process_image now expects a PIL Image.
            pil_img = transforms.ToPILImage()(x_rgb[i])
            img_patches = self.patchifier.process_image(pil_img) # Returns (1, NumPatches, Dim)
            patches_list.append(img_patches.squeeze(0)) # Squeeze batch dim

        padded_patches = nn.utils.rnn.pad_sequence(patches_list, batch_first=True, padding_value=0.0)

        # The input projection is now effectively handled inside the patchifier
        # No separate self.input_proj needed if configured correctly

        # self.stema_model.reset_all_states()
        hidden_states = self.stema_model(padded_patches, task="embedding")

        # Use mean of sequence embeddings for classification
        mean_embedding = hidden_states.mean(dim=1)
        logits = self.classifier_head(mean_embedding)
        return logits


# --- 2. Trainings- und Validierungslogik ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    clip_value = 1.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if torch.isnan(loss):
            print(f"NaN loss detected in {type(model).__name__}. Skipping update.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
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
    base_config = load_config()
    device = get_device(base_config)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_configs = get_model_configs(base_config)
    models_to_compare = ['stema_pico', 'stema_micro', 'stema_nano', 'stema_mimi', 'cnn']

    results = {}

    for model_name in models_to_compare:
        print(f"\n{'=' * 20} Training: {model_name} {'=' * 20}")

        if 'stema' in model_name:
            config = model_configs[model_name]
            # --- AKTIVIERE TERNARY FÜR STEMA MODELLE ---
            config['model']['use_ternary'] = True # This would be used inside the modules if they check the flag
            model = STEMAClassifier(config, num_classes=10).to(device)
            print(f"Model Params: {model.stema_model.get_parameters()}")
        else:
            config = model_configs[model_name]
            model = SimpleCNN(num_classes=10).to(device)



        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        start_time = time.time()

        for epoch in range(args.epochs):
            if hasattr(model, 'reset_all_states'): model.reset_all_states()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_accuracy = validate(model, test_loader, criterion, device)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        end_time = time.time()
        total_time = end_time - start_time
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[model_name] = {
            'history': history,
            'total_time': total_time,
            'throughput': len(train_dataset) * args.epochs / total_time,
            'num_params': num_params,
            'final_accuracy': history['val_accuracy'][-1]
        }
        print(f"Total time: {total_time:.2f}s | Throughput: {results[model_name]['throughput']:.2f} images/s | Num Params: {num_params}")

        del model, optimizer, criterion
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
        plt.plot(range(1, epochs + 1), data['history']['train_loss'], label=f"{name} Train Loss", color=colors[i], linestyle='--')
        plt.plot(range(1, epochs + 1), data['history']['val_loss'], label=f"{name} Val Loss", color=colors[i])
    plt.title('Trainings- & Validierungsverlust über Epochen')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust (Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results/combined_loss.png')
    plt.close()
    print("Graph 'combined_loss.png' gespeichert.")

    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(results.items()):
        plt.plot(range(1, epochs + 1), data['history']['val_accuracy'], label=f"{name} (Val Acc)", color=colors[i])
    plt.title('Validierungsgenauigkeit über Epochen')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results/validation_accuracy.png')
    plt.close()
    print("Graph 'validation_accuracy.png' gespeichert.")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    accuracies = [data['final_accuracy'] for data in results.values()]
    ax1.bar(model_names, accuracies, color=colors)
    ax1.set_title('Finale Test-Genauigkeit')
    ax1.set_ylabel('Genauigkeit (%)')
    ax1.set_ylim(bottom=max(0, min(accuracies) - 5))

    times = [data['total_time'] for data in results.values()]
    ax2.bar(model_names, times, color=colors)
    ax2.set_title('Gesamte Trainingszeit')
    ax2.set_ylabel('Zeit (Sekunden)')

    throughputs = [data['throughput'] for data in results.values()]
    ax3.bar(model_names, throughputs, color=colors)
    ax3.set_title('Durchsatz (Samples/Sekunde)')
    ax3.set_ylabel('Samples / Sekunde')

    fig.suptitle('Modell-Vergleich Zusammenfassung', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison_results/summary_comparison.png')
    plt.close()
    print("Graph 'summary_comparison.png' gespeichert.")

    print("\n--- Zusammenfassung ---")
    print(f"{'Modell':<15} | {'Parameter':<12} | {'Genauigkeit (%)':<15} | {'Zeit (s)':<10} | {'Durchsatz':<15}")
    print("-" * 75)
    for name, data in results.items():
        print(f"{name:<15} | {data['num_params']:<12,d} | {data['final_accuracy']:<15.2f} | {data['total_time']:<10.2f} | {data['throughput']:<15.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vergleichsskript für STEMA-Modellgrößen.")
    parser.add_argument('--epochs', type=int, default=5, help='Anzahl der Trainingsepochen.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch-Größe für das Training.')
    args = parser.parse_args()
    main(args)
