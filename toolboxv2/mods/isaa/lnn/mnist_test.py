import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from test import NeuralSystem


def setup_mnist_data():
    # Daten transformieren
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST-Datensätze laden
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader erstellen
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train_mnist(ff_lsm, train_loader, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1)  # Flatten the images
            target_one_hot = torch.zeros(target.size(0), 10)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)

            # Training
            _, loss = ff_lsm.train(data, None, target_one_hot)
            total_loss += loss

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss:.4f}')

        print(f'Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}')


def _test_mnist(ff_lsm: NeuralSystem, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)  # Flatten the images
            outputs = ff_lsm.forward(data, frozen=True)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy


def visualize_results(ff_lsm, test_loader):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Vorhersagen machen
    with torch.no_grad():
        outputs = ff_lsm.forward(images.view(images.size(0), -1), frozen=True)
    _, predicted = torch.max(outputs, 1)

    # Einige der Bilder anzeigen
    fig = plt.figure(figsize=(12, 48))
    for i in range(10):
        ax = fig.add_subplot(10, 1, i + 1)
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {predicted[i]}, Actual: {labels[i]}')
    plt.tight_layout()
    plt.show()


# Hauptausführung
if __name__ == "__main__":
    # MNIST-Daten laden
    train_loader, test_loader = setup_mnist_data()

    # FF_LSM-Modell initialisieren
    input_size = 28 * 28  # MNIST-Bilder sind 28x28
    hidden_sizes = [256, 128]
    output_size = 10  # 10 Ziffern
    liquid_state_size = 64
    liquid_state_dim = 3

    ff_lsm = NeuralSystem(input_size, hidden_sizes, output_size, liquid_state_size, liquid_state_dim)

    # Training
    train_mnist(ff_lsm, train_loader, epochs=5)

    # Testen
    accuracy = _test_mnist(ff_lsm, test_loader)

    # Ergebnisse visualisieren
    visualize_results(ff_lsm, test_loader)

    # Modell speichern
    ff_lsm.save("models/mnist_model")
