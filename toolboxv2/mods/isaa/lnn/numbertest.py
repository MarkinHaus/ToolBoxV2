if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    from test import (
        NeuralSystem,  # Stellen Sie sicher, dass Sie die Klasse importieren können
    )

    # Hyperparameter
    input_size = 10
    hidden_sizes = [64, 32]
    output_size = 1
    liquid_state_hidden_size = 20
    liquid_state_num_layers = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001


    # Erstellen eines einfachen Datensatzes
    def create_dataset(num_samples=1000):
        X = np.random.randn(num_samples, input_size)
        y = np.sum(X, axis=1, keepdims=True)  # Einfache Summe als Ziel
        return X, y


    # Datensatz erstellen und aufteilen
    X, y = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Daten in PyTorch Tensoren umwandeln
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Datenlader erstellen
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modell initialisieren
    model = NeuralSystem(input_size, hidden_sizes, output_size, liquid_state_hidden_size, liquid_state_num_layers)

    # Optimierer und Verlustfunktion definieren
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Trainingsschleife
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            _, loss = model.train_step(batch_X, None, batch_y)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validierung alle 10 Epochen
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test)
                print(f"Test Loss: {test_loss:.4f}")

    # Abschließender Test
    model.eval()
    with torch.no_grad():
        final_test_output = model(X_test)
        final_test_loss = criterion(final_test_output, y_test)
        print(f"Final Test Loss: {final_test_loss:.4f}")

    # Modell speichern
    model.save("neural_system_model")

    # Modell laden und erneut testen
    loaded_model = NeuralSystem.load("neural_system_model")
    loaded_model.eval()
    with torch.no_grad():
        loaded_test_output = loaded_model(X_test)
        loaded_test_loss = criterion(loaded_test_output, y_test)
        print(f"Loaded Model Test Loss: {loaded_test_loss:.4f}")

    # Beispielvorhersage
    sample_input = torch.randn(1, input_size)
    sample_output = model(sample_input)
    print(f"Sample Input: {sample_input}")
    print(f"Sample Output: {sample_output}")
