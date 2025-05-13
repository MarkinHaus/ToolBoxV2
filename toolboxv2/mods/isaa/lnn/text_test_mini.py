import numpy as np
import torch

from test import NeuralSystem


# Schritt 4: Vorhersage
def predict_next_token(model, input_token):
    input_idx = token_to_idx[input_token]
    input_tensor = torch.zeros(input_size)
    input_tensor[input_idx] = 1

    output = model.forward(input_tensor)
    predicted_idx = torch.argmax(output).item()
    return idx_to_token[predicted_idx]


if __name__ == '__main__':
    # Schritt 1: Datenaufbereitung
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    tokens = list(text)
    vocab = list(set(tokens))
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    input_size = len(vocab)
    hidden_sizes = [126, 32, 126]
    output_size = len(vocab)
    liquid_state_size = 64
    liquid_state_dim = 8

    # Schritt 2: Modellinitialisierung
    model = NeuralSystem(input_size, hidden_sizes, output_size, liquid_state_size, liquid_state_dim, 'mae')

    # Schritt 3: Trainingsschleife
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(tokens) - 1):
            input_token = tokens[i]
            target_token = tokens[i + 1]

            input_idx = token_to_idx[input_token]
            target_idx = token_to_idx[target_token]

            input_tensor = torch.zeros(input_size)
            input_tensor[input_idx] = 1

            target_tensor = torch.zeros(output_size)
            target_tensor[target_idx] = 1

            # Zufälligen negativen Input generieren
            neg_input_tensor = torch.zeros(input_size)
            neg_input_tensor[np.random.choice(len(vocab))] = 1

            _, loss = model.train_step(input_tensor, neg_input_tensor, target_tensor)
            total_loss += loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")


    # Test der Vorhersage
    for char in text[:-2]:
        next_char = predict_next_token(model, char)
        print(f"Current: {char}, Predicted next: {next_char}")

    text = text[0]
    for i in range(10):
        text += predict_next_token(model, text[-1])
    print(f"New text: {text}")
