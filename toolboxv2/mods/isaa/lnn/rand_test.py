
import torch

from test import NeuralSystem

if __name__ == "__main__":

    # Erstellen von Dummy-Daten
    input_size = 10
    hidden_sizes = [20, 30]
    output_size = 5
    liquid_state_hidden_size = 15
    liquid_state_num_layers = 2

    # Dummy-Eingabedaten und Zielwerte
    x_pos = torch.randn(32, input_size)  # 32 Datenpunkte mit 10 Merkmalen
    x_neg = torch.randn(32, input_size)  # 32 negative Datenpunkte
    target = torch.randn(32, output_size)  # Zielwerte für die Datenpunkte

    # Instanziieren des NeuralSystem-Modells
    model = NeuralSystem(input_size, hidden_sizes, output_size, liquid_state_hidden_size, liquid_state_num_layers, 'mae')


    # Definieren des Trainingsschrittes
    def train_test_step(e=100):

        model.train()  # Modell in den Trainingsmodus setzen
        for i in range(e):
            ff_losses, ls_loss = model.train_step(x_pos, x_neg, target)
            if i % 10 == 0:
                print("FeedForward Losses:", sum(ff_losses) / len(ff_losses))
                print("Liquid State Loss:", ls_loss)


    # Anomalie-Erkennung aktivieren, um Fehler zu erkennen
    import torch

    with torch.no_grad():
        model.eval()
        output = model.forward(x_pos)
        M = model.loss_fn(output, target).item()
        print("Model Output: loss", M)

    torch.autograd.set_detect_anomaly(True)

    # Durchführen des Trainingsschrittes
    try:
        train_test_step(500)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

    # Beispielausgabe der letzten Vorhersage
    with torch.no_grad():
        model.eval()
        output = model.forward(x_pos)
        N = model.loss_fn(output, target).item()
        print("Model Output: loss", N)

    print("Advantage = ", M - N)
