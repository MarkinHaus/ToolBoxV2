import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans


def create_heatmap(data, update_interval=1000, is_complex=False, get_data=None, title="heat map"):
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 0:
        data = data.reshape(1, 1)

    if data.ndim > 2:
        data = np.mean(data, axis=tuple(range(2, data.ndim)))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    rows, cols = data.shape
    patches = []
    colors_real = []
    colors_imag = []

    for i in range(rows):
        for j in range(cols):
            square = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
            triangle_real = Polygon([square[0], square[1], square[3]])
            triangle_imag = Polygon([square[1], square[2], square[3]])
            patches.extend([triangle_real, triangle_imag])

            if is_complex:
                colors_real.append(data[i, j].real)
                colors_imag.append(data[i, j].imag)
            else:
                colors_real.append(data[i, j])
                colors_imag.append(data[i, j])

    colors = colors_real + colors_imag
    collection = PatchCollection(patches, cmap='viridis', edgecolors='face')
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    plt.colorbar(collection)

    if is_complex:
        plt.title(title)
    else:
        plt.title(title)

    def update(frame):
        if is_complex:
            new_data = get_data() if get_data is not None else data
            new_colors = list(new_data.real.flatten()) + list(new_data.imag.flatten())
        else:
            new_data = get_data() if get_data is not None else data
            new_colors = list(new_data.flatten())  #* 2

        collection.set_array(np.array(new_colors))
        return [collection]

    ani = FuncAnimation(fig, update, interval=update_interval, blit=True)
    return fig, ani


def transform_state_for_heatmap(state):
    # Überprüfen, ob der Zustand bereits komplex ist
    if np.iscomplexobj(state):
        return state

    # Wenn der Zustand noch nicht komplex ist, konvertieren wir ihn
    if state.shape[-1] == 2:
        return state[..., 0] + 1j * state[..., 1]
    else:
        raise ValueError("Der Zustand muss entweder komplex sein oder die letzte Dimension muss 2 sein.")


def transform_sequential_for_heatmap(model):
    weights = []
    biases = []

    for layer in model:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.cpu().numpy())
            if layer.bias is not None:
                biases.append(layer.bias.data.cpu().numpy())
        elif isinstance(layer, nn.Conv2d):
            # Für Conv2D-Layer flatten wir die Gewichte
            w = layer.weight.data.cpu().numpy()
            weights.append(w.reshape(w.shape[0], -1))
            if layer.bias is not None:
                biases.append(layer.bias.data.cpu().numpy())

    # Kombiniere alle Gewichte und Biases
    all_weights = np.concatenate([w.flatten() for w in weights]) + 0.75
    all_biases = np.concatenate([b.flatten() for b in biases]) - 0.75

    # Kombiniere Gewichte und Biases in ein einziges Array
    combined = np.concatenate([all_weights, all_biases])

    # Reshape zu einem 2D-Array für die Heatmap
    # Wir verwenden eine Quadratwurzel, um es möglichst quadratisch zu machen
    size = int(np.ceil(np.sqrt(combined.shape[0])))
    heatmap_data = np.zeros((size, size))
    heatmap_data.flat[:combined.shape[0]] = combined

    return heatmap_data


def transform_ffn_for_heatmap(layers):
    weights = []
    biases = []

    for layer in layers:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.cpu().numpy())
            if layer.bias is not None:
                biases.append(layer.bias.data.cpu().numpy())

    # Kombiniere alle Gewichte und Biases
    all_weights = np.concatenate([w.flatten() for w in weights]) + 0.75
    all_biases = np.concatenate([b.flatten() for b in biases]) - 0.75

    # Kombiniere Gewichte und Biases in ein einziges Array
    combined = np.concatenate([all_weights, all_biases])

    # Reshape zu einem 2D-Array für die Heatmap
    # Wir verwenden eine Quadratwurzel, um es möglichst quadratisch zu machen
    size = int(np.ceil(np.sqrt(combined.shape[0])))
    heatmap_data = np.zeros((size, size))
    heatmap_data.flat[:combined.shape[0]] = combined

    return heatmap_data


class LiquidStateVisualizer:
    def __init__(self, network):
        self.network = network
        self.ff_network = network.ff_network
        self.liquid_state = network.liquid_state

    def show_full_state(self):
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)

        # FF Network
        ax_ff = fig.add_subplot(gs[0, :])
        self.plot_ff_network(ax_ff)

        # Liquid State
        ax_ls = fig.add_subplot(gs[1, :])
        self.plot_liquid_state(ax_ls)

        # LSM Input Model
        ax_lsm_input = fig.add_subplot(gs[2, 0])
        self.plot_lsm_input_weights(ax_lsm_input)

        # LSM State
        ax_lsm_state = fig.add_subplot(gs[2, 1])
        self.plot_lsm_state(ax_lsm_state)

        # LSM Output Model
        ax_lsm_output = fig.add_subplot(gs[2, 2])
        self.plot_lsm_output_weights(ax_lsm_output)

        plt.tight_layout()
        plt.show()

    def plot_ff_network(self, ax):
        layer_sizes = [self.ff_network.layers[0].in_features] + \
                      [layer[0].out_features for layer in self.ff_network.layers[1:-1]] + \
                      [self.ff_network.layers[-1].out_features]

        ax.set_title("Forward-Forward Network")
        for i, size in enumerate(layer_sizes):
            ax.scatter([i] * size, range(size), s=100)
            if i < len(layer_sizes) - 1:
                for j in range(size):
                    for k in range(layer_sizes[i + 1]):
                        ax.plot([i, i + 1], [j, k], 'k-', alpha=0.1)

        ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax.set_ylim(-1, max(layer_sizes))
        ax.axis('off')

    def plot_liquid_state(self, ax):
        ax.set_title("Liquid State Machine")

        # Input
        input_size = self.liquid_state.size
        ax.scatter([0] * input_size, range(input_size), c='r', s=100, label='Input')

        # Liquid State
        liquid_positions = np.array(
            [(i + 1, j) for i in range(self.liquid_state.dimension) for j in range(self.liquid_state.size)])
        ax.scatter(liquid_positions[:, 0], liquid_positions[:, 1], c='g', s=100, label='Liquid State')

        # Output
        output_size = self.network.output_size
        ax.scatter([self.liquid_state.dimension + 1] * output_size, range(output_size), c='b', s=100, label='Output')

        ax.set_xlim(-0.5, self.liquid_state.dimension + 1.5)
        ax.set_ylim(-0.5, max(input_size, self.liquid_state.size, output_size) - 0.5)
        ax.legend()
        ax.axis('off')

    def plot_lsm_input_weights(self, ax):
        weights = self.liquid_state.i_model[0].weight.detach().numpy()
        ax.set_title("LSM Input Weights")
        im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
        plt.colorbar(im, ax=ax, label='Weight')
        ax.set_xlabel("Input")
        ax.set_ylabel("Liquid State")

    def plot_lsm_state(self, ax):
        state = np.abs(self.liquid_state.state)
        if state.ndim == 1:
            state = np.pad(state, (0, state.size), 'constant').reshape(2, -1)
        while state.ndim > 2 and state.ndim != 3:
            state = state.mean(axis=-1)

        ax.set_title("LSM State")
        im = ax.imshow(state, cmap='hot', aspect='auto', norm=Normalize(vmin=0, vmax=1))
        plt.colorbar(im, ax=ax, label='Activation')
        ax.set_xlabel("Position")
        ax.set_ylabel("Layer")

    def plot_lsm_output_weights(self, ax):
        weights = self.liquid_state.o_model[0].weight.detach().numpy()
        ax.set_title("LSM Output Weights")
        im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
        plt.colorbar(im, ax=ax, label='Weight')
        ax.set_xlabel("Liquid State")
        ax.set_ylabel("Output")

    def show_live_state(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(np.abs(self.liquid_state.state).clip(0, 255), cmap='hot', aspect='auto',
                       norm=Normalize(vmin=0, vmax=1))
        plt.colorbar(im, label='Activation')
        ax.set_title("Live Liquid State")
        ax.set_xlabel("Position")
        ax.set_ylabel("Layer")

        def update(frame):
            im.set_array(np.abs(self.liquid_state.state).clip(0, 255))
            return im,

        from matplotlib.animation import FuncAnimation
        FuncAnimation(fig, update, frames=200, interval=50, blit=True)
        plt.show(block=False)


class SuperLoss(nn.Module):
    def __init__(self, name="mse"):
        super().__init__()
        self.loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCELoss(),
            'bce2': nn.BCEWithLogitsLoss(),
            'huber': nn.HuberLoss()
        }
        self.current_loss = name

    def forward(self, predictions, targets):

        if predictions.shape != targets.shape:
            # print(f"Forward Loss: shapes mismatch: {predictions.shape} {targets.shape}")
            targets = targets.view(predictions.shape)
        return self.loss_functions[self.current_loss](predictions, targets)

    def set_loss(self, loss_name):
        if loss_name in self.loss_functions:
            self.current_loss = loss_name
        else:
            raise ValueError(f"Unbekannte Loss-Funktion: {loss_name}")

    def add_custom_loss(self, name, loss_function):
        self.loss_functions[name] = loss_function

    @staticmethod
    def select_loss(data, task):
        # Hier implementieren wir eine einfache Heuristik zur Auswahl der Loss-Funktion
        if task == 'regression':
            return 'mse'
        elif task == 'classification':
            if len(data.shape) == 2 and data.shape[1] > 1:
                return 'cross_entropy'
            else:
                return 'bce'
        elif task == 'ranking':
            return 'mse'  # Hier könnten Sie eine spezielle Ranking-Loss hinzufügen
        else:
            return 'mse'  # Standardmäßig MSE verwenden

    def super_function(self, data, task):
        optimal_loss = self.select_loss(data, task)
        self.set_loss(optimal_loss)
        return optimal_loss

    def batch_compute_loss(self, batch_predictions, batch_targets):
        """
        Berechnet den Loss für einen Batch von Vorhersagen und Zielen.

        :param batch_predictions: Ein Tensor der Form (batch_size, ...) mit den Vorhersagen
        :param batch_targets: Ein Tensor der Form (batch_size, ...) mit den Zielen
        :return: Ein Tensor der Form (batch_size,) mit den Loss-Werten für jeden Datenpunkt im Batch
        """
        batch_loss = self.forward(batch_predictions, batch_targets)

        # Für einige Loss-Funktionen müssen wir möglicherweise die Dimensionen reduzieren
        if self.current_loss in ['cross_entropy', 'bce']:
            return batch_loss
        else:
            return torch.mean(batch_loss, dim=tuple(range(1, len(batch_loss.shape))))


class NeuralSystem(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, liquid_state_hidden_size, liquid_state_num_layers, loss_fn='mse'):
        super().__init__()

        self.ff_network = ForwardForwardNetwork(input_size, hidden_sizes, input_size)
        self.liquid_state = LiquidState(input_size=input_size, output_size=output_size,
                                        hidden_size=liquid_state_hidden_size,
                                        num_liquid_layers=liquid_state_num_layers)
        self.current_state = None
        self.last_state = None
        self.last_ff_losses = [0]
        self.last_output = None

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.loss_fn = SuperLoss(loss_fn)

        self.apply(self.init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.1)

    def forward(self, x_pos, x_neg=None, frozen=False):
        if not isinstance(x_pos, torch.Tensor):
            x_pos = torch.as_tensor(x_pos, dtype=torch.float32)
        if x_neg is not None and not isinstance(x_neg, torch.Tensor):
            x_neg = torch.as_tensor(x_neg, dtype=torch.float32)
        if x_neg is not None:
            # Training-Modus
            self.last_ff_losses = self.ff_network.forward_forward(x_pos, x_neg)
            ff_output = self.ff_network.predict(x_pos)
        else:
            # Inferenz-Modus
            ff_output = self.ff_network.predict(x_pos)
        output, current_state = self.liquid_state(ff_output, self.current_state if not frozen else None)
        if not frozen:
            self.current_state = current_state
        self.last_state = current_state
        return output

    def train_step(self, x_pos, x_neg, target, auto_find_x_neg=True, num_iterations_x_neg=2):
        if not isinstance(x_pos, torch.Tensor):
            x_pos = torch.as_tensor(x_pos, dtype=torch.float32)
        if x_neg is not None and not isinstance(x_neg, torch.Tensor):
            x_neg = torch.as_tensor(x_neg, dtype=torch.float32)
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=torch.float32)
        if x_neg is None and auto_find_x_neg:
            x_neg = self.find_x_neg(x_pos, target, num_iterations=num_iterations_x_neg)

        self.optimizer.zero_grad()
        self.last_output = self.forward(x_pos, x_neg, frozen=True)
        ls_loss = self.loss_fn(self.last_output, target)
        ls_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradienten-Clipping anwenden
        self.optimizer.step()
        self.scheduler.step(ls_loss)  # Lernrate anpassen
        return self.last_ff_losses, ls_loss.item()

    def find_x_neg_(self, x_pos, target, num_iterations=10):
        if not isinstance(x_pos, torch.Tensor):
            x_pos = torch.as_tensor(x_pos, dtype=torch.float32)
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=torch.float32)

        losses = [0]
        values = [x_pos * -1]
        x_neg_work = x_pos.detach().numpy().copy()
        x_neg_sto = x_pos.detach().numpy().copy()
        x_neg_sto_ = x_pos.detach().numpy().copy()

        # Vordefinierte Anpassungswerte
        adjustments = np.array([0.01, -0.01, 0.05, -0.05])

        for iteration in range(num_iterations):
            # Erstelle eine Matrix von Anpassungen für alle Indizes und Werte
            adj_matrix = adjustments[:, np.newaxis] * iteration

            # Erweitere x_neg_work zu einer 3D-Matrix
            x_neg_expanded = x_neg_work[np.newaxis, np.newaxis, :]

            # Wende alle Anpassungen gleichzeitig an
            x_neg_adjusted = x_neg_expanded + adj_matrix[:, :, np.newaxis]

            # Reshape für einfachere Verarbeitung
            x_neg_adjusted = x_neg_adjusted.reshape(-1, x_neg_work.shape[0])

            # Berechne den Output für alle angepassten x_neg gleichzeitig
            ff_output = self.ff_network.predict(torch.as_tensor(x_neg_adjusted, dtype=torch.float32))
            final_output, _ = self.liquid_state(ff_output)

            # Berechne die Verluste für alle angepassten x_neg
            loss_values = self.loss_fn(final_output.squeeze(0), target.float()).detach().numpy()

            # Finde den besten Verlust und das entsprechende x_neg
            best_loss = loss_values
            best_x_neg = x_neg_adjusted

            if best_loss > max(losses) and not np.array_equal(best_x_neg, x_neg_sto_):
                losses.append(best_loss)
                values.append(best_x_neg.copy())
                x_neg_sto = best_x_neg.copy()

            x_neg_work = x_neg_sto.copy()

        x_neg_final = values[np.argmax(losses)]
        return x_neg_final

    def find_x_neg(self, x_pos, target, num_iterations=10):
        if not isinstance(x_pos, torch.Tensor):
            x_pos = torch.as_tensor(x_pos, dtype=torch.float32)
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=torch.float32)

        losses = [0, ]
        values = [x_pos * -1, ]
        x_neg_work = x_pos.detach().numpy().copy()
        x_neg_sto = x_pos.detach().numpy().copy()
        x_neg_sto_ = x_pos.detach().numpy().copy()
        for _ in range(num_iterations):
            for index in range(len(x_pos)):
                for aj in [.01, -.01, .05, -.05]:
                    if len(x_neg_work) > 1:
                        x_neg_work[index] += aj * _
                    else:
                        x_neg_work += aj * _
                    final_output = self.forward(torch.as_tensor(x_neg_work, dtype=torch.float32),
                                                None, frozen=True)
                    loss_ = self.loss_fn(final_output.squeeze(0), target.float()).item()
                    # print(loss_ > max(losses) , x_neg_work.sum() - x_neg_sto_.sum())
                    if loss_ > max(losses) and x_neg_work.sum() - x_neg_sto_.sum() != 0:
                        losses.append(loss_)
                        values.append(x_neg_work.copy())
                        x_neg_sto = x_neg_work.copy()
            x_neg_work = x_neg_sto.copy()
        x_neg_final = values[losses.index(max(losses))]
        return x_neg_final

    def train_clustering(self, data, n_clusters=2, epochs=100):
        kmeans = KMeans(n_clusters=n_clusters)

        for _ in range(epochs):
            # Führen Sie das Clustering durch
            cluster_labels = kmeans.fit_predict(data)

            # Trainieren Sie das Netzwerk mit den Cluster-Labels
            for input_data, label in zip(data, cluster_labels, strict=False):
                if type(label) in (int, float):
                    label = [label]
                self.train(input_data, None, label)

    def predict_cluster(self, input_data):
        output = self.forward(input_data)
        return torch.argmax(output).item()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath + '.pt')

        config = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'liquid_state_hidden_size': self.liquid_state.hidden_size,
            'liquid_state_num_layers': self.liquid_state.num_liquid_layers,
            'loss_fn': self.loss_fn.current_loss
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath + '_config.json') as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(filepath + '.pt'))
        return model

    def live_ff(self):
        return transform_ffn_for_heatmap(self.ff_network.layers)

    def live_li(self):
        return transform_sequential_for_heatmap(self.liquid_state.i_model)

    def live_lo(self):
        return transform_sequential_for_heatmap(self.liquid_state.o_model)

    def live_ls(self):
        return transform_state_for_heatmap(self.liquid_state.state)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class LiquidState(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_liquid_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_liquid_layers = num_liquid_layers

        # Eingangsschicht
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size))

        # Flüssige Schichten
        self.liquid_layers = nn.ModuleList([
            LiquidLayer(hidden_size) for _ in range(num_liquid_layers)
        ])

        # Ausgangsschicht
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size), nn.LeakyReLU(0.02))

        # Lernbare Parameter für die Zustandsübergänge
        self.L_ABCD = nn.Parameter(torch.Tensor([0.25] * 4))

    def forward(self, x, state=None):
        # Eingangsschicht
        x = self.input_layer(x)

        # Initialisiere den Zustand, wenn keiner gegeben ist
        if state is None:
            state = torch.zeros(self.num_liquid_layers, x.size(0), self.hidden_size, device=x.device)
        else:
            state = state.clone()
        new_states = []
        # Flüssige Schichten
        for i, liquid_layer in enumerate(self.liquid_layers):
            x, new_state = liquid_layer(x, state[i], self.L_ABCD)
            new_states.append(new_state)

        new_state = torch.stack(new_states)

        # Ausgangsschicht
        output = self.output_layer(x)

        return output, new_state

    def read_state(self, state=None):
        if state is None:
            state = torch.zeros(self.num_liquid_layers, 1, self.hidden_size, device=next(self.parameters()).device)

        # Verwende den letzten Zustand der letzten flüssigen Schicht
        last_state = state[-1]
        return self.output_layer(last_state)


class LiquidLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h, L_ABCD):
        dh = torch.tanh(self.W(x) + self.U(h))
        h_new = h + (dh - h) / self.tau

        # Anwenden der lernbaren Parameter L_ABCD
        A, B, C, D = L_ABCD

        return A * h_new + B * h + C * x + D * dh, h_new.clone()


class ForwardForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, threshold=2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),  # Batch-Normalisierung hinzufügen
            nn.LeakyReLU()
        ))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_sizes[i], hidden_sizes[i])
                )
            )
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.threshold = threshold
        self.optimizers = [optim.Adam(layer.parameters()) for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

    def forward_forward(self, x_pos, x_neg):
        losses = []
        for i, layer in enumerate(self.layers):
            self.optimizers[i].zero_grad()
            x_pos_out = torch.relu(layer(x_pos))
            x_neg_out = torch.relu(layer(x_neg))
            loss = torch.mean(torch.relu(1 - x_pos_out)) + torch.mean(torch.relu(1 + x_neg_out))
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizers[i].step()

            # Update inputs for next layer
            x_pos = x_pos_out.detach()  # Detach to prevent in-place modification issues
            x_neg = x_neg_out.detach()  # Detach to prevent in-place modification issues
            losses.append(loss.item())

        return losses

    def predict(self, x):
        # x = x.view(1, -1)
        with torch.no_grad():
            return self.forward(x)



def test_neural_system():
    # Definiere die Dimensionen für den Test
    input_size = 10
    hidden_sizes = [20, 30]
    output_size = 5
    liquid_state_hidden_size = 40
    liquid_state_num_layers = 2

    # Erstelle eine Instanz des NeuralSystems
    neural_system = NeuralSystem(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        liquid_state_hidden_size=liquid_state_hidden_size,
        liquid_state_num_layers=liquid_state_num_layers
    )

    # Teste den Forward-Pass
    x_pos = torch.randn(1, input_size)
    x_neg = torch.randn(1, input_size)
    output = neural_system(x_pos, x_neg)
    assert output.shape == (1, output_size), f"Erwartete Ausgabegröße (1, {output_size}), aber erhielt {output.shape}"

    # Teste das Training
    target = torch.randn(1, output_size)
    loss = neural_system.train_step(x_pos, x_neg, target)
    assert isinstance(loss, float), f"Erwarteter Verlust vom Typ float, aber erhielt {type(loss)}"

    # Teste die Encoder
    text_input = "Hello, world!"
    audio_input = torch.randn(1, 1000)
    image_input = torch.randn(1, 3, 64, 64)
    video_input = torch.randn(1, 30, 3, 64, 64)

    encoded_text = neural_system.encoder(text_input)
    encoded_audio = neural_system.encoder(audio_input)
    encoded_image = neural_system.encoder(image_input)
    encoded_video = neural_system.encoder(video_input)

    assert encoded_text.shape[1] == input_size, f"Erwartete Text-Encodergröße {input_size}, aber erhielt {encoded_text.shape[1]}"
    assert encoded_audio.shape[1] == input_size, f"Erwartete Audio-Encodergröße {input_size}, aber erhielt {encoded_audio.shape[1]}"
    assert encoded_image.shape[1] == input_size, f"Erwartete Bild-Encodergröße {input_size}, aber erhielt {encoded_image.shape[1]}"
    assert encoded_video.shape[1] == input_size, f"Erwartete Video-Encodergröße {input_size}, aber erhielt {encoded_video.shape[1]}"

    # Teste das Speichern und Laden
    neural_system.save("test_model")
    loaded_model = NeuralSystem.load("test_model")

    # Überprüfe, ob das geladene Modell die gleiche Ausgabe produziert
    loaded_output = loaded_model(x_pos, x_neg)
    assert torch.allclose(output, loaded_output), "Das geladene Modell produziert eine andere Ausgabe als das ursprüngliche Modell"

    # Teil 1: Single-Modal, nicht zeitrelevante Daten (Bild)
    def test_single_modal_not_time_relevant():
        image_data = torch.rand(1, 3, 64, 64)  # Simuliertes Bild
        output = neural_system.forward(image_data)
        assert output.shape == (1, output_size), "Ausgabeform für Bild-Input ist falsch"
        print("Test 1 (Single-Modal, nicht zeitrelevant) erfolgreich")

    # Teil 2: Single-Modal, nicht zeitrelevante Daten (Text)
    def test_single_modal_not_time_relevant_text():
        text_data = "Dies ist ein Testtext"
        output = neural_system.forward(text_data)
        assert output.shape == (1, output_size), "Ausgabeform für Text-Input ist falsch"
        print("Test 2 (Single-Modal Text, nicht zeitrelevant) erfolgreich")

    # Teil 3: Single-Modal, zeitrelevante Daten (Audio)
    def test_single_modal_time_relevant():
        audio_data = torch.rand(1, 1, 22050)  # Simulierte Audio-Daten (1 Sekunde bei 22050 Hz)
        output = neural_system.forward(audio_data)
        assert output.shape == (1, output_size), "Ausgabeform für Audio-Input ist falsch"
        print("Test 3 (Single-Modal, zeitrelevant) erfolgreich")

    # Teil 4: Multi-Modal, zeitrelevante Daten (Video)
    def test_multi_modal_time_relevant():
        video_data = torch.rand(1, 30, 3, 64, 64)  # Simuliertes Video (30 Frames)
        output = neural_system.forward(video_data)
        assert output.shape == (1, output_size), "Ausgabeform für Video-Input ist falsch"
        print("Test 4 (Multi-Modal, zeitrelevant) erfolgreich")

    # Ausführen der Tests
    test_single_modal_not_time_relevant()
    test_single_modal_not_time_relevant_text()
    test_single_modal_time_relevant()
    test_multi_modal_time_relevant()

    print("Alle Tests erfolgreich abgeschlossen!")

    print("Alle Tests erfolgreich durchgeführt!")
