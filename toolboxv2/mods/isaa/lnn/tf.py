import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================
# 1. Forward–Forward (FF) Model
# =============================

# In the FF algorithm the network does two forward passes:
#   - A “positive” pass using inputs appended with the correct one-hot label.
#   - A “negative” pass using the same inputs but with a (random) incorrect label.
#
# For each layer we compute a “goodness” (here, the mean squared activation)
# and use a softplus loss to encourage goodness above (for positive data)
# or below (for negative data) a fixed threshold.
#
# The model below is defined as a sequence of Dense+ReLU+LayerNorm layers.
# The call() method can return the list of activations (one per layer)
# so that we can compute the layer‐wise goodness.

class ForwardForwardModel(keras.Model):
    def __init__(self, input_dim, hidden_dims, output_dim, threshold=1.0):
        """
        input_dim should include the extra label dimensions (e.g. 784+10 for MNIST)
        """
        super().__init__()
        self.threshold = threshold
        self.ff_layers = []
        # Build hidden layers
        for h in hidden_dims:
            self.ff_layers.append(layers.Dense(h, activation='relu'))
            self.ff_layers.append(layers.LayerNormalization())
        # Final layer (linear output; the network is trained with a local objective)
        self.ff_layers.append(layers.Dense(output_dim))

    def call(self, x, return_activations=False):
        activations = []
        for layer in self.ff_layers:
            x = layer(x)
            activations.append(x)
        if return_activations:
            return activations
        else:
            return x


# A custom training step that implements the two forward passes.
@tf.function
def ff_train_step(model, optimizer, x, y, threshold=1.0):
    # x: [batch, 784]; y: [batch] (integer labels)
    batch_size = tf.shape(x)[0]
    # Create one-hot label vectors
    y_onehot = tf.one_hot(y, depth=10)
    # Positive input: append correct label
    x_pos = tf.concat([x, y_onehot], axis=1)

    # Negative input: for each sample, choose a wrong label.
    # (We shift the label by a random number between 1 and 9 modulo 10.)
    rand_shift = tf.random.uniform(shape=[batch_size], minval=1, maxval=10, dtype=tf.int32)
    y_neg = (y + rand_shift) % 10
    y_neg_onehot = tf.one_hot(y_neg, depth=10)
    x_neg = tf.concat([x, y_neg_onehot], axis=1)

    with tf.GradientTape() as tape:
        # --- Positive pass ---
        acts_pos = model(x_pos, return_activations=True)
        loss_pos = 0
        for act in acts_pos:
            # Compute goodness per sample (mean squared activation)
            goodness = tf.reduce_mean(tf.square(act), axis=1)
            loss_pos += tf.reduce_mean(tf.nn.softplus(threshold - goodness))

        # --- Negative pass ---
        acts_neg = model(x_neg, return_activations=True)
        loss_neg = 0
        for act in acts_neg:
            goodness = tf.reduce_mean(tf.square(act), axis=1)
            loss_neg += tf.reduce_mean(tf.nn.softplus(goodness - threshold))

        loss = loss_pos + loss_neg

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
    return loss


# For prediction, we “probe” each label candidate by appending each one-hot
# label to the input and then summing the goodness (across all layers).
def ff_predict(model, x, threshold=1.0):
    batch_size = tf.shape(x)[0]
    goodness_per_label = []
    for label in range(10):
        label_onehot = tf.one_hot(tf.fill([batch_size], label), depth=10)
        x_candidate = tf.concat([x, label_onehot], axis=1)
        acts = model(x_candidate, return_activations=True)
        goodness = 0
        for act in acts:
            goodness += tf.reduce_mean(tf.square(act), axis=1)
        goodness_per_label.append(goodness)
    # Stack goodness from each candidate label and choose the label with highest total goodness.
    goodness_all = tf.stack(goodness_per_label, axis=1)  # shape: [batch, 10]
    return tf.argmax(goodness_all, axis=1)


# =============================
# 2. Spiking Liquid (LSM) Model
# =============================

# In a liquid state machine the input is encoded into spike trains (e.g., using Poisson rate coding)
# and then fed into a fixed recurrent network of spiking neurons.
# Here we define a simple Leaky Integrate-and-Fire (LIF) reservoir whose weights are fixed
# and only a final “readout” Dense layer is trained.
#
# For simplicity, we use a for-loop over discrete time steps and a simple threshold (Heaviside)
# activation. (In a full implementation you might use surrogate gradients.)

# Poisson encoder: converts a (normalized) static input into a spike train.
def poisson_encoder(x, time_steps, max_rate=1.0):
    """
    x: [batch, features] with values in [0,1]
    Returns: spikes of shape [batch, time_steps, features]
    """
    x_expanded = tf.expand_dims(x, axis=1)  # [batch, 1, features]
    x_tiled = tf.tile(x_expanded, [1, time_steps, 1])
    random_tensor = tf.random.uniform(tf.shape(x_tiled))
    spikes = tf.cast(random_tensor < x_tiled * max_rate, tf.float32)
    return spikes


# A custom layer implementing an LSM reservoir of LIF neurons.
class LiquidStateMachine(layers.Layer):
    def __init__(self, input_dim, reservoir_size, time_steps,
                 alpha=0.9, threshold=1.0, connectivity=0.1):
        """
        input_dim: number of input features
        reservoir_size: number of spiking neurons in the reservoir
        time_steps: number of time steps for simulation
        alpha: leak constant
        threshold: spiking threshold
        connectivity: probability of a connection in the recurrent weight matrix
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.time_steps = time_steps
        self.alpha = alpha
        self.threshold = threshold
        self.connectivity = connectivity
        # Fixed random input weights
        self.W_in = self.add_weight(
            "W_in", shape=(input_dim, reservoir_size),
            initializer=tf.random_normal_initializer(), trainable=False)
        # Fixed random recurrent weights (sparse)
        rec_init = tf.random.normal((reservoir_size, reservoir_size))
        mask = tf.cast(tf.random.uniform((reservoir_size, reservoir_size)) < connectivity, tf.float32)
        rec_init = rec_init * mask
        self.W_rec = self.add_weight(
            "W_rec", shape=(reservoir_size, reservoir_size),
            initializer=lambda shape, dtype: rec_init, trainable=False)

    def call(self, inputs):
        """
        inputs: spike trains of shape [batch, time_steps, input_dim]
        Returns: a feature vector per sample (e.g., average firing rate) of shape [batch, reservoir_size]
        """
        batch_size = tf.shape(inputs)[0]
        # Initialize reservoir state (membrane potential)
        u = tf.zeros((batch_size, self.reservoir_size))
        # Initialize spikes (for recurrent input)
        spikes = tf.zeros((batch_size, self.reservoir_size))
        # Record spikes for each time step
        spike_record = []
        for t in range(self.time_steps):
            x_t = inputs[:, t, :]  # [batch, input_dim]
            # Compute input current: from external input and recurrent spikes
            I = tf.matmul(x_t, self.W_in) + tf.matmul(spikes, self.W_rec)
            # Update membrane potential
            u = self.alpha * u + I
            # Generate spikes: simple thresholding (non-differentiable; for training the readout only)
            new_spikes = tf.cast(u >= self.threshold, tf.float32)
            # Reset the membrane potential where spikes occurred
            u = u * (1 - new_spikes)
            spikes = new_spikes  # update recurrent input
            spike_record.append(new_spikes)
        # Stack spikes over time and compute the mean firing rate
        spike_record = tf.stack(spike_record, axis=1)  # [batch, time_steps, reservoir_size]
        firing_rate = tf.reduce_mean(spike_record, axis=1)  # [batch, reservoir_size]
        return firing_rate


# Define an SNN model that uses Poisson encoding, the LSM, and a trainable readout.
class SNNModel(keras.Model):
    def __init__(self, input_dim, reservoir_size, time_steps, num_classes):
        super().__init__()
        self.time_steps = time_steps
        self.lsm = LiquidStateMachine(input_dim, reservoir_size, time_steps)
        self.readout = layers.Dense(num_classes)  # only the readout is trainable

    def call(self, x):
        # x: [batch, features] (values in [0,1])
        spikes = poisson_encoder(x, self.time_steps)
        features = self.lsm(spikes)
        logits = self.readout(features)
        return logits


# =============================
# 3. Combined Model: FF -> LSM -> FF
# =============================

# Here we first use a feedforward (dense) layer to process the static input.
# Its output (a feature vector) is then “repeated” in time (as if it were a constant input)
# to the LSM reservoir. Finally, a second FF layer acts as the readout.
class CombinedModel(keras.Model):
    def __init__(self, input_dim, ff_hidden_dim, reservoir_size, time_steps, num_classes):
        super().__init__()
        self.ff_pre = layers.Dense(ff_hidden_dim, activation='relu')
        self.lsm = LiquidStateMachine(ff_hidden_dim, reservoir_size, time_steps)
        self.ff_post = layers.Dense(num_classes)

    def call(self, x):
        # x: [batch, input_dim]
        features = self.ff_pre(x)  # [batch, ff_hidden_dim]
        # Repeat the features over time steps for the reservoir.
        features_time = tf.expand_dims(features, axis=1)  # [batch, 1, ff_hidden_dim]
        features_time = tf.tile(features_time, [1, self.lsm.time_steps, 1])  # [batch, time_steps, ff_hidden_dim]
        reservoir_out = self.lsm(features_time)  # [batch, reservoir_size]
        logits = self.ff_post(reservoir_out)
        return logits


if __name__ == '__main__':
    # =============================
    # Data Loading and Preprocessing (MNIST)
    # =============================

    # Load MNIST (images are normalized to [0,1])
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # =============================
    # Model Instantiation and Training
    # =============================

    # --- 1. Train the Forward–Forward model ---
    # Note: The FF model expects input_dim = 784 + 10 (image pixels + one-hot label)
    ff_input_dim = 784 + 10
    ff_model = ForwardForwardModel(input_dim=ff_input_dim, hidden_dims=[128, 128], output_dim=10, threshold=1.0)
    ff_optimizer = keras.optimizers.Adam()

    # For demonstration we train for a few epochs using our custom training loop.
    batch_size = 128
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    print("Training Forward–Forward model...")
    for epoch in range(3):  # try a few epochs
        epoch_loss = 0
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_val = ff_train_step(ff_model, ff_optimizer, x_batch, y_batch, threshold=1.0)
            epoch_loss += loss_val
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / (step + 1):.4f}")

    # Evaluate FF model on test set
    ff_correct = 0
    ff_total = 0
    for x_batch, y_batch in test_dataset:
        preds = ff_predict(ff_model, x_batch)
        ff_correct += np.sum(preds.numpy() == y_batch.numpy())
        ff_total += x_batch.shape[0]
    ff_accuracy = ff_correct / ff_total
    print(f"Forward–Forward model test accuracy: {ff_accuracy:.4f}")

    # --- 2. Train the SNN (LSM) model ---
    snn_model = SNNModel(input_dim=784, reservoir_size=128, time_steps=20, num_classes=10)
    snn_model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    print("\nTraining SNN (LSM) model...")
    snn_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
    snn_test_loss, snn_test_acc = snn_model.evaluate(x_test, y_test, verbose=0)
    print(f"SNN (LSM) model test accuracy: {snn_test_acc:.4f}")

    # --- 3. Train the Combined (FF -> LSM -> FF) model ---
    combined_model = CombinedModel(input_dim=784, ff_hidden_dim=128, reservoir_size=128, time_steps=20, num_classes=10)
    combined_model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
    print("\nTraining Combined model (FF -> LSM -> FF)...")
    combined_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
    combined_test_loss, combined_test_acc = combined_model.evaluate(x_test, y_test, verbose=0)
    print(f"Combined model test accuracy: {combined_test_acc:.4f}")
