import random

import torch
from torch import nn

from test import NeuralSystem

sdata = {
    "input_size": 8,
    "output_size": 8,
    "liquid_state_hidden_size": 16,
    "liquid_state_num_layers": 3,
    "hidden_sizes": [16, 8],
    "loss_fn": "bce2",
}


def decimal_to_binary(n, num_bits=8):
    return [int(b) for b in format(n, f'0{num_bits}b')]


def generate_data(start, end, exclude=None, _list=None):
    if exclude is None:
        exclude = []
    if _list is None:
        _list = range(start, end)
    return [
        (decimal_to_binary(x, num_bits=sdata["input_size"]),
         decimal_to_binary(x, num_bits=sdata["output_size"]))
        for x in _list if x not in exclude
    ]


def add_noise(binary_list, noise_level=0.1):
    return [max(0, min(1, b + random.uniform(-noise_level, noise_level))) for b in binary_list]


def lern_binary(ff_lsm: NeuralSystem, t_data, epoch=100, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    for _ in range(epoch):
        losses = 0
        losses_ = 0
        for inp, out in t_data:
            __, loss = ff_lsm.train(add_noise(inp), add_noise(inp, 0.5), out, adjust_liquid=False)
            losses += loss
            losses_ += sum(__) / len(__)
        avg_loss = losses / len(t_data)
        print(f"Epoch {_ + 1}/{epoch}, Loss: {avg_loss:.4f} {losses_ / len(t_data):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Test at {_ + 1}")

            accuracy = evaluate(FF_LSM, data_e)
            print(f"Bit Accuracy: {accuracy:.2%}")
            FF_LSM.save(f"./models/binary{_ + 1}_model")

            if accuracy > 99:
                break

            patience_counter = 0


def evaluate(ff_lsm, data):
    correct_bits = 0
    total_bits = 0
    for inp, target in data:
        output = ff_lsm.forward(inp).detach().numpy()
        predicted = (output > 0.5).astype(int)
        correct_bits += sum(p == t for p, t in zip(predicted, target, strict=False))
        total_bits += len(target)
    return correct_bits / total_bits


if __name__ == "__main__":
    start = 0
    end = 255
    exclude = range(1, 255, 3)
    data_t = generate_data(start, end, exclude=exclude)

    print("Training date:", len(data_t))

    # get_eval date
    data_e = generate_data(0, 1, _list=exclude)

    FF_LSM = NeuralSystem(**sdata)
    FF_LSM.liquid_state.L_ABCD = nn.Parameter(torch.Tensor([1.01, .99, 0, 0]))

    accuracy = evaluate(FF_LSM, data_e)
    print(f"Bit Accuracy: {accuracy:.2%}")

    lern_binary(FF_LSM, data_t, epoch=1000)

    # Nach dem Training
    accuracy = evaluate(FF_LSM, data_e)
    print(f"Bit Accuracy: {accuracy:.2%}")
    FF_LSM.save("/models/binary")
