import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from test import (
    NeuralSystem,
    create_heatmap,
    transform_ffn_for_heatmap,
    transform_sequential_for_heatmap,
    transform_state_for_heatmap,
)


def decimalToBinary(n, ziel_laenge=4):
    x = bin(n).replace("0b", "")
    def auffuellen(sublist, laenge):
        return [0] * (laenge - len(sublist)) + sublist
    return auffuellen([int(_) for _ in x][:ziel_laenge], ziel_laenge)


def dc_conv(o_list):
    as_t = False
    if isinstance(o_list, torch.Tensor) or isinstance(o_list[0], torch.Tensor):
        o_list = o_list.detach().tolist()
        as_t = True

    def helper(_):
        if _ >= 0.55:
            return 1
        if _ <= 0.45:
            return 0
        return -1

    r = [helper(round(x, 2)) for x in o_list]
    if as_t:
        r = torch.as_tensor(r, dtype=torch.float32)
    return r


def dc_conv_(o_list):
    as_t = False
    if isinstance(o_list, torch.Tensor) or isinstance(o_list[0], torch.Tensor):
        o_list = o_list.detach().tolist()
        as_t = True

    def helper(_):
        if _ >= 0.51:
            return 1
        if _ <= 0.49:
            return 0
        return -1

    r = [helper(round(x, 2)) for x in o_list]
    if as_t:
        r = torch.as_tensor(r, dtype=torch.float32)
    return r


def dc_conv_c(o_list):
    return decimalToBinary(o_list.index(max(o_list)))


def lern_ninary(ff_lsm: NeuralSystem, t_data, epoch=100):
    losses = 0

    for _ in range(epoch):
        #print(
        #    f"Traininig, {_}/{epoch} target : {min(0.002, 1 / (epoch / 10)):.3f} current : {losses / len(t_data):.8f}")
        # if losses > 0 and losses / len(t_data) < min(0.002, 1 / (epoch / 10)):
        #     break
        losses = 0
        ___ = []
        for _i, (inp, out) in enumerate(t_data):
            # vis.liquid_state = LNN
            # lnn_output = ff_lsm.forward(inp, frozen=False)
            # loss = 0
            if FF_LSM.last_output is not None and dc_conv(list(FF_LSM.last_output.squeeze(0).detach().numpy())) == out:
                __, loss = ff_lsm.train(inp, None, out, adjust_liquid=False, auto_find_x_neg=False)
            else:
                __, loss = ff_lsm.train(inp, None, out, adjust_liquid=False)
            ___ += __
            losses += loss
        print(f"Losses : FF {sum(___) / len(___):.4f}, LS {loss:.4f}, {_}/{epoch} ")
        # lnn.save(f"models/binary")
        #random.shuffle(t_data)


def classification(lsm):
    np.random.seed(42)
    data = np.random.randn(1000, 2)
    data[:500, 0] += 2
    data[500:, 0] -= 2

    # Trainieren Sie das Modell mit Clustering
    lsm.train_clustering(data, n_clusters=2, epochs=100)

    # Testen Sie die Vorhersage
    test_point = np.array([1.5, 0])
    predicted_cluster = lsm.predict_cluster(test_point)
    print(f"Predicted cluster for test point: {predicted_cluster}")


def rl(lsm):
    import gymnasium as gym
    # Erstellen Sie eine Gym-Umgebung
    env = gym.make('Pendulum-v1')

    # Trainieren Sie das Modell mit Reinforcement Learning
    lsm.train_reinforcement(env, num_episodes=1000)


def set_up_dummy():
    return {
        "input_size": 1,
        "hidden_sizes": [1],
        "output_size": 1,
        "liquid_state_size": 1,
        "liquid_state_dim": 1
    }


def set_up_dummyv2_w():
    return {
        "input_size": 1,
        "hidden_sizes": [1],
        "output_size": 1,
        "liquid_state_size": 4,
        "liquid_state_dim": 3
    }


def set_up_lern_ninary():
    return {
        "input_size": 1,
        "output_size": 3,
        "liquid_state_size": 2,
        "liquid_state_dim": 3,
        "hidden_sizes": [2],
    }


def set_up_lern_ninary_38():
    return {
        "input_size": 3,
        "output_size": 8,
        "liquid_state_hidden_size": 3,
        "liquid_state_num_layers": 4,
        "hidden_sizes": [3],
        "loss_fn": "mae",
    }


def run_dummy(epochs=10, target=0.5):
    target = round(target, 3)
    for i in range(epochs):
        out = FF_LSM.forward([target])
        if abs(out.item() - target) <= 0.002:
            loss, l_loss = FF_LSM.train([target], [target], [target])
        else:
            loss, l_loss = FF_LSM.train([target], None, [target])
        print(
            f" out: {out.item()} {i}/{epochs}  target : {target} loss: {sum(loss) / len(loss):.2f},{l_loss} temp: {FF_LSM.liquid_state.temperature}")
        if round(float(out.item()), 3) == target:
            break
    if round(float(out.item()), 3) == target:
        print(f"Accomplished in {i} epochs")


def run_dummyv2(epochs=10, targets=None):
    if targets is None:
        targets = [0.5, -0.5]
    targets = [round(target, 3) for target in targets]

    for _i in range(epochs):
        losses = 0
        losses_ = 0
        for target in targets:
            x_pos = target
            _, l_loss = FF_LSM.train_step([x_pos], None, [target], num_iterations_x_neg=1)

            losses += l_loss
            losses_ += sum(_) / len(_)
        # print(
        #     f" {i+1}/{epochs}  loss: {losses / len(targets):.6f} {losses_ / len(targets):.6f}"
        # )

        # if (losses / len(targets) + (losses_ / len(targets))) < 1 / epochs:
        #     print(f"DONE AAfter : {epochs} {(losses / len(targets) + (losses_ / len(targets)))}")
        #     break
    return losses / len(targets) , (losses_ / len(targets))


def run_binary(epochs, start=0, end=16, exclude=None, i_len=2, o_len=4):
    if exclude is None:
        exclude = [190, 200, 240, 1, 32, 7, 94, 123]

    def generate_data(start_=0, end_=1, list_=None, ex=None):
        def auffuellen(sublist, laenge):
            return [0] * (laenge - len(sublist)) + sublist
        if ex is None:
            ex = []
        if list_ is None:
            list_ = range(start_, end_)
        def ret_helper():
            return [[[float(_) for _ in auffuellen(list(str(x))[:i_len], i_len)],
                                       auffuellen([float(_) for _ in bin(x).replace("0b", "")][:o_len], o_len)] for x in
                                      list_ if x not in ex]
        return ret_helper()

    def generate_data_classes(start_=0, end_=1, list_=None, ex=None):
        def g_let(max_len):
            return [0] * max_len

        def classes(max_len, ind):
            a = g_let(max_len)
            a.insert(ind, 1)
            return a

        def auffuellen(sublist, laenge):
            return [0] * (laenge - len(sublist)) + sublist
        if ex is None:
            ex = []
        if list_ is None:
            list_ = range(start_, end_)
        def ret_helper():
            return [[[float(_) for _ in auffuellen(list(str(x))[:i_len], i_len)], classes(end, x)] for x in
                                      list_ if x not in ex]
        return ret_helper()

    # get_training date
    data_t = generate_data(start, end, ex=exclude)

    print("Training date:", len(data_t), data_t[:3])

    # get_eval date
    data_e = generate_data(list_=exclude)

    print("Evaluation date:", data_e)
    random.shuffle(data_e)
    lern_ninary(FF_LSM, data_t, 5)
    lern_ninary(FF_LSM, data_t, epochs)
    print("Evaluation Remember", data_t[:3])
    lnn_output = [(inp, out, list(FF_LSM.forward(inp, frozen=True).squeeze(0).detach().numpy())) for (inp, out) in
                  data_t if dc_conv_(list(FF_LSM.forward(inp, frozen=True).squeeze(0).detach().numpy())) == out]
    print(len(lnn_output), lnn_output)
    print(dc_conv_(list(FF_LSM.forward(data_e[0][0], frozen=True).squeeze(0).detach().numpy())),
          dc_conv_(list(FF_LSM.forward(data_e[-1][0], frozen=True).squeeze(0).detach().numpy())))
    print("Evaluation New")
    lnn_output = [(inp, out, dc_conv_(list(FF_LSM.forward(inp, frozen=True).squeeze(0).detach().numpy()))) for
                  (inp, out)
                  in data_e]
    print(lnn_output)


def show():
    fig3, ani3 = create_heatmap(transform_ffn_for_heatmap(FF_LSM.ff_network.layers), title="FF-Input-Model")

    fig0, ani0 = create_heatmap(transform_sequential_for_heatmap(FF_LSM.liquid_state.i_model), title="L-I-Model")

    fig1, ani1 = create_heatmap(transform_state_for_heatmap(FF_LSM.liquid_state.state), is_complex=True,
                                title="LiquidState")

    fig2, ani2 = create_heatmap(transform_sequential_for_heatmap(FF_LSM.liquid_state.o_model), title="L-O-Model")

    plt.show()


def set_up_dummyv2():
    return {
        "input_size": 1,
        "output_size": 1,
        "liquid_state_hidden_size": 64,
        "liquid_state_num_layers": 12,
        "hidden_sizes": [64, 32, 64],
        "loss_fn": "abs",
    }

def _test_dummy(test_range=10, start=0, til=12, start_til=0):
    print("TEST: remembering")
    te = 0
    for i in range(start, test_range):
        out = FF_LSM.forward([i / test_range], None, frozen=True).item()
        print(f"in/out: {i / test_range}/{out}  err: {(i / test_range) - out}")
        te += abs((i / test_range) - out)
    print(f"Total error Fs: {te / test_range:.4f}")
    print("TEST: Reverse")
    for i in range(start, test_range)[::-1]:
        out = FF_LSM.forward([i / test_range], None, frozen=True).item()
        print(f"in/out: {i / test_range}/{out}  err: {(i / test_range) - out}")
        te += abs((i / test_range) - out)
    print(f"Total error Fs: {te / test_range:.4f}")
    print("TEST: generelly")
    for i in range(start_til, til):
        p = FF_LSM.forward([i / til], None, frozen=False).item()
        if abs(p - i / til) > 1e-3:
            FF_LSM.forward([i / til], [p], frozen=False).item()
        else:
            FF_LSM.forward([p], None, frozen=False).item()
        # print(p - p1, FF_LSM.last_ff_losses)
        # print(f"in/out: {i/100}/{out[0]}  err: {(i/10)-out[0]}")
    print("Live: Lerining test")
    te = 0
    for i in range(start, test_range):
        out = FF_LSM.forward([i / test_range], None, frozen=True).item()
        print(f"in/out: {i / test_range}/{out}  err: {(i / test_range) - out}")
        te += abs((i / test_range) - out)
    print(f"Total error ALTs: {te / test_range:.4f}")


def run_dummyv2_(epochs=200):
    ts = [x / 100 for x in range(0, 100, 2)]
    random.shuffle(ts)
    train_data = ts[:25]
    val_data = ts[25:]

    for epoch in range(epochs):
        FF_LSM.train()
        train_loss, v = run_dummyv2(epochs=1, targets=train_data)
        FF_LSM.eval()
        val_loss = validate(val_data)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.6f},{v:.6f}, Val Loss: {val_loss:.6f}")

        # if epoch % 10 == 0:
        #     _test_dummy(100, 0, 1, 1)
    _test_dummy(100, 0, 1, 1)


def validate(data):
    total_loss = 0
    for target in data:
        output = FF_LSM.forward([target], frozen=True)
        loss = FF_LSM.loss_fn(output, torch.tensor([target]))
        total_loss += loss.item()
    return total_loss / len(data)


if __name__ == "__main__":
    sdata = set_up_dummyv2()
    FF_LSM: NeuralSystem = NeuralSystem(**sdata)


    def loss_helper(target, value):
        return abs(target - value)


    def custom_loss(pred, target):
        # print(pred, target)
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        return mse + 0.5 * mae


    FF_LSM.loss_fn.add_custom_loss("custom", custom_loss)
    FF_LSM.loss_fn.add_custom_loss("abs", loss_helper)
    # visualizer = LiquidStateVisualizer(FF_LSM)
    # visualizer.show_full_state()
    # visualizer.show_live_state()
    run_dummyv2_(400)
    # visualizer.show_full_state()
    # show()
    # visualizer.show_full_state()
