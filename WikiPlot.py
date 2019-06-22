from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_series(series, title=None, days=None, new_plot=True, plot_show=True, label=None, color='blue', linestyle='solid'):
    if new_plot:
        plt.figure(figsize=(20, 10))

    if days is None:
        days = range(series.shape[0])

    plt.plot(days, series, label=label, color=color, linestyle=linestyle)
    if title is not None:
        plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Views")
    plt.legend()
    if plot_show:
        plt.show()


def plot_prediction(encode, target, prediction, tail_len=50):
    encode_series_tail = np.concatenate([encode[-tail_len:], target[:1]])
    x_encode = encode_series_tail.shape[0]

    pred_steps = prediction.shape[0]

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plot_series(encode_series_tail, title="Series with tail of " + str(tail_len), days=range(1, x_encode + 1), color='blue', new_plot=False, plot_show=False,
                label='history')
    plot_series(target, days=range(x_encode, x_encode + pred_steps), color='orange', new_plot=False, plot_show=False, label='target')
    plot_series(prediction, days=range(x_encode, x_encode + pred_steps), color='teal', linestyle='--', new_plot=False, plot_show=False, label='prediction')

    x_encode = encode.shape[0]
    plt.subplot(1, 2, 2)
    plot_series(encode, title="Full series", days=range(1, x_encode + 1), color='blue', new_plot=False, plot_show=False, label='history')
    plot_series(target, days=range(x_encode, x_encode + pred_steps), color='orange', new_plot=False, plot_show=False, label='target')
    plot_series(prediction, days=range(x_encode, x_encode + pred_steps), color='teal', linestyle='--', new_plot=False, plot_show=False, label='prediction')
    plt.show()
