# utils.py
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

def plot_results(scores_no_model, scores_with_model, figname, win_rate):
    plt.plot(scores_no_model, label="Without Opponent Model")
    if scores_with_model is not None:
        plt.plot(scores_with_model, label="With Opponent Model")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.text(
        0.05, 0.95,
        f"Win Rate: {win_rate:.2f}",
        transform=plt.gca().transAxes,
        ha="left", va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
    )
    plt.legend()
    plt.title(figname)
    plt.savefig(rf'C:\Users\25765\Desktop\Cornell\ORIE-5570 RL with OR\RL Project\project_policy_gradient\results\{figname}.png')
    plt.show()

def moving_average(x, window=100):
    x = np.array(x, dtype=float)
    ma = []

    for i in range(len(x)):
        if i < window:
            # average all data so far
            ma.append(x[:i+1].mean())
        else:
            # sliding window average
            ma.append(x[i-window+1:i+1].mean())

    return np.array(ma)[50:]


# def moving_average(x, window=100): 
#     return np.convolve(x, np.ones(window)/window, mode="valid")

