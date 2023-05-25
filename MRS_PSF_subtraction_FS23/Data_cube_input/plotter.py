import matplotlib.pyplot as plt
import numpy as np


def plot(data, title, vmax=200, show=True):
    data = np.array(data, dtype=np.float64) - float(np.median(data))
    plt.imshow(data, cmap='gray', vmin=0, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()

def cplot(data, title, vmax=200, show=True):
    data = np.array(data, dtype=np.float64) - float(np.median(data))
    plt.imshow(data, cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()