import time
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import os

def _convert_to_xy(data):
    if isinstance(data, dict):
        x,y = zip(*(sorted(data.items())))
    elif isinstance(data, list):
        x = np.arange(len(data))+1
        y = data
    elif isinstance(data, np.ndarray):
        x = np.arange(len(data))+1
        y = list(data) # Assume 1D
    return (x,y)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    prefix = np.array([sum(x[:i+1])/len(x[:i+1]) for i in range(N-1)])
    smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)
    return np.concatenate((prefix, smoothed))

# Data params should be lists or 1D arrays or dicts. `name` is a string
# identifier for the output figure PNG; if not provided, it will default to
# using the current datetime.
def plot(y, title, xlabel, ylabel, path, smooth=False):
    fig = plt.figure(figsize=(8,4), dpi=80)
    ax = fig.add_subplot(111)
    nbins = 10

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x,y = _convert_to_xy(y)
    if smooth:
        y = running_mean(y, 100)
    plt.plot(x,y,label='episode reward')
    n = max(x)
    """
    ax = fig.add_subplot(212)
    title = 'loss over episodes'
    xlabel, ylabel = 'episodes', 'loss'
    nbins = 10

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x,y = _convert_to_xy(train_loss)
    plt.plot(x,y,label='training loss')
    n = max(max(x),n)
    """
    ticks = (np.arange(nbins) + 1) * n//nbins
    plt.xticks(ticks)

    #ax.set_ylim(bottom=0)
    ax.margins(0)
    ax.legend()

    plt.savefig(path)
