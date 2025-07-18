import csv
from matplotlib import pyplot
import pandas as pd

def send_times_csv(inputs, times, fname, label_prefix, batch_sizes=None, input_prefix="steps"):
    """
    Writes timing data to a CSV file. If multiple timing batches are provided,
    each gets a separate column. Otherwise, it's written as a single time series.

    Parameters:
    - inputs: list of step counts
    - times: list of timing lists (or a single timing list)
    - fname: output CSV filename
    - label_prefix: used in column headers (e.g. 'GPU' or 'CPU Parallel')
    - batch_sizes: list of batch sizes if applicable, or None
    """

    data = {input_prefix: inputs}

    if batch_sizes:
        for i, batch_size in enumerate(batch_sizes):
            col_name = f"{label_prefix} - Batch size {batch_size} (s)"
            data[col_name] = times[i]
    else:
        # Normalize times to a list of lists if there is only one batch
        times = [times]
        data[f"{label_prefix}"] = times[0]

    df = pd.DataFrame(data)
    df.to_csv(fname, index=False)

    print(f"Timing data written to {fname}")

def read_timing_csv(fname):
    df = pd.read_csv(fname)
    return df

import matplotlib.pyplot as plt
import pandas as pd

def plot_timings(df, x_col='steps', y_col='time (s)', title='Timing Results', log_x=True, output='timing_results.png'):
    """
    Create a timing plot from a pandas DataFrame.

    Parameters:
    - df: DataFrame containing timing data
    - x_col: name of the column to use for the x-axis (e.g., 'steps')
    - log_x: whether to use logarithmic scale for x-axis
    - output: filename for the saved plot image
    """
    print("begin plotting script")
    plt.figure(figsize=(10, 6))

    # Plot each column except the x-axis
    for col in df.columns:
        if col == x_col:
            continue
        plt.plot(df[x_col], df[col], label=col, marker='o')

    if log_x:
        plt.xscale('log')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output)
    print(f"Plot saved to {output}")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_fps(df, x_col='steps', title='', output=None):
    """
    Plot a grouped bar chart of FPS data across different steps and simulators.

    Parameters:
    - df: pandas DataFrame with one column as x_col (e.g., 'steps')
    - title: plot title
    - output: filename to save the plot (optional)
    """
    steps = df[x_col].values
    metrics = df.drop(columns=[x_col])
    labels = metrics.columns
    values = [metrics[label].values for label in labels]

    x = np.arange(len(steps))  # label locations (0, 1, 2, ...)
    width = 0.8 / len(labels)  # width of each bar within a group

    plt.figure(figsize=(12, 6))

    for i, (label, y) in enumerate(zip(labels, values)):
        offset = (i - len(labels)/2) * width + width/2
        plt.bar(x + offset, y, width, label=label)

    plt.xticks(x, steps)
    plt.xlabel(x_col)
    plt.ylabel("FPS")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()
