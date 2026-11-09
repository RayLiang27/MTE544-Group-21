import matplotlib.pyplot as plt
from utilities import FileReader
import numpy as np

def parabola(x):
    return x**2

def sigmoid(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def plot_errors(filename, typ=None, left_title=None, right_title=None,
                left_legend=None, right_legend=None):

    headers, values = FileReader(filename).read_file()

    time_list = []

    first_stamp = values[0][-1]

    for val in values:
        time_list.append(val[-1] - first_stamp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: X vs Y trajectory
    axes[0].plot([lin[0] for lin in values], [lin[1] for lin in values],
                 label=left_legend if left_legend else "actual trajectory")
    if typ == "para":
        axes[0].plot(np.linspace(0, 1.5, 15), parabola(np.linspace(0, 1.5, 15)), "r--", label="command trajectory")
    elif typ == "sig":
        axes[0].plot(np.linspace(0, 2.5, 25), sigmoid(np.linspace(0, 2.5, 25)), "r--", label="command trajectory")
    if left_title:
        axes[0].set_title(left_title)
    axes[0].legend()
    axes[0].set_xlabel("e (m)")
    axes[0].set_ylabel("e-dot (m/s)")
    axes[0].grid()

    # Right plot: values over time
    thingy = [" (m)", " (m)", " (rad)"]
    for i in range(0, len(headers) - 1):
        if right_legend and i < len(right_legend):
            label = right_legend[i]
        else:
            label = headers[i] + thingy[i]
        axes[1].plot(time_list, [lin[i] for lin in values], label=label)

    if right_title:
        axes[1].set_title(right_title)
    axes[1].legend()
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Value")
    axes[1].grid()

    plt.show()


import argparse

if __name__ == "__main__":

    # Add ability to set titles and legends
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--files', nargs='+', required=True, help='List of files to process')
    parser.add_argument('--type', nargs=1, required=False, help='para or sig')
    parser.add_argument('--left-title', type=str, default=None, help='Title for the left plot')
    parser.add_argument('--right-title', type=str, default=None, help='Title for the right plot')
    parser.add_argument('--left-legend', type=str, default=None,
                        help='Legend label for the left plot data line')
    parser.add_argument('--right-legend', nargs='+', default=None,
                        help='Legend labels for the right plot lines, one per series, in header order')

    args = parser.parse_args()

    print("plotting the files", args.files)

    filenames = args.files
    typ = args.type[0] if args.type else None
    print(typ)
    for filename in filenames:
        plot_errors(filename, typ,
                    left_title=args.left_title,
                    right_title=args.right_title,
                    left_legend=args.left_legend,
                    right_legend=args.right_legend)