""" plotting code """

# system imports
import sys

# third party imports
import numpy as np
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt

# local imports


def plot(x1x2file, h1file, h2file):
    """ x """

    try:
        data_x1x2 = np.loadtxt(x1x2file)
        data_h1 = np.loadtxt(h1file)
        data_h2 = np.loadtxt(h2file)
    except Exception as e:
        print("Failed to load the data files?")
        raise e

    # extract the simulation data
    x1, x2 = data_x1x2[:, 1], data_x1x2[:, 2]

    # plot the histogram of the simulation data
    n1, bins, patches = plt.hist(x1, 50, density=True, facecolor='g', alpha=0.5)
    n2, bins, patches = plt.hist(x2, 50, density=True, facecolor='r', alpha=0.5)

    # extract the SOS data
    q1, h1 = data_h1[:, 0], data_h1[:, 1]
    q2, h2 = data_h2[:, 0], data_h2[:, 1]

    # plot the SOS data
    plt.plot(q1, h1)
    plt.plot(q2, h2)

    # annotate the plot
    plt.xlabel('q')
    plt.ylabel('Probability')
    plt.grid(True)

    # save the figure
    plt.savefig('f.png')


if (__name__ == "__main__"):

    assert len(sys.argv) == 4

    x1x2file = sys.argv[1]
    h1file = sys.argv[2]
    h2file = sys.argv[3]

    print(
        "Attempting to plot using these files:\n"
        f"{x1x2file}"
        f"{h1file}"
        f"{h2file}"
    )

    plot(x1x2file, h1file, h2file)
