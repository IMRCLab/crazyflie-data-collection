import argparse

import matplotlib.pyplot as plt
import numpy as np

import rowan

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_csv")
    args = parser.parse_args()

    data = np.loadtxt(args.file_csv, delimiter=',')
    rpy = rowan.to_euler(rowan.normalize(data[:,7:11]), "xyz")

    fig, ax = plt.subplots(4, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(data[:,0], data[:,1+k])
        ax[0,k].set_ylabel(f"position {axis}[m]")

        ax[1,k].plot(data[:,0], data[:,4+k])
        ax[1,k].set_ylabel(f"velocity {axis}[m/s]")

        ax[2,k].plot(data[:,0], rpy[:,k])
        ax[2,k].set_ylabel(f"rotation {axis}[rad]")

        ax[3,k].plot(data[:,0], data[:,8+k])
        ax[3,k].set_ylabel(f"angular velocity {axis}[rad/s]")

    plt.show()

    fig, ax = plt.subplots(2, 2, sharex='all')
    ax[0,1].plot(data[:,0], data[:,14])
    ax[0,1].set_ylabel(f"M1 [rpm]")
    ax[1,1].plot(data[:,0], data[:,15])
    ax[1,1].set_ylabel(f"M2 [rpm]")
    ax[1,0].plot(data[:,0], data[:,16])
    ax[1,0].set_ylabel(f"M3 [rpm]")
    ax[0,0].plot(data[:,0], data[:,17])
    ax[0,0].set_ylabel(f"M4 [rpm]")
    plt.show()

