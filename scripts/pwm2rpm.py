# import scipy as sp 
import numpy as np
import argparse
import matplotlib.pyplot as plt
# from numpy.polynomial import Polynomial as poly
# from numpy.polynomial.polynomial import Polynomial
import cvxpy as cp


def loadFile(filename):
    fileData = np.loadtxt(filename, delimiter=',', skiprows=1, ndmin=2)
    return fileData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='+')
    args = parser.parse_args()

    data = None
    for f in args.filename:
        data_f = loadFile(f)
        if data is None:
            data = data_f
        else:
            data = np.vstack((data, data_f))

    # thrust[g],pwm,vbat[V],rpm1,rpm2,rpm3,rpm4

    thrust = data[:,0] / 4  # g, per motor
    pwm = data[:,1]         # PWM value
    vbat = data[:,2]        # V, battery voltage, 

    pwm_normalized = pwm / 65535.0

    a = cp.Variable()
    b = cp.Variable()
    # pwm = a + b * thrust
    cost = cp.sum_squares(a + b * thrust - pwm_normalized)
    prob = cp.Problem(cp.Minimize(cost), [])
    prob.solve()
    print("rpm2pwm: {}".format(a.value))
    print("rpm2pwm: {}".format(b.value))

    eval = a.value + b.value * thrust

    # plotting

    fig, ax = plt.subplots(2)
    ax[0].plot(thrust)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('thrust [g]')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].scatter(thrust, pwm_normalized, label='data')
    ax[1].plot(thrust, eval, label='fit')

    ax[1].set_xlabel('thrust')
    ax[1].set_ylabel('pwm')
    ax[1].legend()
    ax[1].grid(True)

    plt.show()
