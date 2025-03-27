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

    thrust = (data[:,0] / 1000.0) / 4 * 9.81  # N, per motor
    pwm = data[:,1]         # PWM value
    vbat = data[:,2]        # V, battery voltage, 

    pwm_normalized = pwm / 65535.0

    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()
    # pwm = a + b * thrust + c * vbat * thrust
    cost = cp.sum_squares(a + b * thrust + c * cp.multiply(vbat, thrust) - pwm_normalized)
    prob = cp.Problem(cp.Minimize(cost), [])
    prob.solve()
    print("rpm2pwm: a {}".format(a.value))
    print("rpm2pwm: b {}".format(b.value))
    print("rpm2pwm: c {}".format(c.value))

    eval = a.value + b.value * thrust + c.value * vbat * thrust

    # plotting

    fig, ax = plt.subplots(2)
    ax[0].plot(thrust)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('thrust [N]')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].scatter(thrust, pwm_normalized, label='data')
    ax[1].scatter(thrust, eval, label='fit')

    ax[1].set_xlabel('thrust [N]')
    ax[1].set_ylabel('pwm [0..1]')
    ax[1].legend()
    ax[1].grid(True)

    plt.show()
