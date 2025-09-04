import numpy as np
import cvxpy as cp
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='+')
    args = parser.parse_args()

    # expected file format: pwm,vbat[V],rpm1,rpm2,rpm3,rpm4
    kws = np.array([[2.26383771e-10, 2.03708101e-10, 2.12204877e-10, 2.01874025e-10]])

    data = None
    for f in args.filename:
        data_f = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
        if data is None:
            data = data_f
        else:
            data = np.vstack((data, data_f))

    assert(data.shape[1] == 6)

    pwm = data[:,0]         # PWM value
    vbat = data[:,1]        # V, battery voltage, 
    rpm = data[:,2:6]
    force = np.repeat(kws, data.shape[0], axis=0) * rpm**2
    rpm_mean = np.mean(rpm, axis=1)

    pwm_normalized = pwm / 65535.0
    vbat_normalized = vbat / 4.2

    # print(np.multiply(rpm_mean, vbat_normalized).shape)
    # exit()

    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()
    d = cp.Variable()

    # pwm = a + b * rpm + c * rpm^2 + d * rpm * vbat_normalized
    # cost = cp.sum_squares(a + b * rpm_mean + c * rpm_mean**2 + d * cp.multiply(rpm_mean**2, vbat_normalized) - pwm_normalized)
    cost = cp.sum_squares(a + b * rpm_mean + c * rpm_mean**2 + d * vbat_normalized - pwm_normalized)
    prob = cp.Problem(cp.Minimize(cost), [])
    prob.solve()
    print("rpm2pwmA: {}".format(a.value))
    print("rpm2pwmB: {}".format(b.value))
    print("rpm2pwmC: {}".format(c.value))
    print("rpm2pwmD: {}".format(d.value))

    # pwm_fitted = a.value + b.value * rpm_mean + c.value * rpm_mean**2 + d.value * np.multiply(rpm_mean**2, vbat_normalized)
    #pwm_fitted = a.value + b.value * rpm_mean + c.value * rpm_mean**2# + d.value * np.multiply(rpm_mean**2, vbat_normalized)
    pwm_fitted = a.value + b.value * rpm_mean + c.value * rpm_mean**2 + d.value * vbat_normalized # + d.value * np.multiply(rpm_mean**2, vbat_normalized)

    # force -> pwm
    # pwm_normalized = sqrtf(force / kappa_f) * b + a

    fig, ax = plt.subplots(4)
    ax[0].plot(pwm_normalized, label="setpoint")
    ax[0].plot(pwm_fitted, label="fitted")
    ax[0].legend()

    ax[1].scatter(rpm[:,0], pwm_normalized, label='M1')
    ax[1].scatter(rpm[:,1], pwm_normalized, label='M2')
    ax[1].scatter(rpm[:,2], pwm_normalized, label='M3')
    ax[1].scatter(rpm[:,3], pwm_normalized, label='M4')
    ax[1].legend()
    ax[1].grid(True)


    ax[2].scatter(rpm_mean, pwm_normalized, label='mean')
    ax[2].scatter(rpm_mean, pwm_fitted, label='fit')
    ax[2].legend()
    # ax[1].grid(True)

    ax[3].scatter(force[:,0], pwm_normalized, label='M1')
    ax[3].scatter(force[:,1], pwm_normalized, label='M2')
    ax[3].scatter(force[:,2], pwm_normalized, label='M3')
    ax[3].scatter(force[:,3], pwm_normalized, label='M4')
    ax[3].legend()
    ax[3].grid(True)


    plt.show()
