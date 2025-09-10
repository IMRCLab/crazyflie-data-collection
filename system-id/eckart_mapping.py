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

    assert(data.shape[1] == 7)

    pwm = data[:,0]         # PWM value
    vbat = data[:,1]        # V, battery voltage, 
    rpm = data[:,2:6]
    thrustGram = data[:,6] / 4.0  # g, per motor
    force = np.repeat(kws, data.shape[0], axis=0) * rpm**2
    rpm_mean = np.mean(rpm, axis=1)

    pwm_normalized = pwm / 65535.0
    vbat_normalized = vbat / 4.2

    # print(np.multiply(rpm_mean, vbat_normalized).shape)
    # exit()

    # float pwm = d00 + d10 * thrustGram + d01 * v + d20 * thrustGram * thrustGram + d11 * thrustGram * v;

    d00 = cp.Variable()
    d10 = cp.Variable()
    d01 = cp.Variable()
    d11 = cp.Variable()
    d20 = cp.Variable()

    # pwm = a + b * rpm + c * rpm^2 + d * rpm * vbat_normalized
    # cost = cp.sum_squares(a + b * rpm_mean + c * rpm_mean**2 + d * cp.multiply(rpm_mean**2, vbat_normalized) - pwm_normalized)
    cost = cp.sum_squares((d00 + d10 * thrustGram + d01 * vbat_normalized + d20 * thrustGram**2 + d11 * cp.multiply(thrustGram, vbat_normalized)) - pwm_normalized)
    prob = cp.Problem(cp.Minimize(cost), [])
    prob.solve()
    print("d00 = {};".format(d00.value))
    print("d10 = {};".format(d10.value))
    print("d01 = {};".format(d01.value))
    print("d11 = {};".format(d11.value))
    print("d20 = {};".format(d20.value))


    # pwm_fitted = a.value + b.value * rpm_mean + c.value * rpm_mean**2 + d.value * np.multiply(rpm_mean**2, vbat_normalized)
    #pwm_fitted = a.value + b.value * rpm_mean + c.value * rpm_mean**2# + d.value * np.multiply(rpm_mean**2, vbat_normalized)
    pwm_fitted = d00.value + d10.value * thrustGram + d01.value * vbat_normalized + d20.value * thrustGram**2 + d11.value * np.multiply(thrustGram, vbat_normalized) # + d.value * np.multiply(rpm_mean**2, vbat_normalized)

    d00 = 0.5543364748044269
    d10 = 0.11442589787133063
    d01 = -0.5067031467944692
    d20 = -0.002283966554392003
    d11 = -0.03255320005438393
    pwm_fitted_old = d00 + d10 * thrustGram + d01 * vbat_normalized + d20 * thrustGram**2 + d11 * np.multiply(thrustGram, vbat_normalized)

    # force -> pwm
    # pwm_normalized = sqrtf(force / kappa_f) * b + a

    fig, ax = plt.subplots(4)
    ax[0].plot(pwm_normalized, label="setpoint")
    ax[0].plot(pwm_fitted, label="fitted")
    ax[0].plot(pwm_fitted_old, label="fitted old")
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
