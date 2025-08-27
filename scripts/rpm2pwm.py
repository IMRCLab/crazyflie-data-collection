import argparse

import matplotlib.pyplot as plt
import numpy as np
import rowan

import cfusdlog

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    end_time = time_fF[-1]

    T = len(data_usd['fixedFrequency']['timestamp'])

    rpm = np.array([
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4'],
    ]).T

    pwm = np.array([
        data_usd['fixedFrequency']['motor.m1'],
        data_usd['fixedFrequency']['motor.m2'],
        data_usd['fixedFrequency']['motor.m3'],
        data_usd['fixedFrequency']['motor.m4'],
    ]).T
    vbat = np.array(data_usd['fixedFrequency']['pm.vbatMV']) / 1000.0

    # fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
    # ax[0,1].plot(time_fF, pwm[:,0])
    # ax[1,1].plot(time_fF, pwm[:,1])
    # ax[1,0].plot(time_fF, pwm[:,2])
    # ax[0,0].plot(time_fF, pwm[:,3])

    # plt.show()

    fig, ax = plt.subplots(1,1)
    for i in range(4):
        plt.scatter(pwm[:,i], rpm[:,i])


    plt.show()

    import cvxpy as cp

    pwm = pwm.flatten()
    rpm = rpm.flatten() / 30000
    pwm_normalized = pwm / 65565.0

    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()
    # pwm = a + b * rpm + c * rpm^2
    cost = cp.sum_squares(a + b * rpm + c * rpm**2 - pwm_normalized)
    prob = cp.Problem(cp.Minimize(cost), [])
    prob.solve()
    print("rpm2pwmA: {}".format(a.value))
    print("rpm2pwmB: {}".format(b.value))
    print("rpm2pwmC: {}".format(c.value))

    pwm_fitted = a.value + b.value * rpm + c.value * rpm**2

    # force -> pwm
    # pwm_normalized = sqrtf(force / kappa_f) * b + a

    fig, ax = plt.subplots(2)
    ax[0].plot(pwm_normalized)
    ax[0].plot(pwm_fitted)

    ax[1].scatter(rpm, pwm_normalized, label='data')
    ax[1].scatter(rpm, pwm_fitted, label='fit')

    ax[1].legend()
    ax[1].grid(True)

    plt.show()

    exit()

    # kappa_f = 2.6835255e-10

    # estimate kappa_f's
    # expect: 35.5g => per rotor: 8.875g

    import cvxpy as cp

    # only consider last 3 seconds (hover)
    start_idx = np.argwhere(time_fF >= end_time - 3)[0][0]
    kappa_f = []
    for i in range(4):
        kw = cp.Variable()
        cost = cp.sum_squares(8.875 * 9.81 / 1000 - kw * rpm[start_idx:-1,i]**2)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()
        print("kf{}: {}".format(i+1, kw.value))
        kappa_f.append(kw.value)
    kappa_f = np.array(kappa_f)

    # exit()

    # kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])

    force = kappa_f * rpm**2
    force_des = np.array([
        data_usd['fixedFrequency']['powerDist.m1d'],
        data_usd['fixedFrequency']['powerDist.m2d'],
        data_usd['fixedFrequency']['powerDist.m3d'],
        data_usd['fixedFrequency']['powerDist.m4d'],
    ]).T


    # pwm = a + b * rpm + c * rpm^2
    rpm2pwmA = -0.006958373447616477
    rpm2pwmB = 1.933811561926461e-05
    rpm2pwmC = 1.0376220271145036e-09
    # rpm_des_from_pwm = (-b +/- sqrt(b^2 - 4ac))2/c
    pwm_normalized = pwm / 65536.0
    rpm_des_from_pwm = (rpm2pwmB - np.sqrt(rpm2pwmB**2 - 4*rpm2pwmC*(rpm2pwmA-pwm_normalized)))/(2*rpm2pwmC)
    force_des_from_pwm = kappa_f * rpm_des_from_pwm**2

    # plot rpm -> forces and pwm -> forces 
    fig, ax = plt.subplots(4,1, sharex='all')
    for k, axis in enumerate(["1", "2", "3", "4"]):
        ax[k].grid()
        ax[k].plot(time_fF, force[:,k], label="RPM")
        ax[k].set_ylabel(f"f{axis} [N]")

        ax[k].plot(time_fF, force_des[:,k], label="desired by ctrl")
        ax[k].plot(time_fF, force_des_from_pwm[:,k], label="PWM")
    ax[0].legend() 

