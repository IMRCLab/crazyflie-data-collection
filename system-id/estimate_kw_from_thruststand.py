import argparse

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='+')
    args = parser.parse_args()

    # Assuming pwm,vbat[V],rpm1,rpm2,rpm3,rpm4,thrust[g] format

    data = None
    for f in args.filename:
        data_f = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
        if data is None:
            data = data_f
        else:
            data = np.vstack((data, data_f))

    assert(data.shape[1] == 7)

    pwm = data[:,0]
    vbat = data[:,1] # V
    omega_rpm = data[:,2:6]
    omega_rad_per_sec = omega_rpm * 2 * np.pi / 60
    thrust_total_grams = data[:,6] # grams total
    thrust_per_rotor_newtons = thrust_total_grams / 4 / 1000.0 * 9.81

    # fit kw's
    kws = np.zeros(4)
    for i in range(4):
        kw = cp.Variable()
        cost = cp.sum_squares(thrust_per_rotor_newtons - kw * omega_rad_per_sec[:,i]**2)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()
        kws[i] = kw.value
    print("K_F", kws)
    K_F = np.mean(kws)
    print("mean K_F", K_F)

    
    T = data.shape[0]
    # print(T)
    # print(np.repeat([kws], T, axis=0))
    # force_from_rpm = np.repeat([kws], T, axis=0) * omega_rad_per_sec**2
    force_from_rpm = K_F * omega_rad_per_sec**2

    # thrust mixing model from 
    # https://www.bitcraze.io/2025/10/keeping-thrust-consistent-as-the-battery-drains/

    vmotor = vbat * pwm / 65535.0

    # # default values from the firmware
    # VMOTOR2THRUST0 = -0.02476537915958403
    # VMOTOR2THRUST1 = 0.06523793527519485
    # VMOTOR2THRUST2 = -0.026792504967750107
    # VMOTOR2THRUST3 = 0.006776789303971145

    # optimize own values
    p = np.polyfit(vmotor, thrust_per_rotor_newtons, deg=3)
    VMOTOR2THRUST0 = p[3]
    VMOTOR2THRUST1 = p[2]
    VMOTOR2THRUST2 = p[1]
    VMOTOR2THRUST3 = p[0]

    print("VMOTOR2THRUST0 = ", VMOTOR2THRUST0)
    print("VMOTOR2THRUST1 = ", VMOTOR2THRUST1)
    print("VMOTOR2THRUST2 = ", VMOTOR2THRUST2)
    print("VMOTOR2THRUST3 = ", VMOTOR2THRUST3)

    thrust_per_rotor_newton_model = VMOTOR2THRUST0 \
        + VMOTOR2THRUST1 * vmotor \
        + VMOTOR2THRUST2 * vmotor * vmotor \
        + VMOTOR2THRUST3 * vmotor * vmotor * vmotor

    fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
    ax[0,1].plot(force_from_rpm[:,0], label="force (from rpm)")
    ax[0,1].plot(thrust_per_rotor_newtons, label="measured")
    ax[0,1].plot(thrust_per_rotor_newton_model, label="model")

    ax[1,1].plot(force_from_rpm[:,1], label="force (from rpm)")
    ax[1,1].plot(thrust_per_rotor_newtons, label="measured")
    ax[1,1].plot(thrust_per_rotor_newton_model, label="model")

    ax[1,0].plot(force_from_rpm[:,2], label="force (from rpm)")
    ax[1,0].plot(thrust_per_rotor_newtons, label="measured")
    ax[1,0].plot(thrust_per_rotor_newton_model, label="model")

    ax[0,0].plot(force_from_rpm[:,3], label="force (from rpm)")
    ax[0,0].plot(thrust_per_rotor_newtons, label="measured")
    ax[0,0].plot(thrust_per_rotor_newton_model, label="model")

    plt.show()
