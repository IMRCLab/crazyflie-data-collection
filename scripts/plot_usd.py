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

    # start_time = np.inf
    # for _,v in data_usd.items():
    #     start_time = min(start_time, v['timestamp'][0])

    start_time = 0

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3
    time_eP = (data_usd['estPose']['timestamp'] - start_time) / 1e3

    dt_eP = np.diff(time_eP)
    print("Mocap rate: {:.1f} Hz ({:.1f})".format(np.mean(1/dt_eP), np.std(1/dt_eP)))
    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # ax[0,0].hist(1/dt_eP)
    # plt.show()

    T = len(data_usd['fixedFrequency']['timestamp'])

    pos = np.array([
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z']]).T

    q = np.array([
        data_usd['fixedFrequency']['stateEstimate.qw'],
        data_usd['fixedFrequency']['stateEstimate.qx'],
        data_usd['fixedFrequency']['stateEstimate.qy'],
        data_usd['fixedFrequency']['stateEstimate.qz']]).T
    rpy = rowan.to_euler(rowan.normalize(q), "xyz")

    pos_mocap = np.array([
        data_usd['estPose']['locSrv.x'],
        data_usd['estPose']['locSrv.y'],
        data_usd['estPose']['locSrv.z']]).T

    q_mocap = np.array([
        data_usd['estPose']['locSrv.qw'],
        data_usd['estPose']['locSrv.qx'],
        data_usd['estPose']['locSrv.qy'],
        data_usd['estPose']['locSrv.qz']]).T
    rpy_mocap = rowan.to_euler(rowan.normalize(q_mocap), "xyz")

    fig, ax = plt.subplots(2, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(time_fF, pos[:,k], label='state estimate')
        ax[0,k].plot(time_eP, pos_mocap[:,k], '.', label='mocap')
        ax[0,k].set_ylabel(f"position {axis}[m]")

    for k, axis in enumerate(["x", "y", "z"]):
        ax[1,k].plot(time_fF, np.degrees(rpy[:,k]), label='state estimate')
        ax[1,k].plot(time_eP, np.degrees(rpy_mocap[:,k]), '.', label='mocap')
        ax[1,k].set_ylabel(f"angle {axis}[deg]")


    ax[0,0].legend()

    # plt.show()

    # motor fun

    rpm = np.array([
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4']]).T
    
    pwm = np.array([
        data_usd['fixedFrequency']['pwm.m1_pwm'],
        data_usd['fixedFrequency']['pwm.m2_pwm'],
        data_usd['fixedFrequency']['pwm.m3_pwm'],
        data_usd['fixedFrequency']['pwm.m4_pwm']]).T
    
    kw = 4.310657321921365e-08
    force_in_grams = kw * rpm**2
    force_in_grams_from_pwm = -5.360718677769569 + pwm * 0.0005492858445116151


    fig, ax = plt.subplots(3, 2, sharex='all')
    # ax[0,1].plot(time_fF, rpm[:,0])
    # ax[0,1].set_ylabel(f"M1 [rpm]")
    # ax[1,1].plot(time_fF, rpm[:,1])
    # ax[1,1].set_ylabel(f"M2 [rpm]")
    # ax[1,0].plot(time_fF, rpm[:,2])
    # ax[1,0].set_ylabel(f"M3 [rpm]")
    # ax[0,0].plot(time_fF, rpm[:,3])
    # ax[0,0].set_ylabel(f"M4 [rpm]")

    ax[0,1].plot(time_fF, force_in_grams[:,0], label="rpm")
    ax[0,1].plot(time_fF, force_in_grams_from_pwm[:,0], label="pwm")
    ax[0,1].set_ylabel(f"M1 [grams]")
    ax[1,1].plot(time_fF, force_in_grams[:,1], label="rpm")
    ax[1,1].plot(time_fF, force_in_grams_from_pwm[:,1], label="pwm")
    ax[1,1].set_ylabel(f"M2 [grams]")
    ax[1,0].plot(time_fF, force_in_grams[:,2], label="rpm")
    ax[1,0].plot(time_fF, force_in_grams_from_pwm[:,2], label="pwm")
    ax[1,0].set_ylabel(f"M3 [grams]")
    ax[0,0].plot(time_fF, force_in_grams[:,3], label="rpm")
    ax[0,0].plot(time_fF, force_in_grams_from_pwm[:,3], label="pwm")
    ax[0,0].set_ylabel(f"M4 [grams]")

    ax[0,0].legend()

    ax[2,0].plot(time_fF, np.sum(force_in_grams, axis=1))
    ax[2,0].set_ylabel(f"total thrust [grams]")

    # estimating motor delay
    for m in range(4):
        best_k = None
        best_err = 1e6
        for k in range(-50, 50):
            err = np.sum(np.abs(force_in_grams[100:-100,m] - force_in_grams_from_pwm[100+k:-100+k,m]))
            if err < best_err:
                best_err = err
                best_k = k
        print("motor delay [ms]", (time_fF[100+best_k] - time_fF[100])*1000)

    # plt.show()

    # f_a fun

    acc_body = np.array([
        data_usd['fixedFrequency']['acc.x'],
        data_usd['fixedFrequency']['acc.y'],
        data_usd['fixedFrequency']['acc.z']]).T * 9.81
    
    acc_world = rowan.rotate(q, acc_body)

    # m*a = m*g + R f_u + f_a
    # f_a = m*(a-g) - R f_u
    mass = 0.038 # kg

    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
        ])
    force = force_in_grams / 1000 * 9.81
    eta = np.empty((force.shape[0], 4))
    f_u = np.empty((force.shape[0], 3))
    tau_u = np.empty((force.shape[0], 3))

    force2 = force_in_grams_from_pwm / 1000 * 9.81
    eta2 = np.empty((force2.shape[0], 4))
    f_u2 = np.empty((force2.shape[0], 3))
    tau_u2 = np.empty((force2.shape[0], 3))

    for k in range(force.shape[0]):
        eta[k] = np.dot(B0, force[k])
        f_u[k] = np.array([0, 0, eta[k,0]])
        tau_u[k] = np.array([eta[k,1], eta[k,2], eta[k,3]])

        eta2[k] = np.dot(B0, force2[k])
        f_u2[k] = np.array([0, 0, eta2[k,0]])
        tau_u2[k] = np.array([eta2[k,1], eta2[k,2], eta2[k,3]])

    f_a = mass * acc_world - rowan.rotate(q, f_u)
    f_a2 = mass * acc_world - rowan.rotate(q, f_u2)
    
    fig, ax = plt.subplots(3, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(time_fF, acc_body[:,k])
        ax[0,k].set_ylabel(f"acc body {axis}[m/s^2]")

    for k, axis in enumerate(["x", "y", "z"]):
        ax[1,k].plot(time_fF, acc_world[:,k])
        ax[1,k].set_ylabel(f"acc world {axis}[m/s^2]")

    for k, axis in enumerate(["x", "y", "z"]):
        ax[2,k].plot(time_fF, f_a[:,k], label="rpm")
        ax[2,k].plot(time_fF, f_a2[:,k], label="pwm")
        ax[2,k].set_ylabel(f"f_a {axis}[N]")

    ax[2,0].legend()

    plt.show()



