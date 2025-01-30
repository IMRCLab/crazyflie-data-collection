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

    # start_time = 0

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3
    # time_eP = (data_usd['estPose']['timestamp'] - start_time) / 1e3

    end_time = time_fF[-1]
    # print(end_time)
    # print(np.argwhere(time_fF >= end_time - 3))
    # exit()

    # dt_eP = np.diff(time_eP)
    # print("Mocap rate: {:.1f} Hz ({:.1f})".format(np.mean(1/dt_eP), np.std(1/dt_eP)))
    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # ax[0,0].hist(1/dt_eP)
    # plt.show()

    T = len(data_usd['fixedFrequency']['timestamp'])

    pos = np.array([
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z']]).T

    # q = np.array([
    #     data_usd['fixedFrequency']['stateEstimate.qw'],
    #     data_usd['fixedFrequency']['stateEstimate.qx'],
    #     data_usd['fixedFrequency']['stateEstimate.qy'],
    #     data_usd['fixedFrequency']['stateEstimate.qz']]).T
    # rpy = rowan.to_euler(rowan.normalize(q), "xyz")

    # pos_mocap = np.array([
    #     data_usd['estPose']['locSrv.x'],
    #     data_usd['estPose']['locSrv.y'],
    #     data_usd['estPose']['locSrv.z']]).T

    # q_mocap = np.array([
    #     data_usd['estPose']['locSrv.qw'],
    #     data_usd['estPose']['locSrv.qx'],
    #     data_usd['estPose']['locSrv.qy'],
    #     data_usd['estPose']['locSrv.qz']]).T
    # rpy_mocap = rowan.to_euler(rowan.normalize(q_mocap), "xyz")

    # fig, ax = plt.subplots(2, 3, sharex='all')
    # for k, axis in enumerate(["x", "y", "z"]):
    #     ax[0,k].plot(time_fF, pos[:,k], label='state estimate')
    #     ax[0,k].plot(time_eP, pos_mocap[:,k], '.', label='mocap')
    #     ax[0,k].set_ylabel(f"position {axis}[m]")

    # for k, axis in enumerate(["x", "y", "z"]):
    #     ax[1,k].plot(time_fF, np.degrees(rpy[:,k]), label='state estimate')
    #     ax[1,k].plot(time_eP, np.degrees(rpy_mocap[:,k]), '.', label='mocap')
    #     ax[1,k].set_ylabel(f"angle {axis}[deg]")


    # ax[0,0].legend()

    # plt.show()

    # rpm = np.array([
    #     data_usd['fixedFrequency']['rpm.m1'],
    #     data_usd['fixedFrequency']['rpm.m2'],
    #     data_usd['fixedFrequency']['rpm.m3'],
    #     data_usd['fixedFrequency']['rpm.m4'],
    # ]).T

    # pwm = np.array([
    #     data_usd['fixedFrequency']['motor.m1'],
    #     data_usd['fixedFrequency']['motor.m2'],
    #     data_usd['fixedFrequency']['motor.m3'],
    #     data_usd['fixedFrequency']['motor.m4'],
    # ]).T

    # fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
    # ax[0,1].plot(time_fF, rpm[:,0])
    # ax[1,1].plot(time_fF, rpm[:,1])
    # ax[1,0].plot(time_fF, rpm[:,2])
    # ax[0,0].plot(time_fF, rpm[:,3])

    # ctrl lee part

    pos = np.array([
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z']]).T
    
    pos_d = np.array([
        data_usd['fixedFrequency']['ctrltarget.x'],
        data_usd['fixedFrequency']['ctrltarget.y'],
        data_usd['fixedFrequency']['ctrltarget.z']]).T
    
    vel = np.array([
        data_usd['fixedFrequency']['stateEstimate.vx'],
        data_usd['fixedFrequency']['stateEstimate.vy'],
        data_usd['fixedFrequency']['stateEstimate.vz']]).T
    
    vel_d = np.array([
        data_usd['fixedFrequency']['ctrltarget.vx'],
        data_usd['fixedFrequency']['ctrltarget.vy'],
        data_usd['fixedFrequency']['ctrltarget.vz']]).T
    
    rpy = np.array([
        data_usd['fixedFrequency']['ctrlLee.rpyx'],
        data_usd['fixedFrequency']['ctrlLee.rpyy'],
        data_usd['fixedFrequency']['ctrlLee.rpyz']]).T
    
    rpy_d = np.array([
        data_usd['fixedFrequency']['ctrlLee.rpydx'],
        data_usd['fixedFrequency']['ctrlLee.rpydy'],
        data_usd['fixedFrequency']['ctrlLee.rpydz']]).T
    
    omega = np.array([
        data_usd['fixedFrequency']['ctrlLee.omegax'],
        data_usd['fixedFrequency']['ctrlLee.omegay'],
        data_usd['fixedFrequency']['ctrlLee.omegaz']]).T
    
    omega_d = np.array([
        data_usd['fixedFrequency']['ctrlLee.omegarx'],
        data_usd['fixedFrequency']['ctrlLee.omegary'],
        data_usd['fixedFrequency']['ctrlLee.omegarz']]).T
    
    # omega_des_dot = np.array([
    #     data_usd['fixedFrequency']['ctrlLee.omegaddx'],
    #     data_usd['fixedFrequency']['ctrlLee.omegaddy'],
    #     data_usd['fixedFrequency']['ctrlLee.omegaddz']]).T
    
    fig, ax = plt.subplots(5, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(time_fF, pos[:,k])
        ax[0,k].plot(time_fF, pos_d[:,k])
        ax[0,k].set_ylabel(f"pos {axis}[m]")

        ax[1,k].plot(time_fF, vel[:,k])
        ax[1,k].plot(time_fF, vel_d[:,k])
        ax[1,k].set_ylabel(f"vel {axis}[m/s]")

        ax[2,k].plot(time_fF, np.degrees(rpy[:,k]))
        ax[2,k].plot(time_fF, np.degrees(rpy_d[:,k]))
        ax[2,k].set_ylabel(f"rot {axis} [deg]")

        ax[3,k].plot(time_fF, np.degrees(omega[:,k]))
        ax[3,k].plot(time_fF, np.degrees(omega_d[:,k]))
        ax[3,k].set_ylabel(f"ang vel {axis}[deg/s]")

        # # ax[4,k].plot(time_fF, np.degrees(omega[:,k]))
        # ax[4,k].plot(time_fF, np.degrees(omega_des_dot[:,k]))
        # ax[4,k].set_ylabel(f"ang acc {axis}[deg/s^2]")

    plt.show()
