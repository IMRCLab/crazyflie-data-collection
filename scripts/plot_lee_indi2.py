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

    t = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    # ctrl lee part
    ctype = "ctrlLee"

    pos = np.array([
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z']]).T
    
    pos_d = np.array([
        data_usd['fixedFrequency']['ctrltargetZ.x'],
        data_usd['fixedFrequency']['ctrltargetZ.y'],
        data_usd['fixedFrequency']['ctrltargetZ.z']]).T / 1000.0
    
    vel = np.array([
        data_usd['fixedFrequency']['stateEstimateZ.vx'],
        data_usd['fixedFrequency']['stateEstimateZ.vy'],
        data_usd['fixedFrequency']['stateEstimateZ.vz']]).T / 1000.0
    
    vel_d = np.array([
        data_usd['fixedFrequency']['ctrltargetZ.vx'],
        data_usd['fixedFrequency']['ctrltargetZ.vy'],
        data_usd['fixedFrequency']['ctrltargetZ.vz']]).T / 1000.0
    
    rpy = np.array([
        data_usd['fixedFrequency'][f'{ctype}.rpyx'],
        data_usd['fixedFrequency'][f'{ctype}.rpyy'],
        data_usd['fixedFrequency'][f'{ctype}.rpyz']]).T
    
    rpy_d = np.array([
        data_usd['fixedFrequency'][f'{ctype}.rpydx'],
        data_usd['fixedFrequency'][f'{ctype}.rpydy'],
        data_usd['fixedFrequency'][f'{ctype}.rpydz']]).T
    
    omega = np.array([
        data_usd['fixedFrequency'][f'{ctype}.omegax'],
        data_usd['fixedFrequency'][f'{ctype}.omegay'],
        data_usd['fixedFrequency'][f'{ctype}.omegaz']]).T
    
    omega_d = np.array([
        data_usd['fixedFrequency'][f'{ctype}.omegarx'],
        data_usd['fixedFrequency'][f'{ctype}.omegary'],
        data_usd['fixedFrequency'][f'{ctype}.omegarz']]).T
    
    # omega_des_dot = np.array([
    #     data_usd['fixedFrequency'][f'{ctype}.omegaddx'],
    #     data_usd['fixedFrequency'][f'{ctype}.omegaddy'],
    #     data_usd['fixedFrequency'][f'{ctype}.omegaddz']]).T
    
    # fig, ax = plt.subplots(5, 3, sharex='all')
    # error = np.linalg.norm(pos - pos_d, axis=1)
    # error_xy = np.linalg.norm(pos[:,0:2] - pos_d[:,0:2], axis=1)

    # start_idx = np.argwhere(time_fF >= 3)[0][0]
    # end_idx = np.argwhere(time_fF >= end_time - 3)[0][0]
    # print("error", np.mean(error[start_idx:end_idx]))
    # print("error_xy", np.mean(error_xy[start_idx:end_idx]))

    fig, ax = plt.subplots(4, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, pos[:,k])
        ax[0,k].plot(t, pos_d[:,k])
        ax[0,k].set_ylabel(f"pos {axis}[m]")

        ax[1,k].plot(t, vel[:,k])
        ax[1,k].plot(t, vel_d[:,k])
        ax[1,k].set_ylabel(f"vel {axis}[m/s]")

        ax[2,k].plot(t, np.degrees(rpy[:,k]))
        ax[2,k].plot(t, np.degrees(rpy_d[:,k]))
        ax[2,k].set_ylabel(f"rot {axis} [deg]")

        ax[3,k].plot(t, np.degrees(omega[:,k]))
        ax[3,k].plot(t, np.degrees(omega_d[:,k]))
        ax[3,k].set_ylabel(f"ang vel {axis}[deg/s]")


        
    # position INDI part
   
    a_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fx'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fy'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fz']]).T
    
    a_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fx'],
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fy'],
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fz']]).T

    fig, ax = plt.subplots(2, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        # ax[0,k].plot(time_fF, a_rpm[:,k], label='from RPM')
        ax[0,k].plot(t, a_rpm_filtered[:,k], label='from RPM (filtered)')
        # ax[0,k].plot(t, a_imu[:,k], label='from IMU')
        ax[0,k].plot(t, a_imu_filtered[:,k], label='from IMU (filtered)')
        ax[0,k].set_ylabel(f"acc {axis}[m/s2]")
    ax[0,0].legend() 

    # attitude indi part  
    tau_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fx'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fy'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fz']]).T
    
    tau_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fx'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fy'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fz']]).T    

    for k, axis in enumerate(["x", "y", "z"]):
        # ax[1,k].plot(t, tau_rpm[:,k], label='from RPM')
        ax[1,k].plot(t, tau_rpm_filtered[:,k], label='from RPM (filtered)')
        # ax[1,k].plot(t, tau_imu[:,k], label='from IMU')
        ax[1,k].plot(t, tau_imu_filtered[:,k], label='from IMU (filtered)')
        ax[1,k].set_ylabel(f"torque {axis}[Nm]")
    ax[1,0].legend()

    plt.show()



