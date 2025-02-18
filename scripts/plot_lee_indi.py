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

    print(end_time)
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

    fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
    ax[0,1].plot(time_fF, rpm[:,0])
    ax[1,1].plot(time_fF, rpm[:,1])
    ax[1,0].plot(time_fF, rpm[:,2])
    ax[0,0].plot(time_fF, rpm[:,3])

    # plt.show()

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
    

    # # force -> pwm mapping
    # pwm_normalized = pwm / 65535.0
    # pwmToThrustA = []
    # pwmToThrustB = []
    # for i in range(4):
    #     a = cp.Variable()
    #     b = cp.Variable()
    #     # a * pwm^2 + b * pwm
    #     cost = cp.sum_squares(a * pwm_normalized[:,i] **2 + b * pwm_normalized[:,i] - force[:,i])
    #     prob = cp.Problem(cp.Minimize(cost), [])
    #     prob.solve()
    #     print("pwmToThrustA{}: {}".format(i+1, a.value))
    #     print("pwmToThrustB{}: {}".format(i+1, b.value))
    #     pwmToThrustA.append(a.value)
    #     pwmToThrustB.append(b.value)
    # pwmToThrustA = np.array(pwmToThrustA)
    # pwmToThrustB = np.array(pwmToThrustB)

    # print(pwmToThrustA, pwmToThrustB)


    # fig, ax = plt.subplots(1, 1, sharex='all', sharey='all', squeeze=False)
    # # ax[0,0].scatter(force[start_idx:-1,0], pwm_normalized[start_idx:-1,0])
    # # ax[0,0].scatter(force[start_idx:-1,1], pwm_normalized[start_idx:-1,1])
    # # ax[0,0].scatter(force[start_idx:-1,2], pwm_normalized[start_idx:-1,2])
    # ax[0,0].scatter(force[:,3], pwm_normalized[:,3])
    # ax[0,0].scatter(pwmToThrustA[3] * pwm_normalized[:,3]**2 + pwmToThrustB[3] * pwm_normalized[:,3], pwm_normalized[:,3])
    # plt.show()
    # exit()

#    # force -> pwm mapping
#     pwm_normalized = pwm / 65535.0
#     f2pA = []
#     f2pB = []
#     for i in range(4):
#         a = cp.Variable()
#         b = cp.Variable()
#         # pwm = a + b * force
#         cost = cp.sum_squares(a + b * force[:,i] - pwm_normalized[:,i])
#         prob = cp.Problem(cp.Minimize(cost), [])
#         prob.solve()
#         print("f2pA{}: {}".format(i+1, a.value))
#         print("f2pB{}: {}".format(i+1, b.value))
#         f2pA.append(a.value)
#         f2pB.append(b.value)
#     f2pA = np.array(f2pA)
#     f2pB = np.array(f2pB)

#     fig, ax = plt.subplots(1, 1, sharex='all', sharey='all', squeeze=False)
#     # ax[0,0].scatter(force[start_idx:-1,0], pwm_normalized[start_idx:-1,0])
#     # ax[0,0].scatter(force[start_idx:-1,1], pwm_normalized[start_idx:-1,1])
#     # ax[0,0].scatter(force[start_idx:-1,2], pwm_normalized[start_idx:-1,2])
#     for i in range(4):
#         ax[0,0].scatter(force[:,i], pwm_normalized[:,i])
#         ax[0,0].scatter(force[:,i], f2pA[i] + f2pB[i] * force[:,i])
    # plt.show()
    # exit()

    # fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
    # ax[0,1].plot(time_fF, force[:,0])
    # ax[1,1].plot(time_fF, force[:,1])
    # ax[1,0].plot(time_fF, force[:,2])
    # ax[0,0].plot(time_fF, force[:,3])
    # plt.show()

    force_sum = np.sum(force, axis=1)
    force_sum_per_prop = np.mean(force, axis=0) / 9.81 * 1000 #f=m*g
    print(force_sum_per_prop)

    fig, ax = plt.subplots(1, 1, sharex='all', sharey='all', squeeze=False)
    ax[0,0].plot(time_fF, force[:,0], label="M1")
    ax[0,0].plot(time_fF, force[:,1], label="M2")
    ax[0,0].plot(time_fF, force[:,2], label="M3")
    ax[0,0].plot(time_fF, force[:,3], label="M4")
    ax[0,0].legend()

    # plt.show()
    # exit()



    # exit()


    # ctrl lee part
    ctype = "ctrlLeeP"

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
    error = np.linalg.norm(pos - pos_d, axis=1)
    error_xy = np.linalg.norm(pos[:,0:2] - pos_d[:,0:2], axis=1)

    start_idx = np.argwhere(time_fF >= 3)[0][0]
    end_idx = np.argwhere(time_fF >= end_time - 3)[0][0]
    print("error", np.mean(error[start_idx:end_idx]))
    print("error_xy", np.mean(error_xy[start_idx:end_idx]))

    fig, ax = plt.subplots(4, 3, sharex='all')
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

    if ctype == "ctrlLeeP":

        ppos = np.array([
        data_usd['fixedFrequency']['stateEstimateZ.px'],
        data_usd['fixedFrequency']['stateEstimateZ.py'],
        data_usd['fixedFrequency']['stateEstimateZ.pz']]).T/1000
        
        est_acc = np.array([
            data_usd['fixedFrequency']['ctrlLeeP.plAccx'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccy'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccz']]).T
        

        des_acc = np.array([
            data_usd['fixedFrequency']['ctrlLeeP.plAccx_des'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccy_des'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccz_des']]).T
        des_acc_tq = np.array([
            data_usd['fixedFrequency']['ctrlLeeP.plAccx_tq'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccy_tq'],
            data_usd['fixedFrequency']['ctrlLeeP.plAccz_tq']]).T 


        pvel_filtered = np.array([
        data_usd['fixedFrequency']['ctrlLeeP.plVelx'],
        data_usd['fixedFrequency']['ctrlLeeP.plVely'],
        data_usd['fixedFrequency']['ctrlLeeP.plVelz']]).T
    
    
        fig, ax = plt.subplots(3, 1, sharex='all')
        for k, axis in enumerate(["x", "y", "z"]):
            ax[k].plot(time_fF, des_acc_tq[:,k], label="tq")
            ax[k].plot(time_fF, des_acc[:,k], label="des")
            ax[k].set_ylabel(f"pacc {axis}[m/s^2]")
        ax[0].legend()

        fig, ax = plt.subplots(3, 1, sharex='all')
        for k, axis in enumerate(["x", "y", "z"]):
            ax[k].plot(time_fF, ppos[:,k], label="pos payload")
            ax[k].set_ylabel(f"ppos {axis}[m]")
        ax[0].legend()

        fig, ax = plt.subplots(3, 1, sharex='all')
        for k, axis in enumerate(["x", "y", "z"]):
            ax[k].plot(time_fF, pvel_filtered[:,k])
            ax[k].set_ylabel(f"pvel {axis}[m/s]") 

        error = np.linalg.norm(ppos - pos_d, axis=1)
        error_xy = np.linalg.norm(ppos[:,0:2] - pos_d[:,0:2], axis=1)

        start_idx = np.argwhere(time_fF >= 3)[0][0]
        end_idx = np.argwhere(time_fF >= end_time - 3)[0][0]
        print("error (payload)", np.mean(error[start_idx:end_idx]))
        print("error_xy (pyaload)", np.mean(error_xy[start_idx:end_idx]))   
        
    # position INDI part
    a_rpm = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_rpmx'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpmy'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpmz']]).T
    
    a_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fx'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fy'],
        data_usd['fixedFrequency'][f'{ctype}.a_rpm_fz']]).T
    
    a_imu = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_imux'],
        data_usd['fixedFrequency'][f'{ctype}.a_imuy'],
        data_usd['fixedFrequency'][f'{ctype}.a_imuz']]).T
    
    a_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fx'],
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fy'],
        data_usd['fixedFrequency'][f'{ctype}.a_imu_fz']]).T

    fig, ax = plt.subplots(2, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        # ax[0,k].plot(time_fF, a_rpm[:,k], label='from RPM')
        ax[0,k].plot(time_fF, a_rpm_filtered[:,k], label='from RPM (filtered)')
        # ax[0,k].plot(time_fF, a_imu[:,k], label='from IMU')
        ax[0,k].plot(time_fF, a_imu_filtered[:,k], label='from IMU (filtered)')
        ax[0,k].set_ylabel(f"acc {axis}[m/s2]")
    ax[0,0].legend() 

    # attitude indi part

    tau_rpm = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_rpmx'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpmy'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpmz']]).T
    
    tau_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fx'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fy'],
        data_usd['fixedFrequency'][f'{ctype}.tau_rpm_fz']]).T
    
    tau_imu = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_x'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_y'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_z']]).T
    
    tau_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fx'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fy'],
        data_usd['fixedFrequency'][f'{ctype}.tau_gyro_fz']]).T    

    for k, axis in enumerate(["x", "y", "z"]):
        # ax[1,k].plot(time_fF, tau_rpm[:,k], label='from RPM')
        ax[1,k].plot(time_fF, tau_rpm_filtered[:,k], label='from RPM (filtered)')
        # ax[1,k].plot(time_fF, tau_imu[:,k], label='from IMU')
        ax[1,k].plot(time_fF, tau_imu_filtered[:,k], label='from IMU (filtered)')
        ax[1,k].set_ylabel(f"torque {axis}[Nm]")
    ax[1,0].legend()

    plt.show()



