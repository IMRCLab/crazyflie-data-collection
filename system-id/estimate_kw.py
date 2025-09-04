import argparse

import matplotlib.pyplot as plt
import numpy as np
import rowan
import cvxpy as cp

import cfusdlog

def apply_motor_filter(action_scaled: np.ndarray,
                       prev_filtered_rpm_proxy: np.ndarray,
                       change: np.ndarray):
    dt = 0.004
    tau = 0.08 #T/4
    alpha = dt / tau  

    next_filtered_rpm_proxy = prev_filtered_rpm_proxy + alpha * (np.sqrt(action_scaled) - prev_filtered_rpm_proxy)
    filtered_thrust = np.square(next_filtered_rpm_proxy)
    return filtered_thrust, next_filtered_rpm_proxy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])


    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    T = len(data_usd['fixedFrequency']['timestamp'])

    # motor fun

    rpm = np.array([
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4']]).T
    
    pwm = np.array([
        data_usd['fixedFrequency']['motor.m1'],
        data_usd['fixedFrequency']['motor.m2'],
        data_usd['fixedFrequency']['motor.m3'],
        data_usd['fixedFrequency']['motor.m4'],
    ]).T
    vbat = np.array(data_usd['fixedFrequency']['pm.vbatMV']) / 1000.0
    vbat_normalized = vbat / 4.2

    mass = 36.5 # g
    expected_thrust_per_rotor_grams = mass / 4
    expected_thrust_per_rotor_newtons = expected_thrust_per_rotor_grams / 1000.0 * 9.81

    # fit kw's
    kws = np.zeros(4)
    for i in range(4):
        kw = cp.Variable()
        cost = cp.sum_squares(expected_thrust_per_rotor_newtons - kw * rpm[:,i]**2)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()
        kws[i] = kw.value
    print(kws)

    print(T)
    print(np.repeat([kws], T, axis=0))
    force_from_rpm = np.repeat([kws], T, axis=0) * rpm**2

    # rpm2pwmA = -0.019336058570985235
    # rpm2pwmB = 1.5712887646269243e-05
    # rpm2pwmC = 2.159053842993265e-09
    # rpm2pwmD = -1.053665583838794e-09

    # rpm2pwmA = -0.01823085705154721
    # rpm2pwmB = 1.3867670990790517e-05
    # rpm2pwmC = 1.4798165098179836e-09
    # rpm2pwmD = None

    rpm2pwmA = 2.053560678849524
    rpm2pwmB = 7.633270898851121e-05
    rpm2pwmC = -1.7034800944419638e-09
    rpm2pwmD = -2.720592906036923

    # print(np.repeat([vbat_normalized], 4, axis=0).T)
    # exit()

    pwm_fitted = rpm2pwmA + rpm2pwmB * rpm + rpm2pwmC * rpm**2 + rpm2pwmD * np.repeat([vbat_normalized], 4, axis=0).T # + rpm2pwmD * np.multiply(rpm**2, np.repeat([vbat_normalized], 4, axis=0).T)
    pwm_normalized = pwm / 65535.0
    
    print(pwm_fitted)
    print(np.mean(rpm,axis=0))
    print(np.mean(pwm,axis=0))
    # exit()

    fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
    ax[0,1].plot(pwm_fitted[:,0], label="PWM (fit)")
    ax[0,1].plot(pwm_normalized[:,0], label="PWM (measured)")

    # ax[0,1].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[1,1].plot(pwm_fitted[:,1], label="PWM (fit)")
    ax[1,1].plot(pwm_normalized[:,1], label="PWM (measured)")

    # ax[1,1].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[1,0].plot(pwm_fitted[:,2], label="PWM (fit)")
    ax[1,0].plot(pwm_normalized[:,2], label="PWM (measured)")

    # ax[1,0].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[0,0].plot(pwm_fitted[:,3], label="PWM (fit)")
    ax[0,0].plot(pwm_normalized[:,3], label="PWM (measured)")
    # ax[0,0].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    plt.show()

    # exit()




    fig, ax = plt.subplots(2,2, sharex='all', sharey='all')
    ax[0,1].plot(force_from_rpm[:,0], label="force (from rpm)")
    ax[0,1].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[1,1].plot(force_from_rpm[:,1], label="force (from rpm)")
    ax[1,1].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[1,0].plot(force_from_rpm[:,2], label="force (from rpm)")
    ax[1,0].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    ax[0,0].plot(force_from_rpm[:,3], label="force (from rpm)")
    ax[0,0].axhline(y = expected_thrust_per_rotor_newtons, color = 'b')

    plt.show()