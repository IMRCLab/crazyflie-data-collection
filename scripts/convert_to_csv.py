import argparse

import numpy as np
import rowan

import cfusdlog

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    parser.add_argument("file_csv")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    result = np.array([
        time_fF,
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z'],
        data_usd['fixedFrequency']['stateEstimate.vx'],
        data_usd['fixedFrequency']['stateEstimate.vy'],
        data_usd['fixedFrequency']['stateEstimate.vz'],
        data_usd['fixedFrequency']['stateEstimate.qw'],
        data_usd['fixedFrequency']['stateEstimate.qx'],
        data_usd['fixedFrequency']['stateEstimate.qy'],
        data_usd['fixedFrequency']['stateEstimate.qz'],
        np.deg2rad(data_usd['fixedFrequency']['gyro.x']),
        np.deg2rad(data_usd['fixedFrequency']['gyro.y']),
        np.deg2rad(data_usd['fixedFrequency']['gyro.z']),
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4']]).T
    
    np.savetxt(args.file_csv, result, delimiter = ",", header="time[s],x[m],y[m],z[m],vx[m/s],vy[m/s],vz[m/s],qw,qx,qy,qz,wx[rad/s],wy[rad/s],wz[rad/s],rpm1,rpm2,rpm3,rpm4")

