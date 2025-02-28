import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cfusdlog


class SplineFitter:
    def __init__(self):
        pass

    def fit(self, t, y, num_segments = 5, degree = 4):
        a = 0.000001 #50000 # weight for the least square error 
        l = 0.0 #0.1 #0.0000001 # weight for the regularization
        coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
        T = t[-1] - t[0]
        T_segment = T / num_segments
        cost = 0
        constraints = []

        j = 0
        for i in range(num_segments):
            start_id = j
            while j < y.shape[0]:
                t_normalized = (t[j] - i*T_segment)/T_segment
                if t_normalized > 1.0:
                    break
                y_poly = sum([coeffs[i][d]*t_normalized**d for d in range(degree+1)])
                cost += a*cp.sum_squares(y_poly - y[j])
                j = j + 1
            cost += l * cp.sum_squares(coeffs[i])

            if i < num_segments - 1:
                x = 1.0 #data_points[end_id - 1][0]
                # Calculate the derivatives at the boundary
                boundary_value    = sum(coeffs[i][d]*x**d for d in range(degree + 1))
                dboundary_value   = sum(d * coeffs[i][d] * x**(d - 1) for d in range(1, degree + 1))
                ddboundary_value  = sum(d * (d - 1) * coeffs[i][d] * x**(d - 2) for d in range(2, degree + 1))
                dddboundary_value  = sum(d * (d - 2) * (d - 1) * coeffs[i][d] * x**(d - 3) for d in range(3, degree + 1))
                x = 0.0
                boundary_value_next   = sum(coeffs[i + 1][d] * x**d for d in range(degree + 1))
                dboundary_value_next  = sum(d * coeffs[i + 1][d] * x**(d - 1) for d in range(1, degree + 1))
                ddboundary_value_next = sum(d * (d - 1) * coeffs[i + 1][d] * x**(d - 2) for d in range(2, degree + 1))
                # dddboundary_value_next = sum(d * (d - 2) * (d - 1) * coeffs[i + 1][d] * x**(d - 3) for d in range(3, degree + 1))
            # add constraints
                constraints.append(boundary_value == boundary_value_next)
                constraints.append(dboundary_value == dboundary_value_next)
                constraints.append(ddboundary_value == ddboundary_value_next)
                # constraints.append(dddboundary_value == dddboundary_value_next)
            
                if degree > 3:
                    x = 1.0
                    dddboundary_value  = sum(d * (d - 1) * (d - 2) * coeffs[i][d] * x**(d - 3) for d in range(3, degree + 1))
                    x = 0.0
                    dddboundary_value_next = sum(d * (d - 1) * (d - 2) * coeffs[i + 1][d] * x**(d - 3) for d in range(3, degree + 1))
                    constraints.append(dddboundary_value == dddboundary_value_next)

                
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        self.coeffs = coeffs
        self.T_segment = T_segment
        self.num_segments = num_segments
        self.degree = degree

    def eval(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([self.coeffs[i][d].value*t_normalized**d for d in range(self.degree+1)])
            result.append(y_poly)
        return np.array(result)
    
    def evald(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([d * self.coeffs[i][d].value*t_normalized**(d-1)/self.T_segment for d in range(1, self.degree+1)])
            result.append(y_poly)
        return np.array(result)
    
    def evaldd(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([d * (d-1) * self.coeffs[i][d].value*t_normalized**(d-2)/(self.T_segment**2) for d in range(2, self.degree+1)])
            result.append(y_poly)
        return np.array(result)


    # def plot(self, t, y):
    #     # Plotting
    #     fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)

    #     for j, coeff in enumerate(coeffs):
    #         x_val = np.linspace(0, 1, 100)
    #         y_est = [sum([coeff[d].value * x**d  for d in range(degree+1)])  for x in x_val]
    #         y_est_der = [sum([(d)*coeff[d].value * x**(d-1) / T_segment  for d in range(1,degree+1)])  for x in x_val]
    #         ax1.plot(x_val*T_segment + j*T_segment, y_est,  linewidth=2)
    #         ax2.plot(x_val*T_segment + j*T_segment, y_est_der,  linewidth=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])

    t = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    pos = np.array([
        data_usd['fixedFrequency']['stateEstimate.x'],
        data_usd['fixedFrequency']['stateEstimate.y'],
        data_usd['fixedFrequency']['stateEstimate.z']]).T
    
    pos_payload = np.array([
        data_usd['fixedFrequency']['stateEstimateZ.px'],
        data_usd['fixedFrequency']['stateEstimateZ.py'],
        data_usd['fixedFrequency']['stateEstimateZ.pz']]).T / 1000.0
    
    vel = np.array([
        data_usd['fixedFrequency']['stateEstimateZ.vx'],
        data_usd['fixedFrequency']['stateEstimateZ.vy'],
        data_usd['fixedFrequency']['stateEstimateZ.vz']]).T / 1000.0
    
    acc = np.array([
        data_usd['fixedFrequency']['stateEstimateZ.ax'],
        data_usd['fixedFrequency']['stateEstimateZ.ay'],
        data_usd['fixedFrequency']['stateEstimateZ.az'] - 9810]).T / 1000.0
    
    omega = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.omegax'],
        data_usd['fixedFrequency'][f'ctrlLeeP.omegay'],
        data_usd['fixedFrequency'][f'ctrlLeeP.omegaz']]).T
    
   
    sf = SplineFitter()

    acc_spline = []
    fig, ax = plt.subplots(3, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, pos[:,k], label="data")

        sf.fit(t, pos[:,k], 50)
        spline = sf.eval(t)
        ax[0,k].plot(t, spline, label="spline")
        ax[0,k].set_ylabel(f"pos {axis}[m]")

        ax[1,k].plot(t, vel[:,k], label="data")
        spline = sf.evald(t)
        ax[1,k].plot(t, spline, label="spline")
        ax[1,k].set_ylabel(f"vel {axis}[m/s]")

        ax[2,k].plot(t, acc[:,k], label="data")
        spline = sf.evaldd(t)
        ax[2,k].plot(t, spline, label="spline")
        acc_spline.append(spline)

        ax[2,k].set_ylabel(f"acc {axis}[m/s^2]")

    ax[0,0].legend()
    plt.show()

    acc_spline = np.array(acc_spline).T
    
    sf = SplineFitter()

    acc_payload_spline = []
    fig, ax = plt.subplots(3, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, pos_payload[:,k], label="data")

        sf.fit(t, pos_payload[:,k], 50)
        spline = sf.eval(t)
        ax[0,k].plot(t, spline, label="payload spline")
        ax[0,k].set_ylabel(f"payload pos {axis}[m]")

        ax[1,k].plot(t, vel[:,k], label="data")
        spline = sf.evald(t)
        ax[1,k].plot(t, spline, label="spline")
        ax[1,k].set_ylabel(f"payload vel {axis}[m/s]")

        ax[2,k].plot(t, acc[:,k], label="data")
        spline = sf.evaldd(t)
        ax[2,k].plot(t, spline, label="spline")
        acc_payload_spline.append(spline)

        ax[2,k].set_ylabel(f"payload acc {axis}[m/s^2]")
    ax[0,0].legend()
    plt.show()

    acc_payload_spline = np.array(acc_payload_spline).T

    # fa vs tau_a
    rpm = np.array([
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4'],
    ]).T

    rpy = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.rpyx'],
        data_usd['fixedFrequency'][f'ctrlLeeP.rpyy'],
        data_usd['fixedFrequency'][f'ctrlLeeP.rpyz']]).T
    import rowan
    q = rowan.from_euler(rpy[:,0], rpy[:,1], rpy[:,2], "xyz", "extrinsic")


    kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
    force = kappa_f * rpm**2
    f = force.sum(axis=1)
    u = np.zeros_like(rpy)
    u[:,2] = f

    mass = 0.038
    mass_p = 0.005
    J = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])

    g = np.array([0,0,-9.81])

    cable_q = (pos_payload - pos) / np.tile(np.linalg.norm(pos_payload - pos, axis=1), [3,1]).T

    # T = (-mass_p * cable_q).dot(acc_payload_spline - g)

    # fa = mass * acc - mass * g - rowan.rotate(q, u) - T*cable_q
    # print(fa)



    fa = [] # fa on uav
    fap = [] # fa on payload

    fac = [] # fa on uav


    for i in range(force.shape[0]):
        T = (-mass_p * cable_q[i]).dot(acc_payload_spline[i] - g)

        T2 = mass_p * np.linalg.norm(acc_payload_spline[i] -g)
        print(T, T2)

        # fa.append(mass * acc[i] - mass * g - rowan.rotate(q[i], u[i]) - T*cable_q[i])
        fa.append((mass * acc[i] - mass * g - rowan.rotate(q[i], u[i]) - T2*cable_q[i])/mass)

        fac.append(mass_p * acc_payload_spline[i] + mass * acc[i] - rowan.rotate(q[i], u[i]) - mass * g - mass_p * g)
        fap.append((mass_p * acc_payload_spline[i] + T2 * cable_q[i] - mass_p * g)/mass_p)

    fa = np.array(fa)
    fap = np.array(fap)
    fac = np.array(fac)



    # # fa version on payload
    # fap = []
    # for i in range(force.shape[0]):
    #     fap.append(mass_p * acc_payload_spline[i] + mass * acc[i] - rowan.rotate(q[i], u[i]) - mass * g - mass_p * g)
    # fap = np.array(fap)

    a_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.a_rpm_fx'],
        data_usd['fixedFrequency'][f'ctrlLeeP.a_rpm_fy'],
        data_usd['fixedFrequency'][f'ctrlLeeP.a_rpm_fz']]).T
    
    a_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.a_imu_fx'],
        data_usd['fixedFrequency'][f'ctrlLeeP.a_imu_fy'],
        data_usd['fixedFrequency'][f'ctrlLeeP.a_imu_fz']]).T

    fig, ax = plt.subplots(1, 3, sharex='all', squeeze=False)
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, fa[:,k], label="fa (on UAV)")
        ax[0,k].plot(t, fap[:,k], label="fa (on payload)")
        # ax[0,k].plot(t, fac[:,k], label="fa (classic)")
        # ax[0,k].plot(t, -(a_rpm_filtered-a_imu_filtered)[:,k]*mass, label="INDI", alpha=0.5)
        ax[0,k].set_ylabel(f"f_a {axis}")
    ax[0,0].legend()

    plt.show()



    fig, ax = plt.subplots(1, 3, sharex='all', squeeze=False)
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, fa[:,k], label="fa (on UAV)")
        ax[0,k].plot(t, fap[:,k], label="fa (on payload)")
        ax[0,k].plot(t, -(a_rpm_filtered-a_imu_filtered)[:,k], label="INDI", alpha=0.5)
        ax[0,k].set_ylabel(f"f_a {axis}")
    ax[0,0].legend()

    plt.show()


    sf = SplineFitter()

    omega_dot_spline = []
    fig, ax = plt.subplots(2, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, omega[:,k], label="data")

        sf.fit(t, omega[:,k], 200)
        spline = sf.eval(t)
        ax[0,k].plot(t, spline, label="spline")
        ax[0,k].set_ylabel(f"omega {axis}[rad/s]")

        ax[1,k].plot(t[0:-1], np.diff(omega[:,k])/np.diff(t), label="data")
        spline = sf.evald(t)
        ax[1,k].plot(t, spline, label="spline")
        ax[1,k].set_ylabel(f"omega dot {axis}[rad/s^2]")
        omega_dot_spline.append(spline)
    ax[0,0].legend()
    plt.show()

    omega_dot_spline = np.array(omega_dot_spline).T

    # mass = 0.034
    # inertia = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])
    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    allocation_matrix = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
        ])
    
    tau_a = []
    for i in range(force.shape[0]):
        eta = allocation_matrix @ force[i]
        tau_a.append(J @ omega_dot_spline[i] - np.cross(J @ omega[i], omega[i]) - eta[1:])

    tau_a = np.array(tau_a)
   
    tau_rpm_filtered = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_rpm_fx'],
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_rpm_fy'],
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_rpm_fz']]).T
        
    tau_imu_filtered = np.array([
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_imu_fx'],
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_imu_fy'],
        data_usd['fixedFrequency'][f'ctrlLeeP.tau_imu_fz']]).T 

    fig, ax = plt.subplots(1, 3, sharex='all', squeeze=False)
    for k, axis in enumerate(["x", "y", "z"]):
        ax[0,k].plot(t, tau_a[:,k], label="tau_a")
        ax[0,k].plot(t, -(tau_rpm_filtered-tau_imu_filtered)[:,k], label="INDI", alpha=0.5)
        ax[0,k].set_ylabel(f"tau_a {axis}")
    ax[0,0].legend()

    plt.show()