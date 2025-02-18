import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cfusdlog


f = "tle67"
data  = cfusdlog.decode(f)['fixedFrequency'] 
starttime = data['timestamp'][0] 
time = ((data['timestamp'] - starttime)/1000.0).tolist()

print(data.keys())
for key in data.keys():
    if "ctrlLee.omegax" in key:
        omega = np.array([data["ctrlLee.omegax"] ,data["ctrlLee.omegay"], data["ctrlLee.omegaz"]])
    elif "ctrlLee.tau_gyro_x" in key:
        J = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])
        J_inv = np.linalg.inv(J)
        omega_dot = J_inv @ np.array([data["ctrlLee.tau_gyro_x"], data["ctrlLee.tau_gyro_y"], data["ctrlLee.tau_gyro_z"]])
    elif "ctrlLee.tau_rpm_fx" in key:
        J = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])
        J_inv = np.linalg.inv(J)
        omega_dot_rpm = J_inv @ np.array([data["ctrlLee.tau_rpm_fx"], data["ctrlLee.tau_rpm_fy"], data["ctrlLee.tau_rpm_fz"]])

    else:
        continue

data_xyz = []
for i in range(3):
    data_i = np.zeros((4,len(time)))
    data_i[0,:] = time
    data_i[1,:] = omega[i,:]
    data_i[2,:] = omega_dot[i,:]
    data_i[3,:] = omega_dot_rpm[i,:]
    data_xyz.append(data_i.T)

degree = 4
num_segments = 250
a = 0.000001 #50000 # weight for the least square error 
l = 0.0 #0.1 #0.0000001 # weight for the regularization
ys_vals = []
ys_der_vals = []
ys_dder_vals = []
axis = ["x", "y", "z"]
for k, data_points in enumerate(data_xyz):
    coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
    num_points = len(data_points)
    len_segment = int(np.ceil(num_points/num_segments))
    T = data_points[-1][0] - data_points[0][0]
    T_segment = T / num_segments
    cost = 0
    x_vals = []
    x_normalized_vals = []
    constraints = []

    j = 0
    for i in range(num_segments):
        # start_id = i*(len_segment-1)
        # end_id   =  min(start_id + len_segment, num_points)
        x_normalized_val = []
        # for j in range(start_id, end_id):
        start_id = j
        while j < len(data_points):
            point = data_points[j]
            t = point[0]
            t_normalized = (t - i*T_segment)/T_segment
            if t_normalized > 1.0:
                break
            x_normalized_val.append(t_normalized)
            print(t, t_normalized)
            y = point[1]
            y_poly = sum([coeffs[i][d]*t_normalized**d for d in range(degree+1)])
            cost += a*cp.sum_squares(y_poly - y)
            j = j + 1
        cost += l * cp.sum_squares(coeffs[i])
        x_vals.append(data_points[start_id:j,0])
        x_normalized_vals.append(x_normalized_val)

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

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 10), sharex=True)


    for j, coeff in enumerate(coeffs):
        x_val = np.linspace(0, 1, 100)
        y_est = [sum([coeff[d].value * x**d  for d in range(degree+1)])  for x in x_val]
        y_est_der = [sum([(d)*coeff[d].value * x**(d-1) / T_segment  for d in range(1,degree+1)])  for x in x_val]
        ax1.plot(x_val*T_segment + j*T_segment, y_est,  linewidth=2)
        ax2.plot(x_val*T_segment + j*T_segment, y_est_der,  linewidth=2)
\
    ax1.set_ylabel(f"omega{axis[k]}")
    ax2.set_ylabel(f"omega_dot{axis[k]}")

    # Plot data items
    ax1.plot(data_points[:, 0], data_points[:, 1], color='blue', label='omega', linewidth=1.2)
    ax2.plot(data_points[:, 0], data_points[:, 2], color='darkorange', label='omega_dot_gyro', linewidth=1.2)
    # ax2.plot(data_points[0:-1, 0], np.diff(data_points[:, 1])/np.diff(data_points[:, 0]), color='black', label='np.diff', linewidth=1.2)

    ax1.grid()
    ax2.grid()
    ax1.set_title('Piecewise Polynomial Regression')
    ax1.legend()
    ax2.legend()
    plt.show()
plt.show()