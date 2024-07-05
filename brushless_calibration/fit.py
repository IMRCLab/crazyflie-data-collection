import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

data_log = np.loadtxt("data/data_calib_brushless.csv", skiprows=1, delimiter=',')
data_scale = np.loadtxt("data/manual_data_calib_brushless.csv", skiprows=1, delimiter=',')


# Plot data; note there seems to be a problem with rpm1 - 3 (very inconsistent readings)

# plt.scatter(data_log[:,1], data_log[:,3]) # pwm vs rpm1
# plt.scatter(data_log[:,1], data_log[:,4]) # pwm vs rpm2
# plt.scatter(data_log[:,1], data_log[:,5]) # pwm vs rpm3
plt.scatter(data_log[:,1], data_log[:,6]) # pwm vs rpm4

plt.show()

data = []
for pwm, weight_in_grams in data_scale:
    data_log_filtered = data_log[data_log[:,1]==pwm]
    rpm_mean = np.mean(data_log_filtered[:,6:7])
    print(pwm, rpm_mean)
    data.append([pwm, weight_in_grams / 4, rpm_mean])
data = np.array(data)

# fit a function (rpm -> thrust)
kw = cp.Variable()
cost = cp.sum_squares(data[:,1] - kw * data[:,2]**2)
prob = cp.Problem(cp.Minimize(cost), [])
prob.solve()
print(kw.value)

plt.plot(data[:,2], data[:,1])

fitted = kw.value * data[:,2]**2
plt.plot(data[:,2], fitted)

plt.show()

# fit a function (pwm -> thrust)
a = cp.Variable()
b = cp.Variable()

cost = cp.sum_squares(data[3:-3,1] - (a + b * data[3:-3,0]))
prob = cp.Problem(cp.Minimize(cost), [])
prob.solve()

plt.plot(data[:,0], data[:,1])
fitted = a.value + b.value * data[:,0]
plt.plot(data[:,0], fitted)
print(a.value, b.value)

plt.show()

# fit a function (pwm -> thrust, default crazyflie firmware)
pwmToThrustA = cp.Variable()
pwmToThrustB = cp.Variable()

thrust_in_N_per_rotor = data[2:-2,1] / 1000 * 9.81
pwm_normalized = data[2:-2,0]/65565

cost = cp.sum_squares(thrust_in_N_per_rotor - (pwmToThrustA * pwm_normalized**2 + pwmToThrustB * pwm_normalized))
prob = cp.Problem(cp.Minimize(cost), [])
prob.solve()

print("pwmToThrustA", pwmToThrustA.value)
print("pwmToThrustB", pwmToThrustB.value)

plt.plot(pwm_normalized, thrust_in_N_per_rotor)
fitted = pwmToThrustA.value * pwm_normalized**2 + pwmToThrustB.value * pwm_normalized
plt.plot(pwm_normalized, fitted)

plt.show()