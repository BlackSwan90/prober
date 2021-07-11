import numpy as np
import matplotlib.pyplot as plt

# v_avg = np.array([-1,-0.1,0,0.1,1])
# v_std = np.array([1e-1,0,0,1e-1,1e-1])
# i_avg = np.array([-1e-10,-1e-12,1e-8,1e-5,1e-3])
# i_std = np.array([1e-10,1e-12,1e-8,1e-5,2e-3])

data = np.loadtxt('IV_curve_1.txt',skiprows=2)
v_avg = data[:,1]
i_avg = data[:,2]


fig, ax = plt.subplots()
ax.set_xscale("linear")
ax.set_yscale("log")

ax.errorbar(v_avg,np.abs(i_avg),
            # xerr=v_std,yerr=i_std,
            marker='o',
            color='black',ecolor='green',
            linewidth=2)

plt.xlabel('V, volts')
plt.ylabel('I, amps')
plt.grid(True)
plt.show()


fig, ax = plt.subplots()
ax.set_xscale("linear")
ax.set_yscale("linear")

ax.errorbar(v_avg,np.abs(i_avg),
            # xerr=v_std,yerr=i_std,
            marker='o',
            color='black',ecolor='green',
            linewidth=2)

plt.xlabel('V, volts')
plt.ylabel('I, amps')
plt.grid(True)
plt.show()