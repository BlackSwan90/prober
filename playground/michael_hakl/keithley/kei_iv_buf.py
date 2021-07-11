import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import win32clipboard

v_avg = np.array([-1,-0.1,0,0.1,1])
v_std = np.array([1e-1,0,0,1e-1,1e-1])
i_avg = np.array([-1e-10,-1e-12,1e-8,1e-5,1e-3])
i_std = np.array([1e-10,1e-12,1e-8,1e-5,2e-3])

filename = 'IV_curve.txt'
# source_list = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
# source_arr = np.asarray(source_list)
# N_source = len(source_arr)
# N_rep = 3
# N_tot = N_source * N_rep
# t_measure = 0.5
# t_stabil = 0.3
#
# manager = visa.ResourceManager()
# ctrl = manager.open_resource("TCPIP0::169.254.25.205::inst0::INSTR", timeout=50)
# ctrl.write("reset()")
#
# # ctrl.write('smua.measure.rangei = 1e-3')
# # ctrl.write('smua.measure.rangev = 2')
#
# ctrl.write("smua.measure.autorangei=smua.AUTORANGE_ON")
# ctrl.write("smua.measure.autorangev=smua.AUTORANGE_ON")
# ctrl.write("smua.source.levelv=0")
# ctrl.write("smua.measure.interval = {t}".format(t=t_measure))
# ctrl.write("smua.measure.delay = {t}".format(t=t_stabil))
#
# ctrl.write("vbuf = smua.makebuffer({N})".format(N=N_tot))
# ctrl.write("vbuf.clear()")
# ctrl.write("ibuf = smua.makebuffer({N})".format(N=N_tot))
# ctrl.write("ibuf.clear()")
#
# ctrl.write('smua.source.output=smua.OUTPUT_ON')
#
# for ind_source in range(N_source):
#     ctrl.write("smua.source.levelv={V}".format(V=source_arr[ind_source] ))
#     ctrl.write("smua.measure.iv(ibuf,vbuf)")
#
# ctrl.write("printbuffer(1,{N},ibuf)".format(N=N_tot)
# vbuf = ctrl.read_raw()
# ctrl.write("printbuffer(1,{N},vbuf)".format(N=N_tot)
# ibuf = ctrl.read_raw()
#
#
# v_arr_flat = np.asarray(vbuf.split(","))
# v_arr = np.reshape(v_arr_flat, (N_source,N_rep) )
# v_avg = v_arr.mean(axis=0)
# v_std = v_arr.std(axis=0)
#
# i_arr_flat = np.asarray(ibuf.split(","))
# i_arr = np.reshape(i_arr_flat, (N_source,N_rep) )
# i_avg = i_arr.mean(axis=0)
# i_std = i_arr.std(axis=0)

data = np.hstack((v_avg, v_std, i_avg, i_std))
# np.savetxt(filename,data, delimiter=",")

win32clipboard.OpenClipboard()
win32clipboard.EmptyClipboard()
win32clipboard.SetClipboardText(data)
win32clipboard.CloseClipboard()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_xscale("linear")
ax1.set_yscale("log")
ax1.errorbar(v_avg, np.abs(i_avg),
            xerr=v_std, yerr=i_std,
            marker='o',
            color='black', ecolor='green',
            linewidth=2)
ax1.set_xlabel('V, volts')
ax1.set_ylabel('I, amps')
ax1.grid(True)

ax2.set_xscale("linear")
ax2.set_yscale("linear")
ax2.errorbar(v_avg, np.abs(i_avg),
            xerr=v_std, yerr=i_std,
            marker='o',
            color='black', ecolor='green',
            linewidth=2)
ax2.set_xlabel('V, volts')
ax2.set_ylabel('I, amps')
ax2.grid(True)
fig.set_size_inches(15, 5, forward=True)
plt.show()

