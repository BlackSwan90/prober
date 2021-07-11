import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import win32clipboard

filename = 'IV_curve_1.txt'
comment = 'test1'



list_IV = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
# list_IV = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]

pt_no = np.arange(len(list_IV)).T
N_IV = len(list_IV)
N_rep = 3
N_tot = N_IV*N_rep
t_measure = 0.5
t_stabil = 0.3

manager = visa.ResourceManager()
ctrl = manager.open_resource("TCPIP0::169.254.25.205::inst0::INSTR",timeout=50)

ctrl.write("reset()")

# ctrl.write('smua.measure.rangei = 1e-3')
# ctrl.write('smua.measure.rangev = 2')

ctrl.write("smua.measure.autorangei=smua.AUTORANGE_ON")
ctrl.write("smua.measure.autorangev=smua.AUTORANGE_ON")
ctrl.write("smua.source.levelv=0")
# ctrl.write("smua.measure.interval = {t}".format(t=t_measure))
ctrl.write("smua.measure.delay = {t}".format(t=t_stabil))

ctrl.write('smua.source.output=smua.OUTPUT_ON')

v_list = []
i_list = []
for i_IV in range(N_IV):
    ctrl.write("smua.source.levelv={V}".format( V=float( list_IV[i_IV] ) ) )

    ctrl.write("vval=smua.measure.v()")
    ctrl.write("print(vval)")
    v_val = float(ctrl.read_raw())

    ctrl.write("ival=smua.measure.i()")
    ctrl.write("print(ival)")
    i_val = float(ctrl.read_raw())

    v_list.append(v_val)
    i_list.append(i_val)

v_arr = np.asarray(v_list)
i_arr = np.asarray(i_list)

data = np.hstack((pt_no,v_arr, i_arr))
column_labels = 'pt_no, V (V), I(A) \n'
header = comment + column_labels
np.savetxt(filename,data, delimiter=",",header=header)

# win32clipboard.OpenClipboard()
# win32clipboard.EmptyClipboard()
# win32clipboard.SetClipboardText(data)
# win32clipboard.CloseClipboard()

# fig, ax1 = plt.subplots(1,1)
# # ax1.set_xscale("linear")
# # ax1.set_yscale("log")
# ax1.errorbar(v_avg, np.abs(i_avg),marker='o',color='black',linewidth=2)
# ax1.set_xlabel('V, volts')
# ax1.set_ylabel('I, amps')
# ax1.grid(True)

