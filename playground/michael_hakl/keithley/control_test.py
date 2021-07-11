import pyvisa as visa
import time

def tryread():
    time.sleep(0.1)
    try:
        reading = ctrl.read_raw()
        return reading
    except Exception:
        tryread()

manager = visa.ResourceManager()
# ctrl = manager.open_resource("TCPIP0::169.254.25.205::inst0::INSTR",timeout=50)
ctrl = manager.open_resource("TCPIP0::169.254.25.205::inst0::INSTR",timeout=50)


time.sleep(1)
ctrl.write("reset()")

ctrl.write('smua.measure.rangei = 1e-3')
ctrl.write("smua.measure.autorangei=smua.AUTORANGE_OFF")
ctrl.write("smua.measure.autorangev=smua.AUTORANGE_OFF")
ctrl.write("smua.source.levelv=0")

ctrl.write('smua.source.output=smua.OUTPUT_ON')

t_imeasurement = 2

v_arr = []
i_arr = []
t_start = time.time()

# for ii in range(4):
#     ctrl.write("smua.source.levelv={voltage}".format(voltage=ii))
#     ctrl.write("vval=smua.measure.v()")
#     ctrl.write("print(vval)")
#     vvalue = ctrl.read_raw()
#     # vvalue = tryread()
#     ctrl.write("ival=smua.measure.i()")
#     ctrl.write("print(ival)")
#     # time.sleep(t_imeasurement)
#     ivalue = ctrl.read_raw()
#     # ivalue = tryread()
#     v_arr.append(vvalue)
#     i_arr.append(ivalue)

# t_end = time.time() - t_start

ctrl.write("smua.measure.filter.enable = smua.FILTER_ON")
ctrl.write("smua.measure.filter.type = smua.FILTER_REPEAT_AVG")
ctrl.write("smua.measure.filter.count = 10")

ctrl.write("smua.measure.interval = 0.5")

ctrl.write("smua.source.levelv=1")
ctrl.write('smua.source.output=smua.OUTPUT_ON')
ctrl.write("smua.measure.count=3")
ctrl.write("buf = smua.makebuffer(20)")
ctrl.write("buf.clear()")
ctrl.write("buf.collecttimestamps=1")
ctrl.write("buf.collectsourcevalues=1")

ctrl.write("smua.measure.i(buf)")
ctrl.write("printbuffer(1,10,buf.readings)")
ivals = ctrl.read_raw()
ctrl.write("printbuffer(1,10,buf.timestamps)")
timestamps = ctrl.read_raw()
ctrl.write("printbuffer(1,10,buf.sourcevalues)")
sourcevalues = ctrl.read_raw()
ctrl.write("printbuffer(1,10,buf.measureranges)")
measureranges = ctrl.read_raw()



# for ii in range(4):
#     ctrl.write("smua.source.levelv={voltage}".format(voltage=ii))
#     ctrl.write("vval=smua.measure.v()")
#     ctrl.write("print(vval)")
#     vvalue = ctrl.read_raw()
#     # vvalue = tryread()
#     ctrl.write("ival=smua.measure.i()")
#     ctrl.write("print(ival)")
#     # time.sleep(t_imeasurement)
#     ivalue = ctrl.read_raw()
#     # ivalue = tryread()
#     v_arr.append(vvalue)
#     i_arr.append(ivalue)


# ctrl.write('getlevel = makegetter(smua.source, "levelv")')
# ctrl.write('v = getlevel()')
# ctrl.write('print(v)')
# tryread()


ctrl.write('smua.measure.rangei =1e-3')
ctrl.write('get_rangei = makegetter(smua.measure, "rangei")')
ctrl.write('val = get_rangei()')
ctrl.write('print(val)')
tryread()

# ctrl.write('smua.measure.rangei = 1e-2')
ctrl.write('get_autorangei = makegetter(smua.measure, "autorangei")')
ctrl.write('val = get_autorangei()')
ctrl.write('print(val)') #0=off,1=on
tryread()

ctrl.write("smua.measure.count = 5")
ctrl.write("mybuffer = smua.makebuffer(5)")
ctrl.write("smua.measure.i(mybuffer)")
ctrl.write("printbuffer(1, 100, mybuffer")


ctrl.write(":OUTP OFF")
ctrl.close()
