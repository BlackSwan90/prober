import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from cycler import cycler
import colorsys
import os
import time
import pyvisa as visa

class instr_class():

    class Keithley2600():

        def __init__(self):
            pass

        def instr_open(self):
            t_measure = 0
            t_stabil = 1

            self.kei_ctrl = self.RM.open_resource("TCPIP0::192.168.104.102::inst0::INSTR", timeout=10000)  # node 1
            time.sleep(1)

            self.kei_ctrl.write("reset()")
            self.kei_ctrl.write("smua.measure.autozero = smua.AUTOZERO_AUTO")
            self.kei_ctrl.write("smua.measure.autorangei = smua.AUTORANGE_ON")
            self.kei_ctrl.write("smua.measure.autorangev = smua.AUTORANGE_ON")
            self.kei_ctrl.write("smua.measure.analogfilter = 1")
            self.kei_ctrl.write("smua.measure.filter.enable = 0")
            # kei_ctrl.write("smua.measure.delay = smua.DELAY_AUTO")
            # kei_ctrl.write("smua.measure.delayfactor = 1")
            self.kei_ctrl.write("smua.measure.delay = {t}".format(t=t_stabil))
            self.kei_ctrl.write("smua.measure.interval = {t}".format(t=t_measure))
            self.kei_ctrl.write("smua.measure.nplc = 1")

            self.kei_ctrl.write("smua.sense = smua.SENSE_REMOTE")  # 4 wire
            # kei_ctrl.write("smua.sense = smua.SENSE_LOCAL") # 2 wire
            self.kei_ctrl.write("display.smua.measure.func = display.MEASURE_DCAMPS")

            self.kei_ctrl.write("smua.source.limiti = 10.0e-3")
            self.kei_ctrl.write("smua.source.settling = 1")
            self.kei_ctrl.write("smua.source.levelv=0")
            self.kei_ctrl.write('smua.source.output=smua.OUTPUT_ON')

        def instr_close(self):
            self.kei_ctrl.write('smua.source.output=smua.OUTPUT_OFF')
            time.sleep(0.1)
            self.kei_ctrl.close()

        def get_iv(self):
            v_m_list = []
            i_m_list = []

            self.kei_ctrl.write("smua.source.levelv={V}".format(V=TM.volt[0]))
            time.sleep(0.1)

            self.kei_ctrl.write("vval=smua.measure.v()")
            self.kei_ctrl.write("print(vval)")
            v_val = float(self.kei_ctrl.read_raw())
            self.kei_ctrl.write("ival=smua.measure.i()")
            self.kei_ctrl.write("print(ival)")
            i_val = float(self.kei_ctrl.read_raw())

            v_m_list.append(v_val)
            i_m_list.append(i_val)

            for ind_v in list(range(1, TM.N_sample_IV, 1)):
                print('applying ', str(TM.volt[ind_v]), 'V')
                self.kei_ctrl.write("smua.source.levelv={V}".format(V=TM.volt[ind_v]))
                time.sleep(0.1)

                self.kei_ctrl.write("vval=smua.measure.v()")
                self.kei_ctrl.write("print(vval)")
                v_val = float(self.kei_ctrl.read_raw())
                self.kei_ctrl.write("ival=smua.measure.i()")
                self.kei_ctrl.write("print(ival)")
                i_val = float(self.kei_ctrl.read_raw())

                v_m_list.append(v_val)
                i_m_list.append(i_val)

            self.kei_ctrl.write('smua.source.output=smua.OUTPUT_OFF')

            v_arr = np.asarray(v_m_list)
            i_arr = np.asarray(i_m_list)

    class Cascade13k():
        def __init__(self):
            pass

        def instr_open(self):
            self.prober_ctrl = self.RM.open_resource("GPIB0::1::INSTR", timeout=10000, send_end=True)  # ,read_termination='r',write_termination='r')
            time.sleep(1)

            # activate contact mode
            self.prober_ctrl.query(":set:contact:active?")
            self.prober_ctrl.query(":set:contact:active ON")
            self.prober_ctrl.query(":set:cont? 2")

        def instr_close(self):
            self.prober_ctrl.close()

        def read_coor(self):
            resp = self.prober_ctrl.query(":mov:abs? 2")
            resp = resp.strip()
            resp = resp.split(' ')
            resp = np.array(resp, dtype=int)
            return resp

        def move_xy(self, coor):
            self.prober_ctrl.query(":mov:abs 2 {x} {y} none".format(x=coor[1], y=coor[1]))
            self.prober_ctrl.query(":mov:abs? 2")

        def touchdown(self):
            self.prober_ctrl.query(":mov:cont 2")
            # self.prober_ctrl.query(":mov:cont? 2")  # true

        def separate(self):
            self.prober_ctrl.query(":mov:sep 2")
            # self.prober_ctrl.query(":mov:sep? 2")  # false

        def go2site(self, coor):
            self.separate()
            self.move_xy(coor)
            self.touchdown()

    class RingSource():
        def __init__(self):
            self.calib = np.genfromtxt(r'c:\py37_xtbox\playground\michael_hakl\prober_control\IR_calib.txt')

        def instr_open(self):
            self.PWS4323_ctrl = self.RM.open_resource("USB0::0x0699::0x0391::C022000::instrrR", timeout=2000)

            # init
            self.PWS4323_ctrl.write("*RST")
            self.PWS4323_ctrl.write('SYSTEM:REMOTE')
            self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')
            self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL 0V')
            self.PWS4323_ctrl.write('SOURCE:CURRENT:LEVEL 0mA')

            # set ranges+levels
            self.PWS4323_ctrl.write('SOURCE:VOLTAGE:RANGE 32.1V')
            self.PWS4323_ctrl.write('SOURCE:CURRENT:LEVEL 2mA')
            self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL 6V')

            # self.PWS4323_ctrl.write('SOURCE:CURRENT:PROTECTION:STATE ON')

        def instr_close(self):
            self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')
            self.PWS4323_ctrl.close()

        def out_on(self):
            self.PWS4323_ctrl.write('SOURCE:OUTPUT ON')

        def out_off(self):
            self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')

        def setInt(self, val):
            self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL {val}V'.format(val=self.calib[val, 1]))
            time.sleep(0.2)

class TM(instr_class):
    def __init__(self):
        self.sites_selection = []
        self.N_site_selection = 0

        self.IRring = 0
        self.ID_w = ''
        self.i_IV = 0

        self.N_sample_IV = 11
        self.volt = np.linspace(-3, 3, num=self.N_sample_IV)
        self.curr = np.zeros((self.N_sample_IV, 1))

        # self.gen_site_map()
        self.gen_wafer_map()

        #self.instr_wakeup()

    def gen_frontend(self):
        self.app = frontend()
        self.app.root.mainloop()

    def gen_site_map(self):
        coor = np.vstack(((np.arange(11) - 5) * 1.5e3, (np.arange(11) - 5) * 1.5e3)).T
        x_grid, y_grid = np.meshgrid(coor[:, 0], coor[:, 1])
        sites_x = np.ravel(x_grid)
        sites_y = np.ravel(y_grid)
        self.N_site = len(sites_x)
        self.site_coor = np.vstack((sites_x, sites_y)).T
        self.site_ID = np.arange(self.N_site)
        self.site_type = np.ravel([np.ones(40), 2 * np.ones(40), 3 * np.ones(40)], order='F')
        self.site_type = np.hstack((self.site_type, np.array([1]))).astype(dtype='int')
        header = '# ID,type,coor_x[um], coor_y[um] \n'
        path = r'C:\py37_xtbox\playground\michael_hakl\prober_control\sites_on_grid.txt'
        f = open(path, 'w')
        f.write(header)
        for i in range(self.N_site):
            f.write('{s_ID},\t {s_type},\t {x},\t {y} \n'.format(s_ID=self.site_ID[i],
                                                                 s_type=int(self.site_type[i]),
                                                                 x=sites_x[i],
                                                                 y=sites_y[i]))
        f.close()

    def gen_wafer_map(self):
        self.die_period = [9410, 7950]
        self.N_r = 17
        self.N_c = 14
        ind_map = np.zeros((self.N_r, self.N_c), dtype='uint8')
        ind_map[0, 5:9] = 1
        ind_map[1, 3:11] = 1
        ind_map[2, 2:12] = 1
        ind_map[3, 2:13] = 1
        ind_map[4, 1:13] = 1
        ind_map[5, 1:] = 1
        ind_map[6, 0:] = 1
        ind_map[7, 0:] = 1
        ind_map[8, 0:] = 1
        ind_map[9:, :] = np.flipud(ind_map[:8, :])
        self.ind_map = ind_map
        self.N_site = np.sum(ind_map)

        self.site_coor = np.zeros((self.N_site, 2))
        i_s = 0
        for i_r in range(self.N_r):
            for i_c in range(self.N_c):
                if self.ind_map[self.N_r - 1 - i_r, i_c] == 1:
                    self.site_coor[i_s, 0] = i_c * self.die_period[0] - 6.5 * self.die_period[0]  # x
                    self.site_coor[i_s, 1] = i_r * self.die_period[1] - 8 * self.die_period[1]  # y
                    i_s += 1

        self.site_ID = np.arange(self.N_site)
        self.site_type = np.ones(self.N_site, dtype='int')
        header = '# ID,type,coor_x[um], coor_y[um] \n'
        path = r'C:\py37_xtbox\playground\michael_hakl\prober_control\3506_map.txt'
        f = open(path, 'w')
        f.write(header)
        for i in range(self.N_site):
            f.write('{s_ID},\t {s_type},\t {x},\t {y} \n'.format(s_ID=self.site_ID[i],
                                                                 s_type=int(self.site_type[i]),
                                                                 x=self.site_coor[i, 0],
                                                                 y=self.site_coor[i, 1]))
        f.close()

        # plt.ion()
        # plt.scatter(x=self.site_coor[:,0],y=self.site_coor[:,1])
        # plt.show()

    def instr_wakeup(self):
        self.RM = visa.ResourceManager()

        self.kei = instr_class.Keithley2600()
        self.kei.instr_open()

        self.prober = instr_class.Cascade13k()
        self.prober.instr_open()

        self.IR = instr_class.RingSource()
        self.IR.instr_open()

    def instr_byebye(self):

        self.kei.instr_close()
        self.prober.instr_close()
        self.IR.instr_close()

class frontend(TM):
    def __init__(self):
        self.generate_layout()

    def generate_layout(self):

        self.root = tk.Tk()
        self.root.wm_title("T&M with IV")

        self.frame1 = tk.Frame(self.root)
        self.frame1.grid(row=1, column=1, columnspan=2, sticky='', padx=20)

        # self.TMmode_radio = tk.IntVar()
        # self.TMmode_radio.set(1)
        # tk.Label(self.frame1, text='Experiment:').grid(row=1, column=1, columnspan=2)
        # tk.Radiobutton(self.frame1, text="Simple IV", variable=self.TMmode_radio, value=1, command=self.cb_TMradio).grid(row=2, column=1)
        # tk.Radiobutton(self.frame1, text="Test structure", padx=5, variable=self.TMmode_radio, value=2, command=self.cb_TMradio).grid(row=2, column=2)

        self.butt_load_sites = tk.Button(self.frame1, text='Load map', command=self.cb_load_site_layout)
        self.butt_load_sites.grid(row=2, column=3)

        tk.Label(self.frame1, text='Select structure:').grid(row=1, column=3, columnspan=2)
        self.butt_allsites = tk.Button(self.frame1, text='All sites', command=self.cb_allsites)
        self.butt_allsites.grid(row=2, column=4)

        self.butt_nosite = tk.Button(self.frame1, text='No site', command=self.cb_nosite)
        self.butt_nosite.grid(row=2, column=5)

        self.butt_load_site_selection = tk.Button(self.frame1, text='Load selection', command=self.cb_load_site_selection)
        self.butt_load_site_selection.grid(row=2, column=6)

        self.butt_save_site_selection = tk.Button(self.frame1, text='Save selection', command=self.cb_save_site_selection)
        self.butt_save_site_selection.grid(row=2, column=7)

        tk.Label(self.frame1, text='Wafer ID:').grid(row=1, column=8)
        self.name_w_var = tk.StringVar()
        self.name_w = tk.Entry(self.frame1, textvariable=self.name_w_var)
        self.name_w.grid(row=2, column=8, padx=20)
        self.name_w.bind("<Return>", self.cb_get_ID_w)

        tk.Label(self.frame1, text='Ref ID site:').grid(row=1, column=9)
        self.name_refsite = tk.StringVar()
        self.name_refsite = tk.Entry(self.frame1, textvariable=self.name_refsite)
        self.name_refsite.grid(row=2, column=9)
        self.name_refsite.bind("<Return>", self.cb_get_refsite)

        tk.Label(self.frame1, text='Data:').grid(row=1, column=10, columnspan=2)
        self.butt_loadIV = tk.Button(self.frame1, text='Load IV', command=self.cb_loadIV)
        self.butt_loadIV.grid(row=2, column=10)

        self.butt_saveIV = tk.Button(self.frame1, text='Save last IV', command=self.cb_saveIV)
        self.butt_saveIV.grid(row=2, column=11)

        self.butt_acquireIV = tk.Button(self.frame1, text='Get single IV', command=self.cb_acquireIV)
        self.butt_acquireIV.grid(row=2, column=12)

        self.butt_runTM = tk.Button(self.frame1, text='Run Selection', command=self.cb_runTM)
        self.butt_runTM.grid(row=2, column=13)

        self.status_IR = tk.IntVar()
        tk.Label(self.frame1, text='IR light').grid(row=1, column=14, columnspan=1, padx=20)
        self.chkbox_status_IR = tk.Checkbutton(self.frame1, variable=self.status_IR, command=self.cb_chkbox_status_IR).grid(row=2, column=14, padx=20)

        # tk.Label(self.frame1, text='Intensity:').grid(row=3, column=13, sticky='se')
        self.scale_setIR = tk.Scale(self.frame1, from_=0, to=100, orient=tk.HORIZONTAL, command=self.cb_setIR)
        self.scale_setIR.grid(row=3, column=14)

        self.butt_clear_graph = tk.Button(self.frame1, text='Clear plot', command=self.cb_clear_graph)
        self.butt_clear_graph.grid(row=2, column=15)

        self.generate_canvas_w()
        self.generate_IVgraph()

        self.frame4 = tk.Frame(self.root)
        self.frame4.grid(row=3, column=1, columnspan=1, sticky='')
        tk.Label(self.frame4, text='Event Log:').grid(row=1, column=1, sticky='W')
        self.field_log = tk.Text(self.frame4, height=20, width=70)
        self.field_log.grid(row=2, column=1)

    def generate_canvas_w(self):
        self.frame2 = tk.Frame(self.root)
        self.frame2.grid(row=2, column=1)
        self.canvas_w = tk.Canvas(self.frame2, width=500, height=500, bd=2, relief='ridge')

        ext = 2 * np.max(np.hypot(TM.site_coor[:, 0], TM.site_coor[:, 1]))  # the most remote die from center defines display perimeter
        um2pt = 470 / (1.1 * ext)  # circle diam[pt]/wafer diam[um]
        center_w = np.array([250, 250])

        # alternative way to generate colors
        # N_color = 6
        # HSV_tuples = [(x / N_color, 1, 0.5) for x in range(N_color)]
        # RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        RGB_palette = ['blue', 'gold', 'chocolate1', 'light grey', 'maroon1', 'purple1']
        self.canvas_s_coor = TM.site_coor * um2pt + center_w
        site_canvas = []
        for i in range(TM.N_site):
            # c = '#{r:02x}{g:02x}{b:02x}'.format(r=int(RGB_tuples[TM.site_type[i]][0] * 255),
            #                                     g=int(RGB_tuples[TM.site_type[i]][1] * 255),
            #                                     b=int(RGB_tuples[TM.site_type[i]][2] * 255))
            c = RGB_palette[TM.site_type[i]]

            site_canvas.append(self.canvas_w.create_rectangle(self.canvas_s_coor[i, 0] - 8, self.canvas_s_coor[i, 1] - 8,
                                                              self.canvas_s_coor[i, 0] + 8, self.canvas_s_coor[i, 1] + 8,
                                                              outline=c, fill='yellow', width=3))
            # canvas_w.create_text(sites_canvas_coor[i_site,0]+10, sites_canvas_coor[i_site,0]+10, text=i_site)
            # self.canvas_w.tag_bind(site_canvas[i], '<Button-1>', self.cb_select_1site)
        self.canvas_w.bind('<Button-1>', self.cb_select_1site)
        self.canvas_w.bind('<ButtonRelease-1>', self.cb_drag_select)

        self.canvas_w.create_oval(10, 10, 500 - 10, 500 - 10, outline='black', width=2)  # wafer border
        self.canvas_w.grid(pady=20)

    def generate_IVgraph(self):
        N_color = 10
        list_color = [plt.get_cmap('jet')(i / N_color) for i in range(N_color)]
        plt.rc('axes', prop_cycle=(cycler('color', list_color)), linewidth=1)
        plt.ioff()

        self.frame3 = tk.Frame(self.root)
        self.frame3.grid(row=2, column=2, rowspan=2, sticky='nsew')
        self.frame3.grid_rowconfigure(0, weight=1)
        self.fig1, self.ax1 = plt.subplots(1, 1)
        self.ax1.set_xlabel('V, volts')
        self.ax1.set_ylabel('I, amps')
        self.ax1.grid(True)
        self.graph_IV = FigureCanvasTkAgg(self.fig1, master=self.frame3)  # A tk.DrawingArea.
        toolbar = NavigationToolbar2Tk(self.graph_IV, self.frame3)
        toolbar.update()
        self.graph_IV.get_tk_widget().pack(pady=20, fill='both', expand=True)

    def cb_TMradio(self):
        self.TMmode = self.TMmode_radio.get()
        print(self.TMmode)

    def cb_loadIV(self):
        self.path_loadIV = tk.filedialog.askopenfile(initialdir=r'c:\py37_xtbox\playground\michael_hakl\tests\tinker', title='Open IV file')
        # path_loadIV = r'c:\py37_xtbox\playground\michael_hakl\tests\tinker\YFO_110Aulong.txt'
        self.loadIV = np.genfromtxt(self.path_loadIV, delimiter=",")[:, 1:]
        self.ax1.plot(self.loadIV[:, 0], self.loadIV[:, 1], marker='o', color='black', linewidth=2)
        self.graph_IV.draw()

    def cb_saveIV(self):
        f = tk.filedialog.asksaveasfile(mode='w', initialdir='/', initialfile=".txt", title='Save last IV')
        if f == None:
            return
        else:
            self.save_IV(f.name, '', np.vstack((TM.volt, TM.curr[:, -1])))

    def cb_get_ID_w(self, event):
        TM.ID_w = self.name_w_var.get()
        self.log_event('wafer ID set as: {}\n'.format(TM.ID_w))

    def cb_get_refsite(self, event):
        TM.refsite = self.name_refsite.get()
        self.log_event('Reference site set as: {}\n'.format(TM.refsite))

    def cb_select_1site(self, event):
        self.select_origin = [self.canvas_w.canvasx(event.x), self.canvas_w.canvasy(event.y)]

    def cb_drag_select(self, event):
        self.select_endpoint = [self.canvas_w.canvasx(event.x), self.canvas_w.canvasy(event.y)]
        coor = self.select_origin + self.select_endpoint
        selection = event.widget.find_overlapping(*coor)

        if len(selection) == 1:
            if self.canvas_w.itemcget(selection[0], 'fill') == 'green':
                self.canvas_w.itemconfig(selection[0], fill='yellow')
                TM.sites_selection.remove(selection[0] - 1)

            elif self.canvas_w.itemcget(selection[0], 'fill') == 'yellow':
                self.canvas_w.itemconfig(selection[0], fill='green')
                TM.sites_selection.append(selection[0] - 1)

        elif len(selection) > 1:
            for site in selection:
                if self.canvas_w.itemcget(site, 'fill') == 'yellow':
                    self.canvas_w.itemconfig(site, fill='green')
                    TM.sites_selection.append(site - 1)

        TM.sites_selection = sorted(TM.sites_selection)

    def cb_allsites(self):
        for i in range(TM.N_site):
            self.canvas_w.itemconfig(i + 1, fill='green')
        TM.sites_selection = list(range(0, TM.N_site, 1))

    def cb_nosite(self):
        for i in range(TM.N_site):
            self.canvas_w.itemconfig(i + 1, fill='yellow')
        TM.sites_selection = []

    def cb_load_site_layout(self):
        self.path_load_sites = tk.filedialog.askopenfile(initialdir=r'c:\py37_xtbox\playground\michael_hakl\prober_Control', title='Open sites')
        sites = np.genfromtxt(self.path_load_sites.name, delimiter=",")
        # cols in file: ID, type, x, y
        TM.site_ID = sites[:, 0].astype(dtype='int')
        TM.site_type = sites[:, 1].astype(dtype='int')
        TM.site_coor = sites[:, 2:]
        TM.N_site = len(TM.site_ID)
        self.generate_canvas_w()

    def cb_load_site_selection(self):
        self.path_load_selection = tk.filedialog.askopenfile(initialdir=r'c:\py37_xtbox\playground\michael_hakl\prober_control',
                                                             title='Open site selection')
        self.cb_nosite()

        TM.sites_selection = (np.genfromtxt(self.path_load_selection.name, delimiter=",").astype(dtype='int')).tolist()

        for site in TM.sites_selection:
            self.canvas_w.itemconfig(site + 1, fill='green')

    def cb_save_site_selection(self):
        f = tk.filedialog.asksaveasfile(mode='w', initialdir=r'c:\py37_xtbox\playground\michael_hakl\prober_control',
                                        initialfile=".txt",
                                        title='Save site selection',
                                        filetypes=(("text file", "*.txt"),),
                                        defaultextension='.txt')
        if f == None:
            return
        else:
            np.savetxt(f.name, TM.sites_selection, delimiter=',', fmt='%d')

    def cb_runTM(self):
        TM.N_site_selection = len(TM.sites_selection)

        path = tk.filedialog.askdirectory(initialdir='/', title='Dir to save TM run')
        if path == None:
            return
        else:
            TM.refsite_coor = TM.site_coor[np.argwhere(TM.site_ID, TM.refsite)]
            TM.site_coor_wref = TM.site_coor - TM.refsite_coor  # set wafer coordinate wrt to ref
            for i in range(TM.N_site_selection):
                TM.prober.go2site(TM.site_coor_wref[i])
                self.log_event('moved to abs {}'.format(TM.site_coor_wref[i, :]))

                self.acquire_IV()
                filename = TM.ID_w + '_' + str(TM.site_ID[TM.sites_selection[i]]) + '.txt'
                self.save_IV(os.path.join(path, filename), '', np.vstack((TM.volt, TM.curr[:, -1])).T)
                self.log_event('IV number of site {i} was measured\n'.format(i=i))

    def cb_acquireIV(self):
        self.acquire_IV()
        self.log_event('IV number: {i} measured\n'.format(i=TM.i_IV))

    def cb_chkbox_status_IR(self):
        TM.IRring = self.status_IR.get()
        self.log_event('IR Ring was set to: {}\n'.format(TM.IRring))
        if TM.IRring == 1:
            TM.IR.out_on()
        if TM.IRring == 0:
            TM.IR.out_off()

    def cb_setIR(self, event):
        self.intensity_IR = self.scale_setIR.get()
        TM.IR.setInt(self.intensity_IR)

    def cb_clear_graph(self):
        self.log_event('Plot cleared\n')
        self.ax1.lines = []
        self.fig1.canvas.draw()
        self.ax1.legend()

    def log_event(self, message):
        self.field_log.insert(tk.END, message)
        self.field_log.see(tk.END)

    def acquire_IV(self):
        TM.i_IV += 1
        I = 20e-12 * np.linspace(-1, 1, num=11) + 50e-12 * np.random.rand(11)
        if TM.i_IV == 1:
            TM.curr = I.reshape(-1, 1)
        else:
            TM.curr = np.append(TM.curr, I.reshape(-1, 1), axis=1)

        self.ax1.plot(TM.volt, I, marker='o', label=TM.i_IV)
        self.ax1.legend(loc='right')
        self.graph_IV.draw()

    def save_IV(self, filename, header, data):
        np.savetxt(filename, data, delimiter=",", header=header)


TM.gen_frontend()

# TM.instr_bye()

