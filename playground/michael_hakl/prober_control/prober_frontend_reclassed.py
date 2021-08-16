import tkinter.filedialog
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from cycler import cycler
import colorsys
import os
import time
import pyvisa as visa
import pandas as pd

class TM_frontend():

    ############ generate app #########
    def __init__(self,TM):
        self.TM = TM
        self.TM.gen_site_map1()
        # self.TM.gen_wafer_map()
        self.generate_layout()

    def generate_layout(self):
        self.root = tk.Tk()
        self.root.wm_title("T&M with IV")
        self.generate_menu()
        self.generate_canvas_w()
        self.generate_IVgraph()
        self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)

    def generate_menu(self):
        self.frame1 = tk.Frame(self.root)
        self.frame1.grid(row=1, column=1, columnspan=2, sticky='', padx=20)

        self.TMmode_radio = tk.IntVar()
        self.TMmode_radio.set(2)
        tk.Label(self.frame1, text='Mode:').grid(row=1, column=1, columnspan=1)
        tk.Radiobutton(self.frame1, text="Oper", variable=self.TMmode_radio, value=1, command=self.cb_TMmode).grid(row=2, column=1)
        tk.Radiobutton(self.frame1, text="Simul",variable=self.TMmode_radio, value=2, command=self.cb_TMmode).grid(row=3, column=1)

        tk.Label(self.frame1, text='Wafer map:').grid(row=1, column=3, columnspan=5)
        self.butt_load_sites = tk.Button(self.frame1, text='Load map', command=self.cb_load_site_layout)
        self.butt_load_sites.grid(row=2, column=3)

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

        tk.Label(self.frame1, text='Ref ID:').grid(row=1, column=9)
        self.name_refsite_var = tk.StringVar()
        self.name_refsite = tk.Entry(self.frame1, textvariable=self.name_refsite_var)
        self.name_refsite.grid(row=2, column=9)
        self.name_refsite.bind("<Return>", self.cb_refsite_from_ID)

        self.toggle_select_ref = tk.IntVar()
        self.frame_ref = tk.Frame(self.frame1)
        self.frame_ref.grid(row=3, column=9, columnspan=1)
        tk.Label(self.frame_ref, text='Select manual').pack(side='left')
        self.chkbox_select_ref = tk.Checkbutton(self.frame_ref, variable=self.toggle_select_ref).pack(side='left')

        tk.Label(self.frame1, text='Data:').grid(row=1, column=10, columnspan=2)
        self.butt_loadIV = tk.Button(self.frame1, text='Load IV', command=self.cb_loadIV)
        self.butt_loadIV.grid(row=2, column=10)

        self.butt_saveIV = tk.Button(self.frame1, text='Save last IV', command=self.cb_saveIV)
        self.butt_saveIV.grid(row=2, column=11)

        tk.Label(self.frame1, text='Experiment').grid(row=1, column=12, columnspan=2)
        self.butt_acquireIV = tk.Button(self.frame1, text='Get single IV', command=self.cb_acquire_IV_single)
        self.butt_acquireIV.grid(row=2, column=12)

        self.butt_runTM = tk.Button(self.frame1, text='Run', command=self.cb_runTM)
        self.butt_runTM.grid(row=2, column=13)

        self.butt_stopTM = tk.Button(self.frame1, text='Stop', command=self.cb_stopTM)
        self.butt_stopTM.grid(row=3, column=13)

        self.status_IR = tk.IntVar()
        tk.Label(self.frame1, text='IR light').grid(row=1, column=14, columnspan=1, padx=20)
        self.chkbox_status_IR = tk.Checkbutton(self.frame1, variable=self.status_IR, command=self.cb_chkbox_status_IR).grid(row=2, column=14, padx=20)

        self.scale_setIR = tk.Scale(self.frame1, from_=0, to=100, orient=tk.HORIZONTAL, command=self.cb_setIR)
        self.scale_setIR.set(0)
        self.scale_setIR.grid(row=3, column=14)

        tk.Label(self.frame1, text='Plot').grid(row=1, column=15, columnspan=1)
        self.butt_clear_graph = tk.Button(self.frame1, text='Clear', command=self.cb_clear_graph)
        self.butt_clear_graph.grid(row=2, column=15)

        self.frame4 = tk.Frame(self.root)
        self.frame4.grid(row=3, column=1, columnspan=1)
        tk.Label(self.frame4, text='Event Log:').grid(row=1, column=1,sticky='w')
        self.field_log = tk.Text(self.frame4)
        self.field_log.grid(row=2, column=1,sticky='nsew')

    def generate_canvas_w(self):
        if self.TM.site_coor_R.any():

            self.frame2 = tk.Frame(self.root)
            self.frame2.grid(row=2, column=1)
            self.canvas_w = tk.Canvas(self.frame2, width=800, height=600, bd=2, relief='ridge')

            ext = 2 * np.max(np.hypot(self.TM.site_coor_R[:, 0], self.TM.site_coor_R[:, 1]))  # the most remote die from center defines display perimeter
            um2pt = 570 / (1.1 * ext)  # circle diam[pt]/wafer diam[um]
            center_w = np.array([300, 300])

            self.RGB_palette = ['blue', 'gold', 'chocolate1', 'light grey', 'maroon1', 'purple1', 'steel blue', 'snow', 'aquamarine', 'chartreuse4', 'sienna4']
            self.canvas_s_coor = self.TM.site_coor_R * um2pt
            self.canvas_s_coor[:, 1] = -self.canvas_s_coor[:, 1]
            self.canvas_s_coor = self.canvas_s_coor + center_w
            site_canvas = []
            for i in range(self.TM.N_site):
                # c = '#{r:02x}{g:02x}{b:02x}'.format(r=int(RGB_tuples[self.TM.site_type[i]][0] * 255),
                #                                     g=int(RGB_tuples[self.TM.site_type[i]][1] * 255),
                #                                     b=int(RGB_tuples[self.TM.site_type[i]][2] * 255))

                i_c = np.where(self.TM.type_groups == self.TM.site_type[i])[0][0]
                c = self.RGB_palette[i_c]

                site_canvas.append(self.canvas_w.create_rectangle(self.canvas_s_coor[i, 0] - 6, self.canvas_s_coor[i, 1] - 6,
                                                                  self.canvas_s_coor[i, 0] + 6, self.canvas_s_coor[i, 1] + 6,
                                                                  outline=c, fill='yellow', width=3))  # y coor of canvas is positive downwards

                # canvas_w.create_text(sites_canvas_coor[i_site,0]+10, sites_canvas_coor[i_site,0]+10, text=i_site)
                # self.canvas_w.tag_bind(site_canvas[i], '<Button-1>', self.cb_select_1site)
            self.canvas_w.bind('<Button-1>', self.cb_select_1site)
            self.canvas_w.bind('<ButtonRelease-1>', self.cb_drag_select)

            self.canvas_w.create_oval(10, 10, 600 - 10, 600 - 10, outline='black', width=2)  # wafer border

            for i_g, g in enumerate(self.TM.type_groups):
                wg_type_group = self.canvas_w.create_rectangle(610, 10 + i_g * 12 + i_g * 10,
                                                               622, 10 + (i_g + 1) * 12 + i_g * 10,
                                                               outline=self.RGB_palette[i_g], fill='light grey', width=3, tags=(g,))
                self.canvas_w.tag_bind(wg_type_group, '<Button-1>', self.cb_select_type_group)
                self.canvas_w.create_text(630, 10 + i_g * 12 + i_g * 10 + 6, font='Arial', text=g, anchor='w')

            self.canvas_w.grid()

        else:
            return

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

    ############## callbacks ############

    def cb_TMmode(self):
        self.TM.mode = self.TMmode_radio.get()
        self.log_event('TM mode was set as {}\n'.format(self.TM.mode))

    def cb_loadIV(self):
        f = tk.filedialog.askopenfile(initialdir=self.TM.path_default_dir, title='Open IV file')
        if f == None:
            return
        else:
            IV = np.genfromtxt(f.name, delimiter=",")
            label = os.path.basename(f.name).split('.')[0]
            self.plot_update(IV,label)

    def cb_saveIV(self):
        f = tk.filedialog.asksaveasfile(mode='w',
                                        initialdir=os.path.join(self.TM.path_default_dir,'data'),
                                        title='Save last IV',
                                        filetypes=(("text file", "*.txt"),),
                                        defaultextension='.txt')
        if f == None:
            return
        else:
            self.TM.save_IV(f.name, '', np.hstack((self.TM.V_m[:,-1], self.TM.I_m[:, -1])) )

    def cb_get_ID_w(self, event):
        self.TM.ID_w = self.name_w_var.get()
        self.log_event('wafer ID set as: {}\n'.format(self.TM.ID_w))

    def cb_refsite_from_ID(self, event):
        self.TM.refsite_ID = self.name_refsite.get()
        self.TM.refsite_ind = np.argwhere(self.TM.site_ID == self.TM.refsite_ID)
        self.log_event('Ref site set as: {}\n'.format(self.TM.refsite_ID))

    def cb_select_1site(self, event):
        self.select_origin = [self.canvas_w.canvasx(event.x), self.canvas_w.canvasy(event.y)]

    def cb_drag_select(self, event):
        #find encircled dies
        self.select_endpoint = [self.canvas_w.canvasx(event.x), self.canvas_w.canvasy(event.y)]
        coor = self.select_origin + self.select_endpoint
        selection = event.widget.find_overlapping(*coor)

        # case for selecting reference die
        if self.toggle_select_ref.get() == 1:
            if len(selection) == 1:
                self.select_ref_manual(selection)

        # case for selecting dies to measure
        if self.toggle_select_ref.get() == 0:
                if len(selection) == 1: # selecting only 1 die
                    if self.canvas_w.itemcget(selection[0], 'fill') == 'green':
                        self.canvas_w.itemconfig(selection[0], fill='yellow')
                        self.TM.site_selection.remove(selection[0] - 1)

                    else:
                        self.fill_green_1click(selection[0])
                        self.TM.site_selection.append(selection[0] - 1)

                elif len(selection) > 1: # selecting multiple dies, green greedy
                    for site in selection:
                            self.fill_green_1click(site)
                            self.TM.site_selection.append(site - 1)

        #update canvas & selection
        self.canvas_w.update()
        self.TM.site_selection = sorted(self.TM.site_selection)

    def cb_allsites(self):
        for i in range(self.TM.N_site):
            self.fill_green_1click(i+1)
        self.canvas_w.update()
        self.TM.site_selection = list(range(0, self.TM.N_site, 1))

    def cb_nosite(self):
        for i in range(self.TM.N_site):
            self.canvas_w.itemconfig(i + 1, fill='yellow')
        self.canvas_w.update()
        self.TM.site_selection = []

    def cb_load_site_layout(self):
        f = tk.filedialog.askopenfile(initialdir=self.TM.path_default_dir, title='Open sites')
        if f == None:
            return
        else:
            # read sites + coors
            self.sites = pd.read_csv(f.name, skiprows=[0, 1]).to_numpy()

            # read theta
            with open(f.name) as ff:
                first_line_theta = ff.readline()
                self.TM.theta = float(first_line_theta.split('=')[1].strip()) / 180 * np.pi

            # cols in file: ID, type, x, y
            self.TM.site_ID = self.sites[:, 0].astype(dtype='str')
            self.TM.site_type = self.sites[:, 1].astype(dtype='str')
            self.TM.site_coor = self.sites[:, 2:].astype(dtype='float')
            self.TM.type_groups = np.unique(self.TM.site_type)
            self.TM.N_site = len(self.TM.site_ID)

            #rotate coor
            R_theta = np.array([[np.cos(self.TM.theta), -np.sin(self.TM.theta)], [np.sin(self.TM.theta), np.cos(self.TM.theta)]])
            self.TM.site_coor_R = np.dot(R_theta, self.TM.site_coor.T).T

            #init canvas
            self.TM.refsite_ind = -1
            self.TM.site_selection = []
            self.generate_canvas_w()

    def cb_load_site_selection(self):
        f = tk.filedialog.askopenfile(initialdir=self.TM.path_default_dir,
                                      title='Open site selection')
        if f == None:
            return
        else:
            self.cb_nosite() #reset selection

            self.TM.site_selection = (np.genfromtxt(f.name, delimiter=",").astype(dtype='int')).tolist()

            for site in self.TM.site_selection:
                self.canvas_w.itemconfig(site + 1, fill='green')

    def cb_save_site_selection(self):
        f = tk.filedialog.asksaveasfile(mode='w', initialdir=self.TM.path_default_dir,
                                        title='Save site selection',
                                        filetypes=(("text file", "*.txt"),),
                                        defaultextension='.txt')
        if f == None:
            return
        else:
            np.savetxt(f.name, self.TM.site_selection, delimiter=',', fmt='%d')

    def cb_select_type_group(self, event):
        type = event.widget.find_closest(event.x, event.y)[0]
        self.TM.select_by_type = self.canvas_w.gettags(type)[0]
        selection_doublelist = np.argwhere(self.TM.site_type == self.TM.select_by_type)

        #flatten list
        self.TM.selection_by_type = []
        for sublist in selection_doublelist:
            for item in sublist:
                self.TM.selection_by_type.append(item)

        for site in self.TM.selection_by_type:
            self.fill_green_1click(site+1)

        self.TM.site_selection = self.TM.site_selection + self.TM.selection_by_type #update actual selection
        self.TM.site_selection = sorted(self.TM.site_selection)

        self.log_event('Group of sites: {} was selected'.format(self.TM.select_by_type))

    def cb_runTM(self):

        # path = tk.filedialog.askdirectory(initialdir=self.TM.path_default_dir,title='Dir to save TM run')
        path = os.path.join(self.TM.path_default_dir,'data')
        if path == None:
            return
        else:

            self.TM.busy = 1  # make setup busy
            self.TM.N_site_selection = len(self.TM.site_selection)

            # sort selected sites into type groups
            self.TM.site_selection_type = self.TM.site_type[self.TM.site_selection]
            self.TM.type_groups_selection = np.unique(self.TM.site_selection_type)
            self.TM.site_groupby_type = []
            for g in self.TM.type_groups_selection:
                self.TM.site_groupby_type.append(np.array(self.TM.site_selection)[self.TM.site_selection_type == g])

            #calculate coor
            self.TM.refsite_coor = self.TM.site_coor_R[np.argwhere(self.TM.site_ID == self.TM.refsite_ID)][0][0] #coor of ref on theta-rotated wafer
            site_difference = self.TM.site_coor_R - self.TM.refsite_coor  # get move between ref and sites

            if self.TM.mode == 1:
                refsite_coor_align = self.TM.prober.read_coor()[:2]  # read x,y of chuck at the alignment position
            else:
                refsite_coor_align = np.array([0,0])

            self.TM.site_coor_R_wref = refsite_coor_align + site_difference  # chuck coor: alignment + move

            for i_g, g in enumerate(self.TM.site_groupby_type):
                for s in g:
                    if self.TM.busy == 1:#check if stopped was pushed

                        if self.TM.mode == 1:
                            self.TM.prober.go2site(self.TM.site_coor_R_wref[s])
                        self.canvas_w.itemconfig(s + 1, width=5)  # highlight current die
                        self.canvas_w.update()
                        self.log_event('move to x, y: {}\n'.format(self.TM.site_coor_R_wref[s, :]))

                        if self.TM.mode == 1:
                            IV = self.acquire_IV(self.TM.site_type[s])
                        elif self.TM.mode == 2:
                            IV = self.TM.acquire_IV_random()
                        self.plot_update(IV,'{x},{y}'.format(x=int(self.TM.site_coor_R[s, 0]), y=int(self.TM.site_coor_R[s, 1])))

                        filename = self.TM.ID_w + '_' + str(self.TM.site_ID[s]) + '.txt'
                        header = '{},'.format(self.TM.site_type[s]) + '{x},{y}\n'.format(x=self.TM.site_coor[s, 0], y=self.TM.site_coor[s, 1])
                        header += 'V[V], I[A]'
                        self.TM.save_IV(os.path.join(path, filename), header, IV)

                        # unhighlight current die + mark as measured
                        self.canvas_w.itemconfig(s + 1, fill='cyan', width=3)
                        if s == self.TM.refsite_ind: #ref die unchanged
                            self.canvas_w.itemconfig(s + 1, fill='cyan', width=5)

                        self.log_event('IV was measured of site: {i}, type: {t}\n'.format(i=s, t=self.TM.type_groups_selection[i_g]))
                        self.canvas_w.update()

                    else: #measurement stopped
                        if self.TM.mode == 1:
                            self.TM.prober.separate()
                        return

            if self.TM.mode == 1:
                self.TM.prober.separate()
            self.log_event('Measurement finished')

    def cb_stopTM(self):
        self.TM.busy = 0
        self.log_event('Measurement stopped by user\n')

    def cb_acquire_IV_single(self):
        if self.TM.mode == 1:
            IV = self.TM.acquire_IV_single()
        elif self.TM.mode == 2:
            IV = self.TM.acquire_IV_random()

        self.plot_update(IV,self.TM.i_IV)
        self.log_event('IV number: {i} measured\n'.format(i=self.TM.i_IV))

    def cb_chkbox_status_IR(self):
        self.TM.IRring = self.status_IR.get()

        if self.TM.mode == 1:
            if self.TM.IRring == 1:
                self.TM.IR.out_on()
            elif self.TM.IRring == 0:
                self.TM.IR.out_off()

        self.log_event('IR Ring was set to: {}\n'.format(self.TM.IRring))

    def cb_setIR(self, event):
        self.intensity_IR = self.scale_setIR.get()
        if self.TM.mode == 1:
            self.TM.IR.setInt(self.intensity_IR)

    def cb_clear_graph(self):
        self.log_event('Plot cleared\n')
        self.ax1.lines = []
        self.ax1.legend([])
        self.fig1.canvas.draw()

    ######### service func ########

    def safe_exit(self):
        print('app exited')
        # instr_byebye(self.TM.kei,self.TM.prober,self.TM.IR)
        time.sleep(0.3)
        self.root.destroy()

    def log_event(self, message):
        self.field_log.insert(tk.END, message)
        self.field_log.see(tk.END)

    def plot_update(self,IV,label):
        self.ax1.plot(IV[:, 0], IV[:, 1], marker='o', linewidth=2,label=label)
        self.graph_IV.draw()

    def select_ref_manual(self,selection):
        # recover site type to restore the color, if reference selected for 2nd time
        if self.TM.refsite_ind != -1:
            i_c = np.where(self.TM.type_groups == self.TM.site_type[self.TM.refsite_ind])[0][0]
            c = self.RGB_palette[i_c]
            self.canvas_w.itemconfig(self.TM.refsite_ind + 1, outline=c, width=3)

        # update refsite
        self.canvas_w.itemconfig(selection[0], outline='red', width=5)
        self.TM.refsite_ind = selection[0] - 1
        self.TM.refsite_ID = self.TM.site_ID[self.TM.refsite_ind]
        self.name_refsite_var.set(self.TM.refsite_ID)
        self.log_event('Ref site was set as: {}\n'.format(self.TM.refsite_ID))

    def fill_green_1click(self,ind_widget):
        if self.canvas_w.itemcget(ind_widget, 'fill') == 'yellow' or self.canvas_w.itemcget(ind_widget, 'fill') == 'cyan':
            self.canvas_w.itemconfig(ind_widget, fill='green')

class TM_session():
    def __init__(self):
        # self.path_default_dir = r'C:\PythonProductionTools\py37_xtbox\playground\michael_hakl3\prober_control\data'
        self.path_default_dir = r'C:\py37_xtbox\playground\michael_hakl\prober_control'

        self.mode = 2
        self.busy = 0
        self.ID_w = ''
        self.theta = 0  # CCW is positive
        self.site_selection = []
        self.N_site_selection = 0
        self.refsite_ind = -1

        self.i_IV = 0

    def instr_ready(self,kei,prober,IR):
        self.kei = kei

        self.prober = prober

        self.IR = IR
        self.IRring = 0

    ###### map generation ######

    def gen_site_map(self):
        # cols in file: ID, type, x, y
        # self.site_ID = np.array([],dtype='str')
        # self.site_type = np.array([],dtype='str')
        # self.type_groups = np.unique(self.site_type)
        self.site_coor = np.zeros(0)
        # R_theta = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        # self.site_coor_R = np.dot(R_theta, self.site_coor.T).T
        self.site_coor_R = np.zeros(0)
        # self.N_site = len(self.site_ID)
        # self.refsite_ind = -1
        # self.site_selection = []

    def gen_site_map1(self):

        f = os.path.join(self.path_default_dir,'xfp01_layout.csv')

        # read sites + coors
        self.sites = pd.read_csv(f, skiprows=[0, 1]).to_numpy()

        # read theta
        with open(f) as ff:
            first_line_theta = ff.readline()
            self.theta = float(first_line_theta.split('=')[1].strip()) / 180 * np.pi

        # cols in file: ID, type, x, y
        self.site_ID = self.sites[:, 0].astype(dtype='str')
        self.site_type = self.sites[:, 1].astype(dtype='str')
        self.type_groups = np.unique(self.site_type)
        self.site_coor = self.sites[:, 2:].astype(dtype='float')
        R_theta = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        self.site_coor_R = np.dot(R_theta, self.site_coor.T).T
        self.N_site = len(self.site_ID)
        self.refsite_ind = -1
        self.site_selection = []

    def gen_site_map2(self):
        coor = np.vstack(((np.arange(11) - 5) * 1.5e3, (np.arange(11) - 5) * 1.5e3)).T
        x_grid, y_grid = np.meshgrid(coor[:, 0], coor[:, 1])
        sites_x = np.ravel(x_grid)
        sites_y = np.ravel(y_grid)
        self.N_site = len(sites_x)
        self.site_coor = np.vstack((sites_x, sites_y)).T
        self.site_ID = np.arange(self.N_site)
        self.site_type = np.ravel([np.ones(40), 2 * np.ones(40), 3 * np.ones(40)], order='F')
        self.site_type = np.hstack((self.site_type, np.array([1]))).astype(dtype='int')
        self.type_groups = np.unique(self.site_type)
        header = '# ID,type,coor_x[um], coor_y[um] \n'
        path = r'C:\PythonProductionTools\py37_xtbox\playground\michael_hakl3\prober_control\sites_on_grid.txt'
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

        self.site_coor_R = np.zeros((self.N_site, 2))
        i_s = 0
        for i_r in range(self.N_r):
            for i_c in range(self.N_c):
                if self.ind_map[self.N_r - 1 - i_r, i_c] == 1:
                    self.site_coor_R[i_s, 0] = i_c * self.die_period[0] - 6.5 * self.die_period[0]  # x
                    self.site_coor_R[i_s, 1] = i_r * self.die_period[1] - 8 * self.die_period[1]  # y
                    i_s += 1

        self.site_ID = np.arange(self.N_site)
        self.site_type = np.ones(self.N_site, dtype='int')
        self.type_groups = np.unique(self.site_type)
        header = '# ID,type,coor_x[um], coor_y[um] \n'
        path = r'C:\py37_xtbox\playground\michael_hakl\prober_control\3506_map.txt'
        f = open(path, 'w')
        f.write(header)
        for i in range(self.N_site):
            f.write('{s_ID},\t {s_type},\t {x},\t {y} \n'.format(s_ID=self.site_ID[i],
                                                                 s_type=int(self.site_type[i]),
                                                                 x=self.site_coor_R[i, 0],
                                                                 y=self.site_coor_R[i, 1]))
        f.close()

        # plt.ion()
        # plt.scatter(x=self.site_coor_R[:,0],y=self.site_coor_R[:,1])
        # plt.show()

    ##### experiment ########

    def acquire_IV_by_type(self, type):
        if type == 'LargeDiodes':
            IV = self.measure_LargeDiodes()
            return IV
        elif type == 'UbmMetalRes':
            IV = self.measure_UbmMetalRes()
            return IV

    def acquire_IV_single(self):

        V_s = np.linspace(-0.5, -0.1, num=3)
        self.N_sample_IV = len(V_s)

        IV = self.kei.measure_iv(V_s)

        if self.i_IV == 0:
            self.V_m = IV[:, 0].reshape(-1,1)
            self.I_m = IV[:, 1].reshape(-1,1)
        else:
            np.append(self.V_m,IV[:,0].reshape(-1,1),axis=1)
            np.append(self.I_m,IV[:,1].reshape(-1,1),axis=1)

        self.i_IV += 1
        return IV

    def acquire_IV_random(self):

        V_s = np.linspace(-1, 0, num=11)
        self.N_sample_IV = len(V_s)

        I = 20e-12 * np.linspace(-1, 1, num=self.N_sample_IV) + 50e-12 * np.random.rand(11)
        IV = np.vstack((V_s,I)).T

        if self.i_IV == 0:
            self.V_m = IV[:, 0].reshape(-1,1)
            self.I_m = IV[:, 1].reshape(-1,1)
        else:
            np.append(self.V_m,IV[:,0].reshape(-1,1),axis=1)
            np.append(self.I_m,IV[:,1].reshape(-1,1),axis=1)

        self.i_IV += 1

        return IV

    def measure_UbmMetalRes(self):
        V_s = np.array([-1, -0.5, -0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1])
        # V = np.array([-1, -0.5, -0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]).reshape(-1, 1)
        IV = self.kei.measure_iv(V_s)
        return IV

    def measure_LargeDiodes(self):
        V_s = np.array([-0.3, -0.2, -0.1])
        # V_s = np.array([-2,-1,-0.8,-0.6,-0.4,-0.3,-0.2,-0.1,0,0.1])
        # V = np.array([-15, -12, -10, -8, -6, -5, -4, -3, -2, -1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6]).reshape(-1, 1)
        IV = self.kei.measure_iv(V_s)
        return IV

    def acquire_LIV(self):
        L = np.array([0, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6]).reshape(-1, 1)
        V_s = np.array([-0.4, -0.3, -0.2, -0.1, 0]).reshape(-1, 1)
        self.V_m = np.zeros((V_s.shape[1], L.shape[1]))
        self.I_m = np.zeros((V_s.shape[1], L.shape[1]))

        for i in range(L.shape[1]):
            self.IR.setV(L[i])
            IV = self.kei.measure_iv(V_s)
            self.V_m[:, i] = IV[:, 0]
            self.I_m[:, i] = IV[:, 1]

        self.IR.setV(0)

    ###### service func #######

    def plot_IV(self,V_m,I_m):
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('V, volts')
        ax.set_ylabel('I, amps')
        ax.grid(True)

        for i in I_m.shape[1]:
            plt.plot(V_m[:,i], self.I_m[:,i], marker='o', label=str(i))

        plt.show()

    def save_IV(self, filename, header, data):
        np.savetxt(filename, data, delimiter=",", header=header)

class Keithley2600():

    def __init__(self):
        pass

    def instr_open(self, RM):
        t_measure = 0
        t_stabil = 1

        self.kei_ctrl = RM.open_resource("TCPIP0::192.168.104.102::inst0::INSTR", timeout=10000)  # node 1
        time.sleep(1)

        self.kei_ctrl.write("reset()")
        self.kei_ctrl.write("smua.measure.autozero = smua.AUTOZERO_AUTO")
        self.kei_ctrl.write("smua.measure.autorangei = smua.AUTORANGE_ON")
        self.kei_ctrl.write("smua.measure.autorangev = smua.AUTORANGE_ON")
        self.kei_ctrl.write("smua.measure.analogfilter = 1")
        self.kei_ctrl.write("smua.measure.filter.enable = 0")
        # kei_ctrl.write("smua.measure.delay = smua.DELAY_AUTO")
        # self.kei_ctrl.write("smua.measure.delay = {t}".format(t=t_stabil))
        # kei_ctrl.write("smua.measure.delayfactor = 1")
        # self.kei_ctrl.write("smua.measure.interval = {t}".format(t=t_measure))
        # self.kei_ctrl.write("smua.measure.nplc = 1")

        self.kei_ctrl.write("smua.sense = smua.SENSE_REMOTE")  # 4 wire
        # kei_ctrl.write("smua.sense = smua.SENSE_LOCAL") # 2 wire
        self.kei_ctrl.write("display.smua.measure.func = display.MEASURE_DCAMPS")

        self.kei_ctrl.write("smua.source.limiti = 10.0e-3")
        # self.kei_ctrl.write("smua.source.settling = 1")
        self.kei_ctrl.write("smua.source.levelv=0")
        self.kei_ctrl.write('smua.source.output=smua.OUTPUT_OFF')

    def instr_close(self):
        self.kei_ctrl.write('smua.source.output=smua.OUTPUT_OFF')
        time.sleep(0.1)
        self.kei_ctrl.close()

    def measure_iv(self, volt):
        self.kei_ctrl.write('smua.source.output=smua.OUTPUT_ON')
        N_sample_IV = len(volt)

        v_m_list = []
        i_m_list = []

        self.kei_ctrl.write("smua.source.levelv={V}".format(V=volt[0]))
        time.sleep(0.1)

        self.kei_ctrl.write("vval=smua.measure.v()")
        self.kei_ctrl.write("print(vval)")
        print('applying ', str(volt[0]), 'V')
        v_val = float(self.kei_ctrl.read_raw())
        self.kei_ctrl.write("ival=smua.measure.i()")
        self.kei_ctrl.write("print(ival)")
        i_val = float(self.kei_ctrl.read_raw())

        v_m_list.append(v_val)
        i_m_list.append(i_val)

        for ind_v in list(range(1, N_sample_IV, 1)):
            print('applying ', str(volt[ind_v]), 'V')
            self.kei_ctrl.write("smua.source.levelv={V}".format(V=volt[ind_v]))

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
        IV = np.vstack((v_arr, i_arr)).T

        return IV

class Cascade13k():
    def __init__(self):
        pass

    def instr_open(self, RM):
        self.prober_ctrl = RM.open_resource("GPIB0::1::INSTR", timeout=10000, send_end=True)  # ,read_termination='r',write_termination='r')
        time.sleep(1)

        # activate contact mode
        # print(self.prober_ctrl.query(":set:contact:active?"))
        self.prober_ctrl.query(":set:contact:active ON")
        # print(self.prober_ctrl.query(":set:cont? 2"))

    def instr_close(self):
        self.prober_ctrl.close()

    def read_coor(self):
        resp = self.prober_ctrl.query(":mov:abs? 2")
        print(resp)
        resp = resp.strip()
        resp = resp.split(' ')
        resp = np.array(resp, dtype=int)
        return resp

    def move_xy(self, coor):
        self.prober_ctrl.query(":mov:abs 2 {x} {y} none".format(x=coor[0], y=coor[1]))  # prober has y-axis positive to door
        # self.prober_ctrl.query(":mov:abs? 2")

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
        self.calib = np.genfromtxt(r'C:\PythonProductionTools\py37_xtbox\playground\michael_hakl3\prober_control\IR_calib.txt')

    def instr_open(self, RM):
        self.PWS4323_ctrl = RM.open_resource("USB0::0x0699::0x0391::C022000::INSTR", timeout=2000)

        # init
        self.PWS4323_ctrl.write("*RST")
        self.PWS4323_ctrl.write('SYSTEM:REMOTE')
        self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')
        self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL 0V')
        self.PWS4323_ctrl.write('SOURCE:CURRENT:LEVEL 2mA')
        self.PWS4323_ctrl.write('SOURCE:VOLTAGE:RANGE 32.1V')

    def instr_close(self):
        self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')
        self.PWS4323_ctrl.close()

    def out_on(self):
        self.PWS4323_ctrl.write('SOURCE:OUTPUT ON')

    def out_off(self):
        self.PWS4323_ctrl.write('SOURCE:OUTPUT OFF')

    def setInt(self, val):
        # max 20mA tot ~ rad. power  8x0.8mW, watch for eye safety!
        # for voltage 6V, I_tot = 1.2mA is safe and corresponds to slider=100
        self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL {val}V'.format(val=self.calib[val, 1]))
        time.sleep(0.2)

    def setV(self, V_val):
        self.PWS4323_ctrl.write('SOURCE:VOLTAGE:LEVEL {val}V'.format(val=V_val))
        time.sleep(0.2)

def instr_wakeup():
        RM = visa.ResourceManager()

        kei = Keithley2600()
        kei.instr_open(RM)

        prober = Cascade13k()
        prober.instr_open(RM)

        IR = RingSource()
        IR.instr_open(RM)

        return (kei,prober,IR)

def instr_byebye(kei,prober,IR):
    kei.instr_close()
    prober.instr_close()
    IR.instr_close()

def main():
    TM = TM_session()
    TM.mode = 2 # 1=operation, 2=simulation mode

    if TM.mode == 1:
        (kei,prober,IR) = instr_wakeup()
        TM.instr_ready()

    app = TM_frontend(TM)
    app.root.mainloop()

    if TM.mode == 1:
        instr_byebye(kei,prober,IR)

def test_setup():
    V = np.array([-0.5, -0.3])
    # IV = TM.kei.measure_iv(V)
    # TM.kei.instr_close()
    #
    # TM.prober.prober_ctrl.query(":set:cont? 2")
    #
    # TM.prober.move_xy(np.array([200,0]))
    # a = TM.prober.prober_ctrl.query(":mov:abs 2 {x} {y} none".format(x=0,y=0))
    # TM.prober.prober_ctrl.query(":mov:abs? 2")
    # TM.prober.read_coor()
    # TM.prober.separate()
    # TM.prober.touchdown()
    # TM.prober.go2site([5125+200,-27117])
    # TM.prober.instr_close()

    #plt.ion()
    # plt.plot(app.TM.site_coor[app.TM.site_selection][:,0],app.TM.site_coor[app.TM.site_selection][:,1])
    # plt.show()

