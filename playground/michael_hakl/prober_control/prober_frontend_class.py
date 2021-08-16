import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from cycler import cycler

def cb_TMradio():
    TMmode = TMmode_radio.get()
    print(TMmode)

def cb_openIV():
    path_loadIV = tk.filedialog.askopenfile(initialdir=r'c:\py37_xtbox\playground\michael_hakl\tests\tinker', title='Open IV file')
    # path_loadIV = r'c:\py37_xtbox\playground\michael_hakl\tests\tinker\YFO_110Aulong.txt'
    loadIV = np.genfromtxt(path_loadIV, delimiter=",")[:, 1:]
    ax1.plot(loadIV[:, 0], loadIV[:, 1], marker='o', color='black', linewidth=2)
    graph_IV.draw()

def cb_saveIV():
    f = tk.filedialog.asksaveasfile(mode='w', initialdir='/', initialfile=".txt", title='Save last IV')
    if f == None:
        return
    else:
        save_IV(f.name, '', np.vstack((TM.volt,TM.curr[:,-1])))

def cb_get_ID_w(event):
    TM.ID_w = name_w_var.get()

def cb_select_site(event):
    selection = event.widget.find_closest(event.x, event.y)[0] - 1

    if canvas_w.itemcget(selection, 'fill') == 'green':
        canvas_w.itemconfig(selection, fill='yellow')
        TM.sites_selection.remove(selection)

    elif canvas_w.itemcget(selection, 'fill') == 'yellow':
        canvas_w.itemconfig(selection, fill='green')
        TM.sites_selection.append(selection)

    TM.sites_selection = sorted(TM.sites_selection)

def cb_allsites():
    for i_site in range(TM.N_site):
        canvas_w.itemconfig(i_site + 1, fill='green')
    TM.sites_selection = list(range(1, TM.N_site + 1, 1))

def cb_nosite():
    for i_site in range(TM.N_site):
        canvas_w.itemconfig(i_site + 1, fill='yellow')
    TM.sites_selection = []

def cb_load_sites():
    path_load_sites = tk.filedialog.askopenfile(initialdir=r'c:\py37_xtbox\playground\michael_hakl\tests\tinker', title='Open sites')
    sites = np.genfromtxt(path_load_sites, delimiter=",")
    gen_site_map(sites)

def cb_runTM():
    TM.N_site_selection = len(TM.sites_selection)

    for i_site in range(TM.N_site_selection):
        acquire_IV()
        filename = TM.ID_w + '_' + str(TM.site_ID[TM.sites_selection[i_site]])
        save_IV(filename,'',np.vstack((TM.volt,TM.curr[:,-1])) )
        log_event('IV number of site {i} was measured\n'.format(i=i_site))

def cb_acquireIV():
    acquire_IV()
    log_event('IV number: {i} measured\n'.format(i=TM.i_IV))

def cb_chkbox_status_IR():
    TM.IRring = status_IR.get()
    log_event('IR Ring was set to: {}\n'.format(TM.IRring))

def cb_setIR(event):
    intensity_IR =  scale_setIR.get()
    print(intensity_IR)

def cb_clear_graph():
    log_event('Plot cleared\n')
    ax1.lines = []
    fig1.canvas.draw()
    ax1.legend()


def log_event(message):
    field_log.insert(tk.END, message)
    field_log.see(tk.END)

def gen_site_map(coor):
    x_grid, y_grid = np.meshgrid(coor[:,0],coor[:,1])
    sites_x = np.ravel(x_grid)
    sites_y = np.ravel(y_grid)
    TM.N_site = len(sites_x)
    TM.sites = [sites_x,sites_y]
    TM.site_ID = np.arange(TM.N_site)

def acquire_IV():
    TM.i_IV += 1
    I = 20e-12 * np.linspace(-1, 1, num=11) + 50e-12 * np.random.rand(11)
    if TM == 1:
        TM.curr = I.reshape(-1, 1)
    else:
        TM.curr = np.append(TM.curr, I.reshape(-1, 1), axis=1)

    ax1.plot(TM.volt, I, marker='o', label=TM.i_IV)
    ax1.legend(loc='right')
    graph_IV.draw()

def save_IV(filename, header, data):
    np.savetxt(filename, data, delimiter=",", header=header)


class class_TM():

    def __init__(self):
        self.sites_selection = []
        self.N_site_selection = 0
        N_color = 10
        list_color = [plt.get_cmap('jet')(i / N_color) for i in range(N_color)]
        plt.rc('axes', prop_cycle=(cycler('color', list_color)), linewidth=1)
        plt.ioff()

        self.IRring = 0
        self.ID_w = ''
        self.i_IV = 0

        self.N_sample_IV = 11
        self.volt = np.linspace(-3, 3, num=self.N_sample_IV)
        self.curr = np.zeros((self.N_sample_IV,1))


def main():
    root = tk.Tk()
    root.wm_title("T&M with IV")

    frame1 = tk.Frame(root)
    frame1.grid(row=1, column=1, columnspan=2, sticky='', padx=20)
    TMmode_radio = tk.IntVar()
    TMmode_radio.set(1)
    tk.Label(frame1, text='Experiment:').grid(row=1, column=1, columnspan=2)
    tk.Radiobutton(frame1, text="Simple IV", variable=TMmode_radio, value=1, command=cb_TMradio).grid(row=2, column=1)
    tk.Radiobutton(frame1, text="Test structure", padx=5, variable=TMmode_radio, value=2, command=cb_TMradio).grid(row=2, column=2)

    tk.Label(frame1, text='Select structure:').grid(row=1, column=3, columnspan=2, padx=20)
    butt_allsites = tk.Button(frame1, text='All sites', command=cb_allsites)
    butt_allsites.grid(row=2, column=3)

    butt_nosite = tk.Button(frame1, text='No site', command=cb_nosite)
    butt_nosite.grid(row=2, column=4)

    butt_nosite = tk.Button(frame1, text='Load sites', command=cb_load_sites)
    butt_nosite.grid(row=2, column=5)

    tk.Label(frame1, text='Wafer ID:').grid(row=1, column=5)
    name_w_var = tk.StringVar()
    name_w = tk.Entry(frame1, text='wafer ID', textvariable=name_w_var)
    name_w.grid(row=2, column=6, padx=20)
    name_w.bind("<Leave>", cb_get_ID_w)

    tk.Label(frame1, text='Data:').grid(row=1, column=6, columnspan=2)
    butt_openIV = tk.Button(frame1, text='Load IV', command=cb_openIV)
    butt_openIV.grid(row=2, column=7)

    butt_saveIV = tk.Button(frame1, text='Save last IV', command=cb_saveIV)
    butt_saveIV.grid(row=2, column=8)

    butt_acquireIV = tk.Button(frame1, text='Get single IV', command=cb_acquireIV)
    butt_acquireIV.grid(row=2, column=9)

    butt_runTM = tk.Button(frame1, text='Run Selection', command=cb_runTM)
    butt_runTM.grid(row=2, column=10)

    status_IR = tk.IntVar()
    chkbox_status_IR = tk.Checkbutton(frame1, text="IR light", variable=status_IR, command=cb_chkbox_status_IR).grid(row=2, column=11, padx=20)

    tk.Label(frame1,text='Intensity:').grid(row=3,column=10,sticky='se')
    scale_setIR = tk.Scale(frame1, from_=0, to=100, orient=tk.HORIZONTAL,command=cb_setIR)
    scale_setIR.grid(row=3,column=11)

    butt_clear_graph = tk.Button(frame1, text='Clear plot', command=cb_clear_graph)
    butt_clear_graph.grid(row=2, column=12)

    frame2 = tk.Frame(root)
    frame2.grid(row=2, column=1)
    canvas_w = tk.Canvas(frame2, width=500, height=500, bd=2, relief='ridge')

    center_w = np.array([250, 250])
    unit_w = 500 / 30000
    sites_canvas_coor = np.hstack((TM.sites[0].reshape(-1, 1), TM.sites[1].reshape(-1, 1))) * unit_w + center_w.T

    site_canvas = []
    for i_site in range(TM.N_site):
        site_canvas.append(canvas_w.create_rectangle(sites_canvas_coor[i_site, 0], sites_canvas_coor[i_site, 1],
                                                     sites_canvas_coor[i_site, 0] + 20, sites_canvas_coor[i_site, 1] + 20,
                                                     outline='black', fill='yellow'))
        # canvas_w.create_text(sites_canvas_coor[i_site,0]+10, sites_canvas_coor[i_site,0]+10, text=i_site)
        canvas_w.tag_bind(site_canvas[i_site], '<Button-1>', cb_select_site)

    canvas_w.create_oval(10, 10, 500 - 10, 500 - 10, outline='black', width=2)  # wafer border
    canvas_w.grid(pady=20)

    frame3 = tk.Frame(root)
    frame3.grid(row=2, column=2, rowspan=2,sticky='nsew')
    frame3.grid_rowconfigure(0, weight=1)
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('V, volts')
    ax1.set_ylabel('I, amps')
    ax1.grid(True)
    graph_IV = FigureCanvasTkAgg(fig1, master=frame3)  # A tk.DrawingArea.
    toolbar = NavigationToolbar2Tk(graph_IV, frame3)
    toolbar.update()
    graph_IV.get_tk_widget().pack(pady=20, fill='both',expand=True)

    frame4 = tk.Frame(root)
    frame4.grid(row=3, column=1, columnspan=1, sticky='')
    tk.Label(frame4, text='Event Log:').grid(row=1, column=1, sticky='W')
    field_log = tk.Text(frame4, height=20, width=70)
    field_log.grid(row=2, column=1)

    root.mainloop()

TM = class_TM()

coor = np.vstack(((np.arange(11) - 5) * 1.5e3, (np.arange(11) - 5) * 1.5e3)).T
gen_site_map(coor)


main()


