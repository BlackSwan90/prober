import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from cycler import cycler


def cb_TMradio():
    TMmode = TMmode_radio.get()
    print(TMmode)

def cb_openIV():
    path_loadIV = tk.filedialog.askopenfile(initialdir='c:\py37_xtbox\playground\michael_hakl\tests\tinker',title='Open IV file')
    # path_loadIV = r'c:\py37_xtbox\playground\michael_hakl\tests\tinker\YFO_110Aulong.txt'
    loadIV = np.genfromtxt(path_loadIV,delimiter=",")[:,1:]
    ax1.plot(loadIV[:,0],loadIV[:,1], marker='o', color='black', linewidth=2)
    graph_IV.draw()

def cb_saveIV():
    f = tk.filedialog.asksaveasfile(mode='w',initialdir='/',initialfile=".txt", title='Save last IV')
    if f == None:
        return
    else:
        save_IV(f.name,'',d[])

def cb_get_ID_w(event):
    TM.ID_w = name_w_var.get()

def cb_select_site(event):
    site_sel = event.widget.find_closest(event.x, event.y)[0]
    
    if canvas_w.itemcget(site_sel, 'fill') == 'green':
        canvas_w.itemconfig(site_sel, fill='yellow')
        TM.site_TM.remove(site_sel)

    elif canvas_w.itemcget(site_sel, 'fill') == 'yellow':
        canvas_w.itemconfig(site_sel, fill='red')

    elif canvas_w.itemcget(site_sel, 'fill') == 'red':
        canvas_w.itemconfig(site_sel, fill='green')
        TM.site_TM.append(site_sel)

    TM.site_TM = sorted(TM.site_TM)

def cb_allsites():
    for i_site in range(d['N_site']):
        canvas_w.itemconfig(i_site+1, fill='green')
    TM.site_TM = list(range(1,d['N_site']+1,1))

def cb_nosite():
    for i_site in range(d['N_site']):
        canvas_w.itemconfig(i_site+1, fill='yellow')
    TM.site_TM = []

def cb_load_selection():

def cb_runTM():
    d['N_IV'] = len(TM.site_TM)
    d['curr'] = np.zeros((11,d['N_IV']))

    for i_site in range(d['N_IV']):
        acquire_IV(d)
        
        ID_w 
        log_event('IV number: {i} measured\n'.format(i=d['i_IV']))

def cb_acquireIV():
    acquire_IV(d)
    log_event('IV number: {i} measured\n'.format(i=d['i_IV']))

def cb_chkbox_status_IR():
    IRring = status_IR.get()
    log_event('IR Ring was set to: {val}\n'.format(val=IRring))

def cb_clear_graph():
    log_event('Plot cleared\n')
    ax1.lines = []
    fig1.canvas.draw()
    ax1.legend()

def log_event(message):
    field_log.insert(tk.END, message)
    field_log.see(tk.END)

def site_map(d):
    x_coor = (np.arange(11) - 5) *1.5e3
    y_coor = x_coor
    x_grid, y_grid = np.meshgrid(x_coor,y_coor)
    d['sites_x'] = np.ravel(x_grid)
    d['sites_y'] = np.ravel(y_grid)
    d['N_site'] = len(d['sites_x'])
    return d

def acquire_IV(d):
    d['i_IV'] += 1
    d['V'] = np.linspace(-3,3,num=11)
    I = 20e-12*np.linspace(-1,1,num=11) + 50e-12*np.random.rand(11)
    #IV = np.hstack((V.reshape(-1,1),I.reshape(-1,1)))
    ax1.plot(d['V'],I,marker='o',label=d['i_IV'])
    ax1.legend(loc='right')
    graph_IV.draw()
    return I

def save_IV(filename, header,data):
    np.savetxt(filename, data, delimiter=",", header=header)

class class_TM():

    def __init__(self):
        self.site_TM = []
        N_color = 10
        list_color = [plt.get_cmap('jet')(i / N_color) for i in range(N_color)]
        plt.rc('axes', prop_cycle=(cycler('color', list_color)), linewidth=1)
        plt.ioff()

        self.status_IR = 0
        self.ID_w = ''

def main():
    root = tk.Tk()
    root.wm_title("T&M with IV")

    frame1 = tk.Frame(root)
    frame1.grid(row=1, column=1, columnspan=2, sticky='',padx=20)
    TMmode_radio = tk.IntVar()
    TMmode_radio.set(1)
    tk.Label(frame1, text='Experiment:').grid(row=1, column=1, columnspan=2)
    tk.Radiobutton(frame1, text="Simple IV", variable=TMmode_radio, value=1, command=cb_TMradio).grid(row=2, column=1)
    tk.Radiobutton(frame1, text="Test structure", padx=5, variable=TMmode_radio, value=2, command=cb_TMradio).grid(row=2, column=2)

    tk.Label(frame1, text='Select structure:').grid(row=1, column=3, columnspan=2,padx=20)
    butt_allsites = tk.Button(frame1, text='all sites', command=cb_allsites)
    butt_allsites.grid(row=2, column=3)

    butt_nosite = tk.Button(frame1, text='no site', command=cb_nosite)
    butt_nosite.grid(row=2, column=4)

    tk.Label(frame1, text='Wafer ID:').grid(row=1, column=5)
    name_w_var = tk.StringVar()
    name_w = tk.Entry(frame1, text='wafer ID', textvariable=name_w_var)
    name_w.grid(row=2, column=5,padx=20)
    name_w.bind("<Leave>", cb_get_ID_w)

    tk.Label(frame1, text='Data:').grid(row=1, column=6, columnspan=2)
    butt_openIV = tk.Button(frame1, text='Load IV',command=cb_openIV)
    butt_openIV.grid(row=2, column=6)

    butt_saveIV = tk.Button(frame1, text='Save last IV', command=cb_saveIV)
    butt_saveIV.grid(row=2, column=7)

    butt_acquireIV = tk.Button(frame1, text='Get single IV', command=cb_acquireIV)
    butt_acquireIV.grid(row=2, column=8)

    butt_runTM = tk.Button(frame1, text='Run Selection', command=cb_runTM)
    butt_runTM.grid(row=2, column=9)

    status_IR = tk.IntVar()
    chkbox_status_IR = tk.Checkbutton(frame1, text="IR light", variable=status_IR,command=cb_chkbox_status_IR).grid(row=2,column=10,padx=20)

    butt_clear_graph = tk.Button(frame1, text='Clear plot', command=cb_clear_graph)
    butt_clear_graph.grid(row=2, column=11)

    frame2 = tk.Frame(root)
    frame2.grid(row=2, column=1)
    canvas_w = tk.Canvas(frame2, width=500, height=500, bd=2, relief='ridge')

    center_w = np.array([250, 250])
    unit_w = 500 / 30000
    sites_canvas_coor = np.hstack((d['sites_x'].reshape(-1, 1), d['sites_y'].reshape(-1, 1))) * unit_w + center_w.T

    site_canvas = []
    for i_site in range(d['N_site']):
        site_canvas.append(canvas_w.create_rectangle(sites_canvas_coor[i_site, 0], sites_canvas_coor[i_site, 1],
                                                     sites_canvas_coor[i_site, 0] + 20, sites_canvas_coor[i_site, 1] + 20,
                                                     outline='black', fill='yellow'))
        # canvas_w.create_text(sites_canvas_coor[i_site,0]+10, sites_canvas_coor[i_site,0]+10, text=i_site)
        canvas_w.tag_bind(site_canvas[i_site], '<Button-1>', cb_select_site)

    canvas_w.create_oval(10, 10, 500 - 10, 500 - 10, outline='black', width=2)  # wafer border
    canvas_w.grid(pady=20)

    frame3 = tk.Frame(root)
    frame3.grid(row=2, column=2,rowspan=2)
    frame3.grid_rowconfigure(0, weight=1)
    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel('V, volts')
    ax1.set_ylabel('I, amps')
    ax1.grid(True)
    graph_IV = FigureCanvasTkAgg(fig1, master=frame3)  # A tk.DrawingArea.
    toolbar = NavigationToolbar2Tk(graph_IV, frame3)
    toolbar.update()
    graph_IV.get_tk_widget().pack(pady=20,fill=tk.X)

    frame4 = tk.Frame(root)
    frame4.grid(row=3, column=1, columnspan=1, sticky='')
    tk.Label(frame4, text='Event Log:').grid(row=1, column=1,sticky='W')
    field_log = tk.Text(frame4, height=20, width=100)
    field_log.grid(row=2, column=1)

    root.mainloop()

d = {}
d['i_IV'] = 0
d = site_map(d)
TM = class_TM()

main()


