#!/usr/bin/env python


import os
import sys
import time
import math
from tkinter import filedialog
from tkinter import *

from optparse import OptionParser

import numpy as np
from scipy.optimize import leastsq

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar
from matplotlib.ticker import ScalarFormatter

from astropy import units as u
from astropy.coordinates import SkyCoord

import parfile
import presto.bestprof as bestprof
import utils

DEG2RAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG    = float('57.295779513082320876798154814105170332405472466564')
C          = float('2.99792458e8')

full_usage = """
usage : fitorbit.py [options] [*.bestprof]

  [-h, --help]        : Display this help

  Bestprof files as produced by Presto

"""

usage = "usage: %prog [options]"


PARAMS = ['RA', 'DEC', 'P0', 'P1', 'PEPOCH', 'PB', 'ECC', 'A1', 'T0', 'OM']

class Param:
    def __init__(self, is_string=False):
        self.val = 0.0
        if is_string:
            self.val = "00:00:00.0"
        self.fit = 0

class Data:
    def __init__(self):
        self.mjds = []
        self.period = []
        self.unc = []

    def add(self, mjd1, period1, ptype='ms'):
        self.mjds.append(mjd1)
        if  ptype=='ms':
            self.period.append(period1)
        elif  ptype=='s':
            self.period.append(period1*1000.)

    def set_mjd(self, mjd):
        self.mjds = mjd

    def set_period(self, period):
        self.period = period

    def set_unc(self, uncertainties):
        self.unc = uncertainties

    def get_mjd(self):
        return self.mjds

    def get_period(self):
        return self.period

    def get_unc(self):
        return self.unc

# Function to calc the expected period at a time x (in MJD) given the parameters
def calc_period(x, DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC):

    if PB:

        k1 = 2*np.pi*A1/(PB*86400.0*np.sqrt(1-ECC*ECC))

        # Calc easc in rad
        easc = 2*np.arctan(np.sqrt((1-ECC)/(1+ECC)) * np.tan(-OM*DEG2RAD/2))
        #print easc
        epperias = T0 - PB/360.0*(RAD2DEG * easc - RAD2DEG * ECC * np.sin(easc))
        #print x,epperias
        mean_anom = 360*(x-epperias)/PB
        mean_anom = np.fmod(mean_anom,360.0)
        #if mean_anom<360.0:
        #  mean_anom+=360.0
        mean_anom = np.where(np.greater(mean_anom, 360.0), mean_anom-360.0, mean_anom)

        # Return ecc_anom (in rad) by iteration
        ecc_anom = utils.eccentric_anomaly(ECC, mean_anom*DEG2RAD)

        # Return true anomaly in deg
        true_anom = 2*RAD2DEG*np.arctan(np.sqrt((1+ECC)/(1-ECC))*np.tan(ecc_anom/2))

        #print "easc=%f  epperias=%f  mean_anom=%f  ecc_anom=%f  true_anom=%f"%(easc,epperias,mean_anom,ecc_anom,true_anom)
        #sys.exit()

        #print RA, DEC
        #dv = utils.deltav(x, RA, DEC, RA-DRA, DEC-DDEC, 2000.0)
        #print dv

        return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) )
        return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*true_anom)+ECC*np.cos(OM) - np.sin(true_anom)*np.sin(OM))
    #return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM) + k1*ECC*np.cos(OM)) ) * (1-dv/3e8)
    #return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400) * (1+k1*np.cos(DEG2RAD*(true_anom+OM)) ) * (1-20000/C)
    else:
        return 1000*(P0+P1*1e-15*(x-PEPOCH)*86400)

# Function to calc Period residual y-f(x,...)
def timing_fcn(param, Pobs, x, fit, fixed_values):
    """
    param : value of the M parameters to fit
    Pobs : array of the f(x) values
    x : array of the x values

    fit : Array of N parameters which indicate the M parameters to fit
    fixed_values : values of the fixed parameters
    """

    nb_fit=0


    # DRA
    if fit[0]!=0:
        nb_fit+=1
    DRA = 0.0

    # DDEC
    if fit[1]!=0:
        nb_fit+=1
    DDEC = 0.0

    # P0
    if fit[2]!=0:
        P0 = param[nb_fit]
        nb_fit+=1
    else:
        P0 = fixed_values[2]

    # P1
    if fit[3]!=0:
        P1 = param[nb_fit]
        nb_fit +=1
    else:
        P1 = fixed_values[3]

    # PEPOCH
    if fit[4]!=0:
        PEPOCH = param[nb_fit]
        nb_fit +=1
    else:
        PEPOCH = fixed_values[4]

    # PB
    if fit[5]!=0:
        PB = param[nb_fit]
        nb_fit +=1
    else:
        PB = fixed_values[5]

    # ECC
    if fit[6]!=0:
        ECC = param[nb_fit]
        nb_fit +=1
    else:
        ECC = fixed_values[6]

    # A1
    if fit[7]!=0:
        A1 = param[nb_fit]
        nb_fit +=1
    else:
        A1 = fixed_values[7]

    # T0
    if fit[8]!=0:
        T0 = param[nb_fit]
        nb_fit +=1
    else:
        T0 = fixed_values[8]

    # A1
    if fit[9]!=0:
        OM = param[nb_fit]
        nb_fit +=1
    else:
        OM = fixed_values[9]

    # RA
    RA = fixed_values[0]
    DEC = fixed_values[1]

    return Pobs - calc_period(x, DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC)

class Application(Frame):

    def __init__(self, filenames, master=None):

        super().__init__(master)
        self.master = master

        self.data = Data()

        suffix = '.bestprof'
        nbestprof = 0
        for fn in filenames:
            if fn.endswith(suffix):
                prof = bestprof.bestprof(fn)
                for minute in np.arange(int(prof.T/60.0+0.5)):
                    t = minute * 60.
                    time = prof.epochi + prof.epochf + minute/1440.0
                    period = prof.p0 + t*(prof.p1 + 0.5*t*prof.p2)

                    self.data.add(time, period, ptype='s')
                nbestprof += 1
                
        print ("Loaded %d bestprof files", nbestprof)

        if (not nbestprof):
            try:
                mjds, periods, uncertainties = np.loadtxt(filenames[0], usecols=(0,1,2), unpack=True)
                self.data.set_mjd(mjds)
                self.data.set_period(periods)
                self.data.set_unc(uncertainties)
                print (self.data.get_mjd(), self.data.get_period(), self.data.get_unc())
            except:
                print ("Input format not recognized")
                raise

        # Variables Init
        self.ra_str = "00:00:00"
        self.dec_str = "+00:00:00"
        self.init_parameters()

        # Build Main window
        self.create_ui()

        self.draw_options()

        self.draw_param()

        self.set_entries()

        # Add graphic box and display Label
        self.xlabel="MJD"
        self.ylabel="Period (ms)"
        self.fig = Figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(1,1,1)
        
        ###########
        #left, width = 0.1, 0.8
        #rect1 = [left, 0.1, width, 0.7]
        #rect2 = [left, 0.8, width, 0.1]


        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.canvas = FigureCanvas(self.fig, master=master)
        #self.canvas.grid(row=8)
        #self.canvas.draw()
        #self.box_param.pack_start(self.canvas, True, True, 0)

        # Add Toolbar box
        #toolbar = NavigationToolbar(self.canvas, self)
        #self.box_param.pack_start(toolbar, False, False)

        # plot
        print ("Have Unc?", self.data.get_unc())
        if len(self.data.get_unc()):
            self.ax1.errorbar(self.data.get_mjd(), self.data.get_period(), yerr=self.data.get_unc(), color='r',fmt='o',zorder=10)
        else:
            self.ax1.scatter(self.data.get_mjd(), self.data.get_period(),color='r',s=20,edgecolor='r',marker='o',zorder=10)

        self.canvas.get_tk_widget().grid(row=6,columnspan=6, sticky=N+S+E+W)
        self.canvas.draw()

    # Quit Function
    #def quit(self):
        #gtk.main_quit();
        #return False

    def donothing(self):
        #print ('Action "%s" activated' % action.get_name())
        print('donothing')
        
    def create_ui(self):
        
        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_parfile)
        filemenu.add_command(label="Save", command=self.write_parfile)
        #filemenu.add_command(label="Save as...", command=self.donothing)

        filemenu.add_separator()

        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo", command=self.donothing)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", command=self.donothing)
        helpmenu.add_command(label="About...", command=self.donothing)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        self.master.config(menu=menubar)
        #root.mainloop()

    def init_parameters(self):
        """
        Init timing parameters
             fit_flag[] : which parameters to fit
             fit_values=[] : values of parameters
        """
        self.param = parfile.Parfile()

        # Array for LM fit
        self.fit_flag=[]
        self.fit_values=[]
        self.param2fit=[]
        self.mjds2=[]
        self.ps2=[]

        # Dict p2f for parameters to fit
        self.p2f={}
        self.label=[]
        for PARAM in PARAMS:
            if PARAM=="RA" or PARAM=="DEC":
                self.p2f[PARAM] = Param(is_string=True)
            else:
                self.p2f[PARAM] = Param()
            self.label.append(PARAM)
            self.p2f[PARAM].val = 0.0


        # Init self.fit to 0
        for i in range(len(self.p2f)):
            self.fit_flag.append(0)


    def read_parfile(self, filename):
        self.parfile = filename
        self.param.read(self.parfile)

        s = SkyCoord(self.param.RAJ, self.param.DECJ, unit=(u.hourangle, u.deg), frame='icrs')

        self.p2f['RA'].val = s.ra.radian
        self.p2f['DEC'].val = s.dec.radian
        self.p2f['P0'].val = self.param.P0
        self.p2f['P1'].val = self.param.P1/1e-15
        self.p2f['PEPOCH'].val = self.param.PEPOCH
        self.p2f['PB'].val = self.param.PB
        self.p2f['ECC'].val = self.param.ECC
        self.p2f['A1'].val = self.param.A1
        self.p2f['T0'].val = self.param.T0
        self.p2f['OM'].val = self.param.OM

    def write_parfile(self):
        for PARAM in PARAMS:
            self.param.set_param(PARAM, self.p2f[PARAM].val)
        filename =  filedialog.asksaveasfilename(title = "Save file")
        self.param.write(filename)

    def plot_model(self, widget=None):

        self.get_entries()

        # Init arrays
        xs=np.linspace(min(self.data.get_mjd()), max(self.data.get_mjd()), 20000)


        ys=calc_period(xs, 0.0, 0.0, self.p2f['P0'].val, self.p2f['P1'].val, self.p2f['PEPOCH'].val, self.p2f['PB'].val, self.p2f['ECC'].val, self.p2f['A1'].val, self.p2f['T0'].val, self.p2f['OM'].val, self.p2f['RA'].val, self.p2f['DEC'].val)


        # Convert into a Numpy array
        ys=np.asarray(ys)

        # Redraw plot
        self.ax1.cla()
        #print "Have Unc?", self.data.get_unc()
        if len(self.data.get_unc()):
            self.ax1.errorbar(self.data.get_mjd(), self.data.get_period(), yerr=self.data.get_unc(), color='r',fmt='o',zorder=10)
        else:
            self.ax1.scatter(self.data.get_mjd(), self.data.get_period(),color='r',s=20,edgecolor='r',marker='o',zorder=10)
        line, = self.ax1.plot(xs, ys)

        # Label and axis
        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        self.canvas.draw()


    def fit_model(self, widget=None):
        """
        Function to perform the fit of selected parameters to the values
        """

        # Retrieve values of parameters
        self.fit_values = []
        self.get_entries()
        for ii in range(len(self.p2f)):
            self.fit_values.append(self.p2f[self.label[ii]].val)

        # Set input parameters
        self.intput_param = []
        for ii,param in enumerate(self.v):
            if param.get():
                self.intput_param.append( self.fit_values[ii] )
                self.fit_flag[ii] = 1
            else:
                self.fit_flag[ii] = 0

        # If no parameters will be fitted, return now !
        if not self.intput_param:
            return

        # Retrieve which points to include (points in the window)
        self.ps2=[]
        self.mjds2=[]
        xmin,xmax=self.ax1.get_xlim()
        for mjd,period in zip(self.data.get_mjd(), self.data.get_period()):
            if(xmin<mjd and mjd<xmax):
                self.mjds2.append(mjd)
                self.ps2.append(period)

        self.mjds2 = np.asarray(self.mjds2)
        self.ps2 = np.asarray(self.ps2)
        #print self.mjds2,self.ps2

        # Do least square fit
        print ('Input Parameters :\n',self.intput_param)
        plsq = leastsq(timing_fcn, self.intput_param, args=(self.ps2, self.mjds2, self.fit_flag, self.fit_values))
        print ('Parameters fitted :\n', plsq[0])

        print ('chi**2 = ',np.sum(np.power(timing_fcn(self.intput_param,self.ps2, self.mjds2, self.fit_flag, self.fit_values),2)))
        # Return new parameters values in boxes
        j=0
        for i,dofit in enumerate(self.fit_flag):
            #print i,dofit, plsq
            if dofit:
                if sum(self.fit_flag)>=1:
                    val = plsq[0][j]
                else:
                    val = plsq[0]

                self.p2f[self.label[i]].val = val
                j+=1

        # Update the parameters entries
        self.set_entries()

        # Update the plot
        self.plot_model()

    def open_parfile(self):
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file")
        print (root.filename)
        if root.filename:
            self.read_parfile(root.filename)
            self.set_entries()
            if len(self.mjds):
                self.plot_model()

    def set_entries(self):
        for ii in range(len(self.p2f)):
            self.entry[ii].delete(0, END)
            if self.label[ii]=='RA':
                s = SkyCoord(self.p2f['RA'].val, self.p2f['DEC'].val , unit='radian', frame='icrs')
                print( s.ra.to_string(sep=':', unit='hourangle'))
                self.entry[ii].insert(0, s.ra.to_string(sep=':', unit='hourangle'))
            elif self.label[ii]=='DEC':
                s = SkyCoord(self.p2f['RA'].val, self.p2f['DEC'].val, unit='radian', frame='icrs')
                print(  s.dec.to_string(sep=':', unit=u.deg))
                self.entry[ii].insert(0, s.dec.to_string(sep=':', unit=u.deg))
            else:
                self.entry[ii].insert(0, self.p2f[self.label[ii]].val)

    def get_entries(self):
        for ii in range(len(self.p2f)):
            if self.label[ii]=='RA':
                self.ra_str = self.entry[ii].get()
                s = SkyCoord(self.ra_str, self.dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                self.p2f[self.label[ii]].val = s.ra.radian
            elif self.label[ii]=='DEC':
                self.dec_str = self.entry[ii].get()
                s = SkyCoord(self.ra_str, self.dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                self.p2f[self.label[ii]].val = s.dec.radian
            else:
                self.p2f[self.label[ii]].val = float(self.entry[ii].get())

    def key_press_menu(self, event):
        """
        """

        if event.key=='x':
            self.fit_model()
        if event.key=='p':
            self.plot_model()


    def draw_options(self):

        # Button "Load Parfile"
        bpar = Button(self.master, text="Load ParFile", command = self.open_parfile)
        bpar.grid(row=0, column=0) 

        # Button "Save"
        bsave = Button(self.master, text="Save ParFile", command = self.write_parfile)
        bsave.grid(row=1, column=0)

        # Button "Plot Model"
        bplot = Button(self.master, text="Plot Model", command = self.plot_model)
        bplot.grid(row=2, column=0)

        # Button "Fit"
        bfit = Button(self.master, text="Fit Model", command = self.fit_model)
        bfit.grid(row=3, column=0)

    # Which param to held fixed
    def set_fit(self, ii):
        #print ("Parameter %s was toggled %s" % (data, ("OFF", "ON")[widget.get_active()]))
        print (self.v[ii].get())
        if self.v[ii].get():
            self.v[ii].set(0)
        else:
            self.v[ii].set(1)

    def draw_param(self):

        #self.ivar = IntVar()
        self.v = [0] * len(PARAMS)
        self.checkbut = [None] * len(PARAMS)
        self.entry = [None] * len(PARAMS)
        ii = 0
        for i in range(0,5):
            for j in range(0,2):
                self.v[ii] = IntVar()
                self.v[ii].set(0)
                self.checkbut[ii] = Checkbutton(self.master, text = self.label[ii], variable = self.v, command= lambda ii=ii: self.set_fit(ii))
                self.checkbut[ii].grid(row=i, column=1+j*2)
                self.entry[ii] = Entry(self.master)
                self.entry[ii].grid(row=i, column=2+j*2)
                ii += 1


if __name__ == '__main__':

    usage = "usage: %prog [options] -f period.dat"

    parser = OptionParser(usage)
    parser.add_option("-c", "--convert_f", action="store_true", dest="freq", default=False, help="Use frequency")
    parser.add_option("-m", "--ms", action="store_true", dest="ms", default=False, help="Use period in ms")
    parser.add_option("-f", "--file", type="string", dest="filename", help="Input file")

    (opts, args) = parser.parse_args()

    if len(args)==0:
        print (full_usage)


    root = Tk()
    root.title('fitorbit.py')
    root.geometry("950x700+10+10")
    app = Application(args, master=root)
    app.mainloop()
