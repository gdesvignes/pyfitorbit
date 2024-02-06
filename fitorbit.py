#!/usr/bin/env python


import os
import sys
import time
import math
from tkinter import filedialog
from tkinter import *

from optparse import OptionParser

import numpy as np
from lmfit import Model

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
import textwrap

DEG2RAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RAD2DEG    = float('57.295779513082320876798154814105170332405472466564')
C          = float('2.99792458e8')

full_usage = """
usage : fitorbit.py [options] [*.bestprof or period file]

  [-h, --help]        : Display this help

  Bestprof files as produced by Presto

  File with MJD, periods, period uncertainties

  

"""

usage = "usage: %prog [options]"


PARAMS = ['RA', 'DEC', 'P0', 'P1', 'PEPOCH', 'PB', 'ECC', 'A1', 'T0', 'OM']

class Data:
    def __init__(self):
        self.mjds = np.array([])
        self.period = np.array([])
        self.unc = np.array([])

    def add(self, mjd1, period1, unc1, ptype='ms'):
        self.mjds = np.append(self.mjds, mjd1)
        self.unc = np.append(self.unc, unc1)
        #print (mjd1, period1, unc1)
        if  ptype=='ms':
            self.period = np.append(self.period, period1)
        elif  ptype=='s':
            self.period = np.append(self.period, period1*1000.)

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

    #print(DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC)
    #p0 = par0
    #pb = par1
    #a1 = par2
    #om = par5
    #e = par4
    #t0 = par3

    
    if PB:

        k1 = 2*np.pi*A1/(PB*86400.0*np.sqrt(1-ECC*ECC))

        #easc = 2*np.arctan(np.sqrt((1-ECC)/(1+ECC)) * np.tan(-np.radians(OM)/2)) # (rad)
        #epperias = T0 - PB/360.0*(np.degrees(easc) - 180./np.pi * ECC * np.sin(easc))

        dt = (x-T0)*86400.
        speri = np.fmod(dt, PB*86400.)
        speri[speri<0] += PB*86400.
        mean_anom = 360*speri/(PB*86400)
        mean_anom = np.fmod(mean_anom,360.0)
        mean_anom = np.where(np.greater(mean_anom, 360.0), mean_anom-360.0, mean_anom)

        # Return ecc_anom (in rad) by iteration
        ecc_anom = utils.eccentric_anomaly2(ECC, mean_anom*DEG2RAD)
        #print (ecc_anom)
        
        # Return true anomaly in deg
        true_anom = 2*RAD2DEG*np.arctan(np.sqrt((1+ECC)/(1-ECC))*np.tan(ecc_anom/2))

        #print(DRA, DDEC, P0, P1, PEPOCH, PB, ECC, A1, T0, OM, RA, DEC)
        #print(k1, mean_anom, true_anom)
        
        #print RA, DEC
        #dv = utils.deltav(x, RA, DEC, RA-DRA, DEC-DDEC, 2000.0)
        #print dv
        return 1000*(P0+P1*(x-PEPOCH)*86400) * (1 + k1*np.cos(DEG2RAD*(true_anom+OM)) + k1*ECC*np.cos(OM*DEG2RAD))
        #return 1000*(P0+P1*(x-PEPOCH)*86400) * (1 + k1*np.cos(DEG2RAD*(true_anom+OM)) + k1*ECC*np.cos(OM*DEG2RAD)) * (1-dv/3e8)
    else:
        return 1000*(P0+P1*(x-PEPOCH)*86400)


class Application(Frame):

    def __init__(self, filenames, master=None, input_parfile=None):

        super().__init__(master)
        self.master = master
        have_bestprof = False

        self.data = Data()

        if filenames:
            suffix = '.bestprof'
            for fn in filenames:
                if fn.endswith(suffix):
                    have_bestprof = True
                    prof = bestprof.bestprof(fn)
                    #print (fn, prof.epochi_bary, prof.epochf_bary, prof.p0_bary)
                    for minute in np.arange(int(prof.T/60.0+0.5)):
                        t = minute * 60.
                        time = prof.epochi_bary + prof.epochf_bary + minute/1440.0
                        period = prof.p0_bary + t*(prof.p1_bary + 0.5*t*prof.p2_bary)
                        unc = (prof.p0err_bary**2 + prof.p1err_bary**2 + prof.p2err_bary**2)**.5
                        self.data.add(time, period, unc, ptype='s')

            if not have_bestprof:
                try:
                    mjds, periods, uncertainties = np.loadtxt(filenames[0], usecols=(0,1,2), unpack=True)
                    self.data.set_mjd(mjds)
                    self.data.set_period(periods)
                    self.data.set_unc(uncertainties)
                    #print (self.data.get_mjd(), self.data.get_period(), self.data.get_unc())
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

        if input_parfile:
            self.read_parfile(input_parfile)
            self.set_entries()
            #if len(self.data.mjds):
            #    self.plot_model()
        
        # Add graphic box and display Label
        self.plot_orbital=False
        self.xlabel="MJD"
        self.ylabel="Period (ms)"
        self.fig = Figure(facecolor='white')
        #self.fig.canvas.mpl_connect('key_press_event', self.key_press_menu)
        self.master.bind('<KeyPress>', self.key_press_menu)
        self.master.bind('<Return>', self.onReturn)
        self.ax1 = self.fig.add_subplot(3, 1, (1,2))
        self.ax2 = self.fig.add_subplot(3, 1, 3)
        self.ax1.format_coord = lambda x, y: ""
        self.ax2.format_coord = lambda x, y: ""
        
        ###########
        #left, width = 0.1, 0.8
        #rect1 = [left, 0.1, width, 0.7]
        #rect2 = [left, 0.8, width, 0.1]


        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        self.ax2.set_xlabel(self.xlabel)
        self.ax2.set_ylabel("Residuals (mP0)")
        self.ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        
        self.canvas = FigureCanvas(self.fig, master=master)
        #self.canvas.grid(row=8)
        #self.canvas.draw()
        #self.box_param.pack_start(self.canvas, True, True, 0)

        # Add Toolbar box
        #toolbar = NavigationToolbar(self.canvas, self)
        #self.box_param.pack_start(toolbar, False, False)

        # plot
        #print ("Have Unc?", self.data.get_unc())
        if len(self.data.get_unc()):
            self.ax1.errorbar(self.data.get_mjd(), self.data.get_period(), yerr=self.data.get_unc(), color='r',fmt='o',zorder=10)
        else:
            self.ax1.scatter(self.data.get_mjd(), self.data.get_period(),color='r',s=20,edgecolor='r',marker='o',zorder=10)

        self.canvas.get_tk_widget().grid(row=6,columnspan=6, sticky=N+S+E+W)

        toolbarFrame = Frame(master=root)
        toolbarFrame.grid(row=7,columnspan=6)
        toolbar = NavigationToolbar(self.canvas, toolbarFrame)

        self.connect()
        self.cidAx1 = self.ax1.callbacks.connect('xlim_changed', self.on_xlims_change)
        #self.ax1.callbacks.connect('ylim_changed', self.on_ylims_change)
        self.cidAx2 = self.ax2.callbacks.connect('xlim_changed', self.on_xlims_change)
        #self.ax2.callbacks.connect('ylim_changed', self.on_ylims_change)
        
        xmin, xmax = self.ax1.get_xlim()
        self.oxmin, self.oxmax =  self.ax1.get_xlim()
        self.ax2.set_xlim(xmin, xmax)
                          
        self.canvas.draw()

    def onReturn(self, *event):
        self.master.focus_set()
        
    def connect(self):
        #print("Connect to callbacks")
        self.cidAx1 = self.ax1.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.cidAx2 = self.ax2.callbacks.connect('xlim_changed', self.on_xlims_change)

    def disconnect(self):
        self.ax1.callbacks.disconnect(self.cidAx1)
        self.ax2.callbacks.disconnect(self.cidAx2)
        
    def on_xlims_change(self, event_ax):
        
        self.xmin, self.xmax =  event_ax.get_xlim()
        #self.ax1.set_xlim(xmin, xmax)
        #print(xmin, xmax)
        self.disconnect()
        self.ax2.set_xlim(self.xmin, self.xmax)
        self.connect()
            
    def donothing(self):
        #print ('Action "%s" activated' % action.get_name())
        print('donothing')

    def print_help(self):
        print (textwrap.dedent("""

        HELP: Press the following key to:
        ---------------------------------
        p: Plot periods vs MJD
        o: Plot periods vs orbital phase
        x: Fit the parameters
        """))
        
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
        helpmenu.add_command(label="Help Index", command=self.print_help)
        helpmenu.add_command(label="About...", command=self.donothing)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        self.master.config(menu=menubar)
        #root.mainloop()

    def init_parameters(self):
        """
        """
        self.param = parfile.Parfile()

        # Dict p2f for parameters to fit
        self.p2f={}
        self.label=[]
        for PARAM in PARAMS:
            self.label.append(PARAM)
            self.p2f[PARAM] = 0.0

    def read_parfile(self, filename):
        self.parfile = filename
        self.param.read(self.parfile)

        s = SkyCoord(self.param.RAJ, self.param.DECJ, unit=(u.hourangle, u.deg), frame='icrs')

        self.p2f['RA'] = s.ra.radian
        self.p2f['DEC'] = s.dec.radian
        self.p2f['P0'] = self.param.P0
        self.p2f['P1'] = self.param.P1
        self.p2f['PEPOCH'] = self.param.PEPOCH
        self.p2f['PB'] = self.param.PB
        self.p2f['ECC'] = self.param.ECC
        self.p2f['A1'] = self.param.A1
        self.p2f['T0'] = self.param.T0
        self.p2f['OM'] = self.param.OM

    def has_model(self):
        #if self.p2f['P0'] and self.p2f['PB'] and self.p2f['A1']:
        if self.p2f['P0']:
            return True
        else:
            return False
        
    def write_parfile(self):
        for PARAM in PARAMS:
            self.param.set_param(PARAM, self.p2f[PARAM])
        filename =  filedialog.asksaveasfilename(title = "Save file")
        self.param.write(filename)

    def plot_model(self, widget=None):

        self.get_entries()

        # Init arrays
        xs=np.linspace(min(self.data.get_mjd()), max(self.data.get_mjd()), 20000)

        if self.has_model():
            ys=calc_period(xs, 0.0, 0.0, self.p2f['P0'], self.p2f['P1'], self.p2f['PEPOCH'], self.p2f['PB'], self.p2f['ECC'], self.p2f['A1'], self.p2f['T0'], self.p2f['OM'], self.p2f['RA'], self.p2f['DEC'])

            # Convert into a Numpy array
            ys=np.asarray(ys)

        # Redraw plot
        self.ax1.cla()
        self.ax2.cla()
        self.connect()
        #print "Have Unc?", self.data.get_unc()

        if self.plot_orbital and self.has_model():
            Xval = np.modf((self.data.get_mjd()-self.p2f['T0'])/self.p2f['PB'])[0]
            Xval = np.where(Xval<0, Xval+1, Xval)
            #print (Xval)
            XMval = np.modf((xs-self.p2f['T0'])/self.p2f['PB'])[0]
            XMval = np.where(XMval<0, XMval+1, XMval)

            ids = np.argsort(XMval)
            YMval = ys[ids]
            XMval = XMval[ids]
            self.xlabel = "Orbital phase"
        elif self.plot_orbital:
            print("No orbital model")
            Xval = self.data.get_mjd()
            XMval = xs
            if self.has_model():
                YMval = ys
            self.xlabel = "MJD"
        else:
            Xval = self.data.get_mjd()
            XMval = xs
            if self.has_model():
                YMval = ys
            self.xlabel = "MJD"
        
        if len(self.data.get_unc()):
            self.ax1.errorbar(Xval, self.data.get_period(), yerr=self.data.get_unc(), color='r',fmt='o',zorder=10)
        else:
            self.ax1.scatter(Xval, self.data.get_period(),color='r',s=20,edgecolor='r',marker='o',zorder=10)

        if self.has_model():
            line, = self.ax1.plot(XMval, YMval, zorder=20)
        
        # Label and axis
        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        if self.has_model():
            ym = calc_period(self.data.get_mjd(), 0.0, 0.0,
                         self.p2f['P0'], self.p2f['P1'],
                         self.p2f['PEPOCH'], self.p2f['PB'],
                         self.p2f['ECC'], self.p2f['A1'],
                         self.p2f['T0'], self.p2f['OM'],
                         self.p2f['RA'], self.p2f['DEC'])
        
        ### Plot residuals ###
        self.ax1.set_xlim(self.xmin, self.xmax)
        #xmin, xmax = self.ax1.get_xlim()
        if self.has_model():
            if len(self.data.get_unc()):
                self.ax2.errorbar(Xval, (self.data.get_period() - ym) / self.p2f['P0'], yerr=self.data.get_unc(), color='r',fmt='o',zorder=10)
            else:
                self.ax2.scatter(Xval, (self.data.get_period() - ym) / self.p2f['P0'], color='r',s=20,edgecolor='r',marker='o',zorder=10)
        self.ax2.set_xlim(self.xmin, self.xmax)
        self.ax2.set_xlabel(self.xlabel)
        self.ax2.set_ylabel("Residuals (mP0)")
        self.ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        
        self.canvas.draw()


    def fit_model(self, widget=None):
        """
        Function to perform the fit of selected parameters to the values
        """
        # create lmfit model
        fmodel = Model(calc_period)
        
        # Retrieve values of parameters
        self.get_entries()
        params = fmodel.make_params(DRA=0, DDEC=0, P0=self.p2f['P0'], \
                                    P1=self.p2f['P1'], \
                                    PEPOCH=self.p2f['PEPOCH'], \
                                    PB=self.p2f['PB'], \
                                    ECC=self.p2f['ECC'], \
                                    A1=self.p2f['A1'], \
                                    T0=self.p2f['T0'], \
                                    OM=self.p2f['OM'], \
                                    RA=self.p2f['RA'], \
                                    DEC=self.p2f['DEC'])
        
        # Set boundaries for eccentricity
        params['ECC'].min = 0.0
        params['ECC'].max = 1.0
        
        # Set which parameters to fit
        do_fit = False
        params['DRA'].vary = False
        params['DDEC'].vary = False
        for ii,param in enumerate(self.v):
            if param.get():
                params[self.label[ii]].vary = True
                do_fit = True
            else:
                params[self.label[ii]].vary = False
                
        # If no parameters to fit, return now !
        if not do_fit:
            return

        # Retrieve which points to include (points in the window)
        if self.plot_orbital:
            xmin = np.min(self.data.get_mjd())
            xmax = np.max(self.data.get_mjd())
        else:
            xmin,xmax=self.ax1.get_xlim()
        mjds = self.data.get_mjd()
        periods = self.data.get_period()
        uncs = self.data.get_unc()
        self.mjds2 = mjds[np.where((xmin<mjds) & (mjds<xmax))]
        self.periods2 = periods[np.where((xmin<mjds) & (mjds<xmax))]
        if uncs.size==0:
            # Do the actual fit
            result = fmodel.fit(self.periods2, params, x=self.mjds2, max_nfev=50)
        else:
            print("Do weighted fit")
            self.unc2 = uncs[np.where((xmin<mjds) & (mjds<xmax))]
            # Do the actual fit
            result = fmodel.fit(self.periods2, params, x=self.mjds2, weights=1/self.unc2, max_nfev=50)
        print(result.fit_report())
        
        for par in PARAMS:
            self.p2f[par] = result.params[par].value
                
        # Wrap parameters if needed  
        while self.p2f['A1'] < 0:
            self.p2f['A1'] *= - 1
            self.p2f['T0'] += self.p2f['PB']/2.
        while self.p2f['ECC'] < 0:
            self.p2f['ECC'] *= -1
            self.p2f['OM'] += 180.
            self.p2f['T0'] += self.p2f['PB']/2.
        while self.p2f['OM'] < 0:
            self.p2f['OM'] += 360.
        while self.p2f['OM'] > 360:
            self.p2f['OM'] -= 360.
            
            
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
            if len(self.data.mjds):
                self.plot_model()

    def set_entries(self):
        for ii in range(len(self.p2f)):
            self.entry[ii].delete(0, END)
            if self.label[ii]=='RA':
                s = SkyCoord(self.p2f['RA'], self.p2f['DEC'] , unit='radian', frame='icrs')
                self.entry[ii].insert(0, s.ra.to_string(sep=':', unit='hourangle'))
            elif self.label[ii]=='DEC':
                s = SkyCoord(self.p2f['RA'], self.p2f['DEC'], unit='radian', frame='icrs')
                self.entry[ii].insert(0, s.dec.to_string(sep=':', unit=u.deg))
            else:
                self.entry[ii].insert(0, self.p2f[self.label[ii]])

    def get_entries(self):
        for ii in range(len(self.p2f)):
            if self.label[ii]=='RA':
                self.ra_str = self.entry[ii].get()
                s = SkyCoord(self.ra_str, self.dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                self.p2f[self.label[ii]] = s.ra.radian
            elif self.label[ii]=='DEC':
                self.dec_str = self.entry[ii].get()
                s = SkyCoord(self.ra_str, self.dec_str, unit=(u.hourangle, u.deg), frame='icrs')
                self.p2f[self.label[ii]] = s.dec.radian
            else:
                try:
                    self.p2f[self.label[ii]] = float(self.entry[ii].get())
                except ValueError:
                    pass
                    
    def key_press_menu(self, event):
        """
        """
        #print('you pressed', event.keycode, event.char)
        if event.char=='h':
            self.print_help()
        if event.char=='x':
            self.fit_model()
        if event.char=='p':
            self.plot_orbital=False
            self.xmin = self.oxmin
            self.xmax = self.oxmax
            self.plot_model()
        if event.char=='o' and self.has_model():
            self.plot_orbital=True
            self.xmin =	0
            self.xmax =	1
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
        if self.v[ii].get():
            self.v[ii].set(0)
        else:
            self.v[ii].set(1)

    def draw_param(self):

        self.v = [0] * len(PARAMS)
        self.checkbut = [None] * len(PARAMS)
        self.entry = [None] * len(PARAMS)
        ii = 0
        for i in range(0,5):
            for j in range(0,2):
                self.v[ii] = IntVar()
                self.v[ii].set(0)
                if self.label[ii]=='PEPOCH' or self.label[ii]=='RA' or self.label[ii]=='DEC': state = 'disabled'
                else: state = 'normal'
                self.checkbut[ii] = Checkbutton(self.master, text = self.label[ii], variable = self.v, command= lambda ii=ii: self.set_fit(ii), state=state)
                self.checkbut[ii].grid(row=i, column=1+j*2)
                self.entry[ii] = Entry(self.master)
                self.entry[ii].grid(row=i, column=2+j*2)
                ii += 1


if __name__ == '__main__':

    usage = """usage: %prog [options] <bestprof files or period files>
    Press the following Keys:
        p: Plot periods vs MJD
        o: Plot periods vs orbital phase
        f: Fit the parameters
"""

    parser = OptionParser(usage)
    #parser.add_option("-c", "--convert_f", action="store_true", dest="freq", default=False, help="Use frequency")
    #parser.add_option("-m", "--ms", action="store_true", dest="ms", default=False, help="Use period in ms")
    parser.add_option("-f", "--parfile", type="string", dest="parfile", help="Input parfile")

    (opts, args) = parser.parse_args()

    if len(args)==0:
        print (full_usage)

    if opts.parfile:
        parfn = opts.parfile
    else: parfn = None

    root = Tk()
    root.title('fitorbit.py')
    root.geometry("950x700+10+10")
    app = Application(args, master=root, input_parfile=parfn)
    app.mainloop()
