# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for some photocurrent measurements with DF spectra overlay for different TPDT MLAggs (varying AuNP size)

- CA Photocurrent
- OCP Photocurrent
- LSV Photocurrent
- Dark LSV
- Dark CV
- DF Map
- SERS Map

Copied from 2024-08-28_BPDT_MLAgg_Photocurrent_Plotter.py

Data:

(samples: 2024-09-04_TPDT_MLAggs_FTO)

"""

from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import statistics
import scipy
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm
from pylab import *
from pybaselines.polynomial import modpoly
import nplab
import h5py
import natsort
import os

from nplab import datafile
from nplab.analysis.general_spec_tools import spectrum_tools as spt
from nplab.analysis.general_spec_tools import npom_sers_tools as nst
from nplab.analysis.general_spec_tools import agg_sers_tools as ast
from nplab.analysis.SERS_Fitting import Auto_Fit_Raman as afr
from nplab.analysis.il322_analysis import il322_calibrate_spectrum as cal
from nplab.analysis.il322_analysis import il322_SERS_tools as SERS
from nplab.analysis.il322_analysis import il322_DF_tools as df

from lmfit.models import GaussianModel

save = False
save_to_h5 = False


#%% Data storage classes


class Particle(): 
    def __init__(self):
        self.name = None

class Peak():
    def __init__(self, height, mu, width, baseline, **kwargs):
        self.height = height
        self.mu = mu
        self.width = width
        self.baseline = baseline


#%% h5 files

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
if save_to_h5 == True:
    save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-09-24_Processed_Au_Mirror_FTO_PEC_Data.h5")

## PEC h5 Files

my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-24_Bare_FTO_Au_Mirror_Photocurrent.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-09-24_Au_Mirror_FTO Photocurrent Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def exp_decay(x, m, t, b):
    return m * np.exp(-t * x)

def cottrell(x, a, b, c):
    
    return (a/(sqrt(pi*x + c))) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c

def reciprocal(x, a, b, c):
    
    return (a/(x+b)) + c

def linear(x, a, b):
    
    return a*x + b
    
        
#%% Functions for saving data to h5

def save_ca(particle, overwrite = False):
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    
    ca_spectra = particle.ca_spectra
    
    for ca in ca_spectra:
    
        ## Get CA attributes, remove troublesome attributes
        try:
            ca.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(ca.__dict__)
        attrs.pop('rc_params')
    
        ## Save 
        group.create_dataset(name = ca.name + '_%d', data = ca.y, attrs = attrs)
        

def save_ocp(particle, overwrite = False):
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    
    # OCP Currents
    
    ocp_spectra = particle.ocp_spectra
    
    for ocp in ocp_spectra:
    
        ## Get ocp attributes, remove troublesome attributes
        try:
            ocp.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(ocp.__dict__)
        attrs.pop('rc_params')
    
        ## Save 
        group.create_dataset(name = ocp.name + '_%d', data = ocp.y, attrs = attrs)
        
    
    # OCP Voltages
    
    ocp_spectra = particle.ocp_spectra_voltage
    
    for ocp in ocp_spectra:
    
        ## Get ocp attributes, remove troublesome attributes
        try:
            ocp.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(ocp.__dict__)
        attrs.pop('rc_params')
    
        ## Save 
        group.create_dataset(name = ocp.name + '_%d', data = ocp.y, attrs = attrs)
    

def save_cv(particle, overwrite = False):
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    
    cv_spectra = particle.cv_spectra
    
    for cv in cv_spectra:
    
        ## Get CA attributes, remove troublesome attributes
        try:
            cv.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(cv.__dict__)
        attrs.pop('rc_params')
    
        ## Save 
        group.create_dataset(name = cv.name + '_%d', data = cv.y, attrs = attrs)
        

def save_lsv(particle, overwrite = False):
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    
    lsv_spectra = particle.lsv_spectra
    
    for lsv in lsv_spectra:
    
        ## Get CA attributes, remove troublesome attributes
        try:
            lsv.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(lsv.__dict__)
        attrs.pop('rc_params')
    
        ## Save 
        group.create_dataset(name = lsv.name + '_%d', data = lsv.y, attrs = attrs)        
        
        
#%% EChem plotting functions

def plot_cv(particle, save = False, save_to_h5 = False):
    
    spectra = particle.cv_spectra
    power_dict = particle.power_dict
    size = particle.size
    # df_avg = particle.df_avg
    
    
    fig, ax = plt.subplots(1, 1, figsize=[14,8])
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot CV spectra
    
    for spectrum in spectra:
        
        ax.plot(spectrum.x, spectrum.y * 10**6)
        
        
    ax.set_xlabel('Potential v. Ag/AgCl (V)', fontsize = 'large')
    ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
    # fig.subplots_adjust(right=0.9)
    fig.suptitle(particle.name + ' CV', fontsize = 'x-large', horizontalalignment='center')   
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
      
    
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + particle.name + ' CV' + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
        
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str(particle.name)]
        except:
            group = save_h5.create_group(str(particle.name))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = 'CV_jpg_%d', data = fig_array)
        
        
def plot_lsv(particle, save = False, save_to_h5 = False):
    
    spectra = particle.lsv_spectra
    power_dict = particle.power_dict
    size = particle.size
    # df_avg = particle.df_avg
    
    
    fig, ax = plt.subplots(1, 1, figsize=[16,8])
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
                
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, ((spectrum.y_smooth* 10**6) + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)

    
    ## Light on/off bars
    # ax.set_xlim(0.1, 0.4)
    # ax.set_ylim(-0.01, 0.2)
    ylim = ax.get_ylim()  
    xlim = ax.get_xlim()    
    ax.bar(np.arange((spectrum.x_raw.min() + spectrum.toggle/2), spectrum.x_raw.max() + (spectrum.toggle/2), spectrum.toggle*2) - 0.0007, height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
    # ax.text(s = 'On', x = (spectrum.x.min() + spectrum.toggle), y = ylim[0], fontsize = 'small')
    # ax.text(s = 'Off', x = (spectrum.x.min()), y = ylim[0], fontsize = 'small')
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])    
    
    ax.set_xlabel('Voltage (V)', fontsize = 'large')
    ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 50 nm FWHM', rotation=270, fontsize = 'large', labelpad = 40)
    cbar_ax.yaxis.set_label_position("right")
    fig.suptitle(particle.name + ' LSV', fontsize = 'large', horizontalalignment='center', x = 0.55, y = 0.95)
    

    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + particle.name + ' LSV' + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
        
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str(particle.name)]
        except:
            group = save_h5.create_group(str(particle.name))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = 'LSV_zoom_jpg_%d', data = fig_array)
            
       
def plot_ca_pec(particle, save = False, save_to_h5 = False):
    
    spectra = particle.ca_spectra
    power_dict = particle.power_dict
    size = particle.size
    # df_avg = particle.df_avg
    
    voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    fig, axes = plt.subplots(len(voltages), 2, figsize=[17,16], width_ratios = (2,1))
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot Light Toggle CA on left
    
    for j, voltage in enumerate(voltages):
    
        ax = axes[j][0]
    
        ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
        offset = 0.0
        
        if j < len(axes)-1:
            ax.xaxis.set_ticklabels([])
        
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            if spectrum.voltage != voltage:
                continue
            
            color = cmap(norm(spectrum.wavelength))  
            ax.plot(spectrum.x, ((spectrum.y_smooth * 10**6) + i * offset), label = spectrum.voltage, color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
        
        ## Light on/off bars
        ax.set_xlim(20, 500)
        ylim = ax.get_ylim()    
        ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
        ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
        ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
        ax.set_ylim(ylim)
        
        
    # Plot PEC on right
    
    for j, voltage in enumerate(voltages):
    
        ax = axes[j][1]
    
        ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
        offset = 0.0
        
        if j < len(axes)-1:
            ax.xaxis.set_ticklabels([])
        
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            if spectrum.voltage != voltage:
                continue
            
            ## Plot
            color = cmap(norm(spectrum.wavelength))  
            if spectrum.wavelength > 440:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                 
        ax2 = ax.twinx()
        # ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        if j == int(len(voltages)/2):
            ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
            ax2.yaxis.set_label_position("left")
        ax.yaxis.tick_right()
        ax2.yaxis.tick_left()
        ax.set_xlim(400, 900)
        
    axes[-1][0].set_xlabel('Time (s)', fontsize = 'x-large')
    axes[int((len(axes) - 1)/2)][0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0, 0.15, 0.05, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 50 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
    cbar_ax.yaxis.tick_left()
    cbar_ax.yaxis.set_label_position("left")
    fig.suptitle(particle.name + ' CA Photocurrent', fontsize = 'xx-large', horizontalalignment='center', x = 0.45, y = 0.94)
    
    
    axes[-1][1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    axes[int((len(axes) - 1)/2)][1].set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[int((len(axes) - 1)/2)][1].yaxis.set_label_position("right")
    for ax in axes:
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
           
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + particle.name + ' CA Photocurrent' + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
        
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str(particle.name)]
        except:
            group = save_h5.create_group(str(particle.name))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = 'CA Photocurrent_jpg_%d', data = fig_array)
        

def plot_subtracted_ipce(particle, save = False, save_to_h5 = False):

    
    # Calculate Au Absorption

    au = np.genfromtxt(r"C:\Users\il322\Desktop\Offline Data\Yakubovsky-117nm.txt", skip_header = 1)
    wln = au[:,0] * 1E-6 ## wln in m
    n = au[:,1]
    k = au[:,2]
    nk = n+(1j*k)

    ## Abs from alpha
    alpha = (4 * np.pi * k)/(wln) # alpha in 1/m
    A = (1 - np.exp(-1 * alpha * 60E-9)) * 100 # Abs = 100% * [1 - e^(-alpha * d)]; d = 60 nm

    ## Reflectance
    R = np.abs((1 - nk)/(1 + nk))**2 # R at top interface
    R2 = np.abs((1 - nk)/(1 + nk))**2 # R at bottom interface
    thick = particle.size * 1e-9
    T_1 = (1 - R) # I after 1st air-Au interface
    T_2 = (1 - R) * e**(-alpha * thick) # I after Au thin film of thickness = thick 
    T_3 = (1 - R2) * T_2 # I after 2nd Au-air interface
    A = T_1 - T_3 # Absorption in thin film approximation
    wln = wln *1E9
    Au = SERS.SERS_Spectrum(wln, A)
    ''' Note that A doesn't take into account absorption from cascading back reflections at Au interfaces'''
    
    
    spectra = particle.ca_spectra
    power_dict = particle.power_dict
    size = particle.size
    # df_avg = particle.df_avg
    
    voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    fig, axes = plt.subplots(len(voltages), 2, figsize=[12,16], width_ratios = (1,1))
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    for j, voltage in enumerate(voltages):
    
        
        # Plot regular IPCE on left
        
        ax = axes[j][0]
        
        ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
        offset = 0.0
        
        if j < len(axes)-1:
            ax.xaxis.set_ticklabels([])
        
        x = []
        y = []
        
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            if spectrum.voltage != voltage:
                continue
            
            ## Plot
            color = cmap(norm(spectrum.wavelength))  
            if spectrum.wavelength > 450: # Here
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                scatt = ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                x.append(spectrum.wavelength)
                y.append(np.mean(spectrum.ipce))
                
        ## Plot Au absorption
        ### Interpolate & Normalize Au abs to IPCE
        x = np.array(x)
        y = np.array(y)
        Au.truncate(400, 900)
        Au.normalise(norm_range = (0, y.max()))
        Au_interp = np.interp(x, Au.x, Au.y_norm)
        au_line = ax.plot(Au.x, Au.y_norm, color = 'darkgoldenrod', linestyle = 'dashed', zorder = 0, label = 'Au abs')

        ## Plot DF        
        ax2 = ax.twinx()
        # ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        if j == int(len(voltages)/2):
            # ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
            ax2.yaxis.set_label_position("left")
        ax.yaxis.tick_right()
        ax2.yaxis.tick_left()
        ax.set_xlim(400, 900)
        
        
        # Plot subtracted IPCE on right
        
        ax = axes[j][1]
        ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
        subtracted_line = ax.plot(x, y - Au_interp, color = 'darkgoldenrod', zorder = 0)
        
        ## Plot DF        
        ax2 = ax.twinx()
        # df_line = ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        # if j == int(len(voltages)/2):
            # ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
        ax2.yaxis.set_ticklabels([])
        ax.yaxis.tick_right()
        # ax2.yaxis.tick_left()
        ax.set_xlim(400, 900)
       

    # Labelling
    
    ## Label axes
    # fig.subplots_adjust(right=1.1)
    fig.suptitle(particle.name + ' CA Photocurrent', fontsize = 'xx-large', horizontalalignment='center', y = 1.02)
    axes[-1][0].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    axes[-1][1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    axes[int((len(axes) - 1)/2)][0].set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[int((len(axes) - 1)/2)][1].set_ylabel('Subtracted IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[int((len(axes) - 1)/2)][0].yaxis.set_label_position("right")
    axes[int((len(axes) - 1)/2)][1].yaxis.set_label_position("right")
    for ax in axes:
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)

    ## Legend
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    # fig.legend(handles = [scatt_plus, scatt_minus, au_line[0], subtracted_line[0], df_line[0]], labels = ['IPCE +', 'IPCE -', 'Au abs', 'IPCE - Au abs', 'DF'], ncols = 5, bbox_to_anchor=(0.93, 0.99))

    fig.tight_layout()    
    
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + particle.name + ' Subtracted CA Photocurrent ' + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
                
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str(particle.name)]
        except:
            group = save_h5.create_group(str(particle.name))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = 'Subtracted CA Photocurrent_jpg_%d', data = fig_array)


        
def plot_ocp_pec(particle, save = False, save_to_h5 = False):
    
    spectra = particle.ocp_spectra
    power_dict = particle.power_dict
    size = particle.size
    # df_avg = particle.df_avg    
    
    fig, axes = plt.subplots(2, 2, figsize=[18,13], width_ratios = (2,1))
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot OCP Currents on top
    
    ## Plot Light Toggle OCP on left
    
    ax = axes[0][0]
    offset = 0.0
    
    ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, ((spectrum.y_smooth * 10**6) + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ## Light on/off bars
    ax.set_xlim(20, 500)
    ylim = ax.get_ylim()    
    ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
    ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')    
    ax.set_ylim(ylim)
            
    ## Plot PEC on right
    
    ax = axes[0][1]
    offset = 0.0
    
    ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        if spectrum.wavelength > 450:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
            ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            # for ipce in spectrum.ipce:
                # ax.scatter(spectrum.wavelength, ipce, color = 'black', s = 150, marker = marker, linewidth = 4)
          
    ax2 = ax.twinx()
    # ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
    ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
    ax2.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax2.yaxis.tick_left()
    ax.set_xlim(400, 900)
    
    
    # Plot OCP Voltages on bottom
    
    spectra = particle.ocp_spectra_voltage
    
    ## Plot Light Toggle OCP on left
    
    ax = axes[1][0]
    offset = 0.0
    
    for i, spectrum in enumerate(spectra):
        if spectrum.wavelength < 7050:        
            linetype = 'solid'
            color = cmap(norm(spectrum.wavelength))  
            ax.plot(spectrum.x[150:], ((spectrum.y_smooth[150:]) * 10**3 + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)

    ax.set_xlim(20, 500)
    ylim = ax.get_ylim()  
    ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max(), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'white', alpha = 0.2, zorder = 0, label = 'On')
    ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0, label = 'Off')
    ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.set_ylim(ylim)
            
    ## Plot PEC on right
    
    ax = axes[1][1]
    offset = 0.0
        
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        if spectrum.wavelength > 450:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ax2 = ax.twinx()
    # ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
    ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
    ax2.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax2.yaxis.tick_left()
    ax.set_xlim(400, 900)
        
    axes[-1][0].set_xlabel('Time (s)', fontsize = 'x-large')
    axes[0][0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
    axes[1][0].set_ylabel('Potential (mV)', fontsize = 'x-large')
    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([-0.03, 0.15, 0.05, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 50 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
    cbar_ax.yaxis.tick_left()
    cbar_ax.yaxis.set_label_position("left")
    fig.suptitle(particle.name + ' OCP Photocurrent', fontsize = 'xx-large', horizontalalignment='center', x = 0.45, y = 0.94)
    
    
    axes[-1][1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    axes[0][1].set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[1][1].set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[0][1].yaxis.set_label_position("right")
    axes[1][1].yaxis.set_label_position("right")    
    for ax in axes:
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        print
        fig.savefig(save_dir + particle.name + ' OCP Photocurrent' + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
        
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str(particle.name)]
        except:
            group = save_h5.create_group(str(particle.name))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = 'OCP Photocurrent_jpg_%d', data = fig_array)


# plot_ca_pec(particles[1])
# plot_ocp_pec(particles[1])


#%% EChem & Photocurrent Processing functions


def process_power_calibration(particle):

    group = particle.power_calibration_group ## Should be h5 group with individual data points
    
    power_dict = {} 

    for key in group.keys():

        power = group[key]
        wavelength = int(power.attrs['wavelength'])
        power = float(np.array(power))
        new_dict = {wavelength : power}
        power_dict.update(new_dict)

    particle.power_dict = power_dict
    particle.__dict__.pop('power_calibration_group')


def process_ca(particle):
    
    spectra = particle.ca_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        voltage = voltage[0]
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.x += particle.ca_offset[j]
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = name
        spectrum.__dict__.pop('dset')
    
        ## Baseline, smooth
        spectrum.truncate(5, None)    
        ## Cottrell fit baseline subtraction
        x = spectrum.x
        y = spectrum.y
        
        try:
            popt, pcov = curve_fit(f = cottrell, 
                                xdata = x,
                                ydata = y)
            
        except:
            print('\nFit Error')
            print('\n'+spectrum.name+'\n')
            popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]  

        # spectrum.truncate(100, None)

        spectrum.y_baseline = cottrell(spectrum.x, *popt)
        spectrum.y_baselined = spectrum.y - spectrum.y_baseline
        spectrum.y_smooth = spectrum.y_baselined
        spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 1000, fs = 40000)
                    
        ## Attribute handling
        spectrum.wavelength = int(name[7:10])
        spectrum.fwhm = 50
        spectrum.voltage = np.round(voltage, 1)
        spectrum.toggle = 50
        
        
        # Calculate PEC (light on - light off) at each toggle point of CA curve
        
        toggles = int(spectrum.x.max()/(spectrum.toggle*2))
        spectrum.toggles = toggles
        pec = []
        ipce = []
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 50 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 50 : dark_index])
            current = light_current - dark_current
            pec.append(current/particle.power_dict[spectrum.wavelength]) # Power normalized photocurrent in A/mW
            ipce.append((np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * particle.power_dict[spectrum.wavelength]) * 100) # IPCE as %
        spectrum.pec = pec
        spectrum.ipce = ipce
        
        spectra[j] = spectrum
    
    particle.ca_spectra = spectra
    
    
def process_ocp_current(particle):

    spectra = particle.ocp_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.x += 0
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = 'currents_' + name
        spectrum.__dict__.pop('dset')
             
        # spectrum.truncate(100, None)
        spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 40000)
                    
        ## Attribute handling
        spectrum.wavelength = int(name[name.find('nm') - 3:name.find('nm')])
        spectrum.fwhm = 50
        spectrum.toggle = 50
        
        
        # Calculate PEC (light on - light off) at each toggle point of CA curve
        
        toggles = int(spectrum.x.max()/(spectrum.toggle*2))
        spectrum.toggles = toggles
        pec = []
        ipce = []
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 50 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 50 : dark_index])
            current = light_current - dark_current
            pec.append(current/particle.power_dict[spectrum.wavelength]) # Power normalized photocurrent in A/mW
            ipce.append((np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * particle.power_dict[spectrum.wavelength]) * 100) # IPCE as %
        spectrum.pec = pec
        spectrum.ipce = ipce
        
        spectra[j] = spectrum
        
    particle.ocp_spectra = spectra
    
    
def process_ocp_voltage(particle):

    spectra = particle.ocp_spectra_voltage ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
                
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        potential = spectrum.attrs['Potential (V)']
        timestamp = spectrum.attrs['creation_timestamp']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(potential), particle_name = particle.name)
        spectrum.x += 0
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = 'voltages_' + name
        spectrum.voltage = 'OCP'
        spectrum.__dict__.pop('dset')
    
        ## Baseline, smooth
        spectrum.truncate(20, None)    
        ## Cottrell fit baseline subtraction
        x = spectrum.x
        y = spectrum.y
        
        # try:
        #     popt, pcov = curve_fit(f = cottrell, 
        #                         xdata = x,
        #                         ydata = y)
        #     spectrum.y_baseline = cottrell(spectrum.x, *popt)

        # except:
        #     print('\nFit Error')
        #     print('\n'+particle.name + ': ' + spectrum.name+'\n')
        spectrum.y_baseline = spt.baseline_als(y = spectrum.y, lam = 1e6, p = 1e-5)
        # spectrum.truncate(100, None)

        spectrum.y_baselined = spectrum.y - spectrum.y_baseline
        spectrum.y_smooth = spectrum.y_baselined
        spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 1000, fs = 40000)

        ## Attribute handling
        spectrum.wavelength = int(name[name.find('nm') - 3:name.find('nm')])
        spectrum.fwhm = 50
        # spectrum.voltage = np.round(voltage, 1)
        spectrum.sample = 'MLAgg'
        spectrum.toggle = 50
        
        
        # Calculate PEC (light on - light off) at each toggle point of CA curve
        
        toggles = int(spectrum.x.max()/(spectrum.toggle*2))
        spectrum.toggles = toggles
        pec = []
        ipce = []
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 50 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 50 : dark_index])
            current = light_current - dark_current
            pec.append(current/particle.power_dict[spectrum.wavelength]) # Power normalized photovoltage in V/mW
            ipce.append((np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * particle.power_dict[spectrum.wavelength]) * 100) # IPCE as %, but doesn't make sense for voltage...
        spectrum.pec = pec
        spectrum.ipce = ipce
        
        spectra[j] = spectrum
        
    particle.ocp_spectra_voltage = spectra        
    

def process_cv(particle):
    
    spectra = particle.cv_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        print(name)
        timestamp = spectrum.attrs['creation_timestamp']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = name
        spectrum.__dict__.pop('dset')
        
        # Baseline, smooth

        # spectrum.truncate(100, None)

        spectrum.y_baseline = 0
        spectrum.y_baselined = spectrum.y - spectrum.y_baseline
        spectrum.y_smooth = spectrum.y_baselined
        # spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 1000, fs = 40000)
                    
        spectra[j] = spectrum
    
    particle.cv_spectra = spectra
    

def process_lsv(particle):
    
    spectra = particle.lsv_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        scanrate = spectrum.attrs['Scanrate (V/s)']
        times = spectrum.attrs['Time (s)']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = name
        spectrum.__dict__.pop('dset')
        
        ## Attribute handling
        if 'dark' in name:
            spectrum.wavelength = 0
        else:
            spectrum.wavelength = int(name[name.find('nm') - 3:name.find('nm')])
        spectrum.fwhm = 50
        spectrum.toggle = 10.05
        spectrum.toggle = spectrum.toggle * scanrate
    
        # Baseline, smooth
        # spectrum.truncate(100, None)

        spectrum.y_baseline = 0
        spectrum.y_baselined = spectrum.y - spectrum.y_baseline
        spectrum.y_smooth = spectrum.y_baselined
        # spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 1000, fs = 40000)
                    
        spectra[j] = spectrum
    
    particle.lsv_spectra = spectra
    

#%%

''' Run to here and import spydata'''


#%%

''' Processing '''


#%% Photocurrent

particles = []


# Au mirror

this_particle = Particle()
this_particle.name = '100 nm Au mirror'
this_particle.size = 100
this_particle.power_calibration_group = my_h5['power_calibration_0']
group = my_h5['100nm_Au_1']
skip_spectra = ['PEC_OCP_750nm_50nmFWHM_toggle_50s_OCP_0',
                'PEC_OCP_775nm_50nmFWHM_toggle_50s_OCP_0',
                'PEC_OCP_800nm_50nmFWHM_toggle_50s_OCP_0',
                'PEC_OCP_825nm_50nmFWHM_toggle_50s_OCP_0',
                'PEC_OCP_850nm_50nmFWHM_toggle_50s_OCP_0']

## CA Spectra
spectra = []
for key in group.keys():
    if 'PEC_CA' in key and key not in skip_spectra:   
            spectrum = group[key]
            spectra.append(spectrum)              
spectra.sort(key=lambda x: x.attrs['creation_timestamp'], reverse = False)
this_particle.ca_spectra = spectra
this_particle.ca_offset = np.zeros(len(spectra))
this_particle.ca_offset[0:32] = -7

## CV Spectra
spectra = []
for key in group.keys():
    if 'CV' in key and key not in skip_spectra:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.cv_spectra = spectra

## LSV Spectra
spectra = []
for key in group.keys():
    if 'LSV' in key and key not in skip_spectra and 'test' not in key:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.lsv_spectra = spectra

## OCP Current spectra
spectra = []
for key in group.keys(): 
    if 'OCP' in key and key not in skip_spectra and 'test' not in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra = spectra
            
## OCP voltage spectra
spectra = []
for key in group.keys():
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra_voltage = spectra

particles.append(this_particle)


# Bare FTO

this_particle = Particle()
this_particle.name = 'Bare FTO'
this_particle.size = 0
this_particle.power_calibration_group = my_h5['power_calibration_0']
group = my_h5['Bare_FTO_0']
skip_spectra = []

## CA Spectra
spectra = []

for key in group.keys():
    if 'PEC_CA' in key and key not in skip_spectra:   
            spectrum = group[key]
            spectra.append(spectrum)                   
this_particle.ca_spectra = spectra
this_particle.ca_offset = np.zeros(len(spectra))

## CV Spectra
spectra = []
for key in group.keys():
    if 'CV' in key and key not in skip_spectra:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.cv_spectra = spectra

## LSV Spectra
spectra = []
for key in group.keys():
    if 'LSV' in key and key not in skip_spectra and 'test' not in key:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.lsv_spectra = spectra

## OCP Current spectra
spectra = []
for key in group.keys(): 
    if 'OCP' in key and key not in skip_spectra and 'test' not in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra = spectra
            
## OCP voltage spectra
spectra = []
for key in group.keys():
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra_voltage = spectra

particles.append(this_particle)


for particle in particles:
    
    process_power_calibration(particle)
    process_ca(particle)
    process_ocp_current(particle)
    process_ocp_voltage(particle)
    process_cv(particle)
    process_lsv(particle)
    
    if save == True:
        save_ca(particle)
        save_ocp(particle)
        save_cv(particle)
        save_lsv(particle)

control_particles = particles


#%%

''' Save spydata here '''


#%% Plotting

''' Run from here on for plotting after importing spydata'''


#%% Plot PEC & Echem

for particle in particles:
    plot_ca_pec(particle, save = save, save_to_h5 = save_to_h5)
    plot_cv(particle, save = save, save_to_h5 = save_to_h5)
    plot_ocp_pec(particle, save = save, save_to_h5 = save_to_h5)
    plot_lsv(particle, save = save, save_to_h5 = save_to_h5)
    plot_subtracted_ipce(particle, save = save, save_to_h5 = save_to_h5)


#%% Testing background subtraction

# Cottrell equation background subtraction

# group = my_h5['BPDT_20nm_MLAgg_0_']['Potentiostat']   
# spectrum = group['PEC_CA_500nm_50nmFWHM_toggle_50s_CA_0.4V_0']
# timestamp = spectrum.attrs['creation_timestamp']
# voltage = spectrum.attrs['Levels_v (V)']
# # voltage = 0
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# spectrum.timestamp = timestamp[timestamp.find('T')+1:]
# # spectrum.name = key
# # spectrum.truncate(20,None)

# x = spectrum.x
# y = spectrum.y

# try:
#     popt, pcov = curve_fit(f = cottrell, 
#                     xdata = x,
#                     ydata = y)
        
# except:
#     print('Fit Error')
#     # popt = [15.5,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 6), 262]
#     popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]
#     # popt = [0.001,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 1), 262]

# fit_y = cottrell(spectrum.x, *popt)

# plt.plot(spectrum.x, spectrum.y, color = 'black')
# plt.plot(spectrum.x, fit_y, color = 'red')
# # plt.plot(spectrum.x, spectrum.y - fit_y, color = 'blue')
# # plt.xlim(20, 500)
# # plt.ylim(-2, -1.5)

  
#%% LSV Background Testing

# def process_lsv(particle):
    
#     spectra = particle.lsv_spectra ## Should be list of h5 groups with spectra you want
    
#     for j, spectrum in enumerate(spectra[3:4]):
        
#         name = str(spectrum)
#         name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
#         timestamp = spectrum.attrs['creation_timestamp']
#         scanrate = spectrum.attrs['Scanrate (V/s)']
#         times = spectrum.attrs['Time (s)']
#         spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
#         spectrum.timestamp = timestamp[timestamp.find('T')+1:]
#         spectrum.name = name
#         spectrum.__dict__.pop('dset')
        
#         ## Attribute handling
#         if 'dark' in name:
#             spectrum.wavelength = 0
#         else:
#             spectrum.wavelength = int(name[name.find('nm') - 3:name.find('nm')])
#         spectrum.fwhm = 50
#         spectrum.toggle = 10.05
#         spectrum.toggle = spectrum.toggle * scanrate
    
    
#         # Baseline, smooth
    
#         ## Iterative polynomial fit for LSV background

#         background = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 10, max_iter = 1000, tol = 1, mask_initial_peaks = True)[0]
#         spectrum.background = background
        

#         ## Plot
#         fig, ax = plt.subplots(1,1,figsize=[12,9])
#         ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#         ax.set_ylabel('SERS Intensity (cts/mW/s)')
#         offset = 50000
#         spectrum.plot(ax = ax, plot_y = spectrum.y)
#         ax.set_xlim(0.3, 0.4)
#         ax.set_ylim(0, 0.2e-6)
#         spectrum.plot(ax = ax, plot_y = spectrum.background)
        
#         fig, ax = plt.subplots(1,1,figsize=[12,9])
#         ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#         ax.set_ylabel('SERS Intensity (cts/mW/s)')
#         offset = 50000
#         spectrum.plot(ax = ax, plot_y = spectrum.y)
#         # ax.set_xlim(0.3, 0.4)
#         # ax.set_ylim(0, 0.2e-6)
#         spectrum.plot(ax = ax, plot_y = spectrum.background)
        
#         fig, ax = plt.subplots(1,1,figsize=[12,9])
#         ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#         ax.set_ylabel('SERS Intensity (cts/mW/s)')
#         offset = 50000
#         # spectrum.plot(ax = ax, plot_y = spectrum.y)
#         # ax.set_xlim(0.3, 0.4)
#         # ax.set_ylim(0, 0.2e-6)
#         # spectrum.plot(ax = ax, plot_y = spectrum.background)
        
#         ax.plot(spectrum.x, spectrum.y - background)
#         # fig.suptitle(particle.name)    
#         # ax.legend()
#         # ax.set_xlim(350, 1900)
#         # ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
#         # plt.tight_layout(pad = 0.8)      
        
#         # spectrum.truncate(100, None)

#         spectrum.y_baseline = 0
#         spectrum.y_baselined = spectrum.y - spectrum.y_baseline
#         spectrum.y_smooth = spectrum.y_baselined
#         # spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 1000, fs = 40000)
                    
#         spectra[j] = spectrum
    
#     particle.lsv_spectra = spectra
    


# def plot_lsv(particle, save = False, save_to_h5 = False):
    
#     spectra = particle.lsv_spectra
#     # power_dict = particle.power_dict
#     size = particle.size
#     df_avg = particle.df_avg
    
    
#     fig, ax = plt.subplots(1, 1, figsize=[16,8])
#     my_cmap = plt.get_cmap('nipy_spectral')
#     cmap = my_cmap
#     norm = mpl.colors.Normalize(vmin=420, vmax=830)
#     offset = 0.0
    
#     for i, spectrum in enumerate(spectra[3:4]):
        
#         linetype = 'solid'
                
#         color = cmap(norm(spectrum.wavelength))  
#         ax.plot(spectrum.x, ((spectrum.y_smooth* 10**6) + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)


    
    
#     ## Light on/off bars
#     # ax.set_xlim(0.1, 0.4)
#     # ax.set_ylim(-0.01, 0.2)
#     ylim = ax.get_ylim()  
#     xlim = ax.get_xlim()    
#     ax.bar(np.arange((spectrum.x_raw.min() + spectrum.toggle/2), spectrum.x_raw.max() + (spectrum.toggle/2), spectrum.toggle*2) - 0.0007, height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
#     # ax.text(s = 'On', x = (spectrum.x.min() + spectrum.toggle), y = ylim[0], fontsize = 'small')
#     # ax.text(s = 'Off', x = (spectrum.x.min()), y = ylim[0], fontsize = 'small')
#     ax.set_ylim(ylim[0], ylim[1])
#     ax.set_xlim(xlim[0], xlim[1])    
    
#     ax.set_xlabel('Voltage (V)', fontsize = 'large')
#     ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
#     fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
#     cbar_ax.set_ylabel('Centre Wavelength (nm) - 50 nm FWHM', rotation=270, fontsize = 'large', labelpad = 40)
#     cbar_ax.yaxis.set_label_position("right")
#     fig.suptitle(particle.name + ' LSV', fontsize = 'large', horizontalalignment='center', x = 0.55, y = 0.95)
    

#     ## Save
#     if save == True:
#         save_dir = get_directory(str(size) + ' nm')
#         fig.savefig(save_dir + particle.name + ' LSV' + '.svg', format = 'svg', bbox_inches='tight')
#         plt.close(fig)
        
        
#     ## Save to h5 file as jpg
#     if save_to_h5 == True:
#         try:
#             group = save_h5[str(particle.name)]
#         except:
#             group = save_h5.create_group(str(particle.name))

#         canvas = FigureCanvas(fig)
#         canvas.draw()
#         fig_array = np.array(canvas.renderer.buffer_rgba())
#         group.create_dataset(name = 'LSV_zoom_jpg_%d', data = fig_array)



# this_particle = Particle()
# this_particle.name = 'BPDT 57 nm MLAgg'
# this_particle.size = 57
# my_h5 = h5_dict[this_particle.size]
# this_particle.df_avg = df_avg_dict[this_particle.size]
# this_particle.power_calibration_group = my_h5['power_calibration_0']
# group = my_h5['BPDT_57nm_MLAgg_2']
# skip_spectra = ["currents_PEC_OCP_450nm_50nmFWHM_toggle_50s_OCP_0"]


# ## LSV Spectra
# spectra = []
# for key in group.keys():
#     if 'LSV' in key and key not in skip_spectra and 'test' not in key:        
#             spectrum = group[key]
#             spectra.append(spectrum)           
# this_particle.lsv_spectra = spectra
# process_lsv(this_particle)
# # plot_lsv(this_particle)
