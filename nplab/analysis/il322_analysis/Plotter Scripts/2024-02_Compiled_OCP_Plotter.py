# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter to compare OCP & extinction for various samples

Need to import data from previous PEC processing scripts

Will continue to update with OCP data for the paper

Data from:


Copied from 2024-12-04_Compiled PEC Plotter

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


skip_ca_particles = []
skip_ocp_particles = []

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


#%%

def get_directory():
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-02_Compiled OCP Analysis\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%%

''' Run to here and import processed spydata'''

#%%

for particle in bpdt_particles:
    particle.transmittance = None
    for spectrum in particle.ocp_spectra_voltage:
        spectrum.delay_time = 0
    
for particle in tpdt_particles:
    particle.transmittance = None
    for spectrum in particle.ocp_spectra_voltage:
        spectrum.delay_time = 0

for particle in cots_particles:
    particle.transmittance = None  
    for spectrum in particle.ocp_spectra_voltage:
        spectrum.delay_time = 0
    
for particle in mtapp_particles:
    particle.transmittance = None  
    for spectrum in particle.ocp_spectra_voltage:
        spectrum.delay_time = 0
        
for particle in sam_particles:
    particle.transmittance = None  
    for spectrum in particle.ocp_spectra_voltage:
        spectrum.delay_time = 0

#%%

particles = cotapp_particles + bpdt_particles+ control_particles + cots_particles + tpdt_particles + mtapp_particles + sam_particles


# Color

my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
color_dict = {20 : cmap(0), 35 : cmap(2), 57 : cmap(1), 70 : cmap(3), 'ITO' : 'grey', 'Au mirror' : 'goldenrod'}


#%% EChem plotting functions

def plot_cv(particle, save = False, save_to_h5 = False):
    
    spectra = particle.cv_spectra
    power_dict = particle.power_dict
    size = particle.size
    df_avg = particle.df_avg
    
    
    fig, ax = plt.subplots(1, 1, figsize=[14,8])
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot CV spectra
    
    for spectrum in spectra:
        
        ax.plot(spectrum.x, spectrum.y * 10**6, label = int(spectrum.scanrate * 1000))
        
        
    ax.set_xlabel('Potential v. Ag/AgCl (V)', fontsize = 'large')
    ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
    # fig.subplots_adjust(right=0.9)
    fig.suptitle(particle.name + ' CV', fontsize = 'x-large', horizontalalignment='center')   
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    ax.legend(title = 'Scanrate (mV/s)')  
    
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
    df_avg = particle.df_avg
    
    
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
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 15 nm FWHM', rotation=270, fontsize = 'large', labelpad = 40)
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
            
       
def plot_ca_pec(particle, angle = None, save = False, save_to_h5 = False, xlim = [40, 500]):
    
    spectra = particle.ca_spectra
    power_dict = particle.power_dict
    size = particle.size
    df_avg = particle.df_avg
    
    voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    if angle is None:
        angle = 0
    
    fig, axes = plt.subplots(len(voltages), 2, figsize=[17,16], width_ratios = (2,1))
    fig.subplots_adjust(wspace = 0.3)
    
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
        
        i_xlim = [np.where(np.round(spectra[0].x, 1) == xlim[0])[0][0], np.where(np.round(spectra[0].x, 1) == xlim[1])[0][0]]   
        
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            if spectrum.voltage != voltage:
                continue
            if spectrum.angle != angle:
                continue
            if spectrum.wavelength < 7000:
                color = cmap(norm(spectrum.wavelength))  
                ax.plot(spectrum.x[i_xlim[0]:i_xlim[1]], (((spectrum.y_smooth[i_xlim[0]:i_xlim[1]] - spectrum.y_smooth[np.where(np.round(spectrum.x, 1) == 50.0)]) * 10**6) + i * offset), label = spectrum.voltage, color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
        
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
        
        wavelengths = []
        ipces = []
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            if spectrum.voltage != voltage:
                continue
            if spectrum.angle != angle:
                continue
            
            ## Plot IPCE
            color = cmap(norm(spectrum.wavelength))  
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                ipces.append(np.mean(spectrum.ipce))
        ax.plot(wavelengths, ipces, color = 'black', zorder = 0, linewidth = 2)

        ## Plot DF
        if df_avg is not None:        
            ax2 = ax.twinx()
            ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.3, zorder = 0)
            if j == int(len(voltages)/2):
                ax2.set_ylabel('DF Intensity', fontsize = 'x-large', labelpad = 5)
                ax2.yaxis.set_label_position("left")
            ax2.yaxis.tick_left()
        ax.set_xlim(400, 900)
        ax.yaxis.tick_right()

    axes[-1][0].set_xlabel('Time (s)', fontsize = 'x-large')
    axes[int((len(axes) - 1)/2)][0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0, 0.15, 0.05, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 15 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
    cbar_ax.yaxis.tick_left()
    cbar_ax.yaxis.set_label_position("left")
    title = particle.name + ' CA Photocurrent'
    if angle != 0:
        title = particle.name + ' CA Photocurrent ' + str(angle) + 'deg'
    fig.suptitle(title, fontsize = 'xx-large', horizontalalignment='center', x = 0.45, y = 0.94)
    
    
    axes[-1][1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    axes[int((len(axes) - 1)/2)][1].set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
    axes[int((len(axes) - 1)/2)][1].yaxis.set_label_position("right")
    for ax in axes:
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
           
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
    df_avg = particle.df_avg
    
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
            if spectrum.wavelength > 40: # Here
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
        ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        if j == int(len(voltages)/2):
            ax2.set_ylabel('DF Intensity', fontsize = 'x-large')
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
        df_line = ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
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
    fig.legend(handles = [scatt_plus, scatt_minus, au_line[0], subtracted_line[0], df_line[0]], labels = ['IPCE +', 'IPCE -', 'Au abs', 'IPCE - Au abs', 'DF'], ncols = 5, bbox_to_anchor=(0.93, 0.99))

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

        
def plot_ocp_pec(particle, angle = None, save = False, save_to_h5 = False, xlim = [500, 1500]):
    
    spectra = particle.ocp_spectra
    power_dict = particle.power_dict
    size = particle.size
    if angle is None:
        angle = 0
    df_avg = particle.df_avg    
    
    fig, axes = plt.subplots(2, 2, figsize=[18,13], width_ratios = (2,1))
    fig.subplots_adjust(wspace = 0.3)
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot OCP Currents on top
    
    ## Plot Light Toggle OCP on left
    
    ax = axes[0][0]
    offset = 0.0
    
    ax.xaxis.set_ticklabels([])
    i_xlim = [np.where(np.round(spectra[0].x, 1) == xlim[0])[0][0], np.where(np.round(spectra[0].x, 1) == xlim[1])[0][0]]    
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.angle != angle:
            continue
        linetype = 'solid'
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x[i_xlim[0]:i_xlim[1]], (((spectrum.y[i_xlim[0]:i_xlim[1]] - spectrum.y_smooth[np.where(np.round(spectrum.x, 1) == 500)]) * 10**6) + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ## Light on/off bars
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()    
    ax.bar(np.arange(spectrum.delay_time + (spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
    ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')    
    ax.set_ylim(ylim)
            
    ## Plot PEC on right
    
    ax = axes[0][1]
    offset = 0.0
    
    ax.xaxis.set_ticklabels([])
    
    wavelengths = []
    ipces = []
    for i, spectrum in enumerate(spectra):
        
        if spectrum.angle != angle:
            continue
        linetype = 'solid'
            
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
            ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            wavelengths.append(spectrum.wavelength)
            ipces.append(np.mean(spectrum.ipce))
        ax.plot(wavelengths, ipces, color = 'black', zorder = 0, linewidth = 2)
        
    ## Plot DF
    if df_avg is not None:
        ax2 = ax.twinx()
        ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        ax2.set_ylabel('DF Intensity', fontsize = 'x-large', labelpad = 5)
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        
    ax.set_xlim(400, 900)
    ax.yaxis.tick_right()

    
    # Plot OCP Voltages on bottom
    
    spectra = particle.ocp_spectra_voltage
    
    ## Plot Light Toggle OCP on left
    
    ax = axes[1][0]
    offset = 0.0
    i_xlim = [np.where(np.round(spectra[0].x, 1) == xlim[0])[0][0], np.where(np.round(spectra[0].x, 1) == xlim[1])[0][0]]    

    
    for i, spectrum in enumerate(spectra):
        if spectrum.angle != angle:
            continue
        linetype = 'solid'
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x[i_xlim[0]:i_xlim[1]], ((spectrum.y_smooth[i_xlim[0]:i_xlim[1]] - spectrum.y_smooth[np.where(np.round(spectrum.x, 1) == 600.0)]) * 10**3 + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()  
    ax.bar(np.arange(spectrum.delay_time + (spectrum.toggle/2), spectrum.x.max(), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0, label = 'On')
    # ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0, label = 'Off')
    ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.set_ylim(ylim)
            
    ## Plot PEC on right
    
    ax = axes[1][1]
    offset = 0.0
        
    wavelengths = []
    pvs = []
    for i, spectrum in enumerate(spectra):
        
        if spectrum.angle != angle:
            continue
        linetype = 'solid'
            
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            wavelengths.append(spectrum.wavelength)
            pvs.append(np.abs(np.mean(spectrum.pec)) * 10**3)
        ax.plot(wavelengths, pvs, color = 'black', zorder = 0, linewidth = 2)
            
    ## Plot DF
    if df_avg is not None:
        ax2 = ax.twinx()
        ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
        ax2.set_ylabel('DF Intensity', fontsize = 'x-large', labelpad = 5)
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        
    ax.set_xlim(400, 900)
    ax.yaxis.tick_right()
        
    axes[-1][0].set_xlabel('Time (s)', fontsize = 'x-large')
    axes[0][0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
    axes[1][0].set_ylabel('Potential (mV)', fontsize = 'x-large')
    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([-0.03, 0.15, 0.05, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Centre Wavelength (nm) - 15 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
    cbar_ax.yaxis.tick_left()
    cbar_ax.yaxis.set_label_position("left")
    title = particle.name + ' OCP Photocurrent'
    if angle != 0:
        title = particle.name + ' OCP Photocurrent ' + str(angle) + 'deg'
    fig.suptitle(title, fontsize = 'xx-large', horizontalalignment='center', x = 0.45, y = 0.94)
    
    
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
        fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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


def plot_ocp_photovoltage(particle, angle = None, save = False, save_to_h5 = False, xlim = [600, 1500]):

    spectra = particle.ocp_spectra_voltage
    power_dict = particle.power_dict
    size = particle.size
    matplotlib.rcParams.update({'font.size': 36})

    df_avg = particle.transmittance 
    
    fig, axes = plt.subplots(1, 2, figsize=[22,9], width_ratios = (2,1.2))
    fig.subplots_adjust(wspace = 0.25)
    
    my_cmap = plt.get_cmap('nipy_spectral')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=420, vmax=830)
    offset = 0.0
    
    
    # Plot Light Toggle OCP on left
    
    ax = axes[0]
    offset = 0.0
    i_xlim = [np.where(np.round(spectra[0].x, 1) == xlim[0])[0][0], np.where(np.round(spectra[0].x, 1) == xlim[1])[0][0]]    
    
    
    for i, spectrum in enumerate(spectra):

        if spectrum.delay_time is None:
            spectrum.delay_time = 0
        linetype = 'solid'
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x[i_xlim[0]:i_xlim[1]], ((spectrum.y_smooth[i_xlim[0]:i_xlim[1]] - spectrum.y_smooth[np.where(np.round(spectrum.x, 1) == xlim[0])]) * 10**3 + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()  
    ax.bar(np.arange(spectrum.delay_time + (spectrum.toggle/2), spectrum.x.max(), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0, label = 'Off')
    ax.xaxis.set_ticklabels(list(map(int, ax.get_xticks() - xlim[0])))
    ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 0) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'medium')
    ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'medium')
    ax.set_ylim(ylim)
    
            
    # Plot PV on right
    
    ax = axes[1]
    offset = 0.0
        
    wavelengths = []
    pvs = []
    for i, spectrum in enumerate(spectra):
        
        # if spectrum.angle != angle:
        #     continue
        linetype = 'solid'
            
        ## Plot
        color = cmap(norm(spectrum.wavelength))
        # color = 'black'
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 3)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 2)
            wavelengths.append(spectrum.wavelength)
            pvs.append((np.mean(spectrum.pec)) * 10**3)
        ax.plot(wavelengths, pvs, color = 'black', zorder = 2, linewidth = 2)    
    pvs = np.array(pvs)
    wavelengths = np.array(wavelengths)
    ax.fill_between(wavelengths, pvs, where = pvs > 0, interpolate = True, color = 'red', alpha = 0.25, zorder = 0)
    ax.fill_between(wavelengths, pvs, where = pvs < 0, interpolate = True, color = 'blue', alpha = 0.25, zorder = 0)
        
    ## Plot DF
    if df_avg is not None:
        ax2 = ax.twinx()
        ax2.plot(df_avg.x, 100 - df_avg.y, color = 'black', alpha = 0.3, zorder = 1)
        ax2.set_ylabel('Extinction (%)', fontsize = 'large', labelpad = 5)
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        ymax = (np.percentile((100 - df_avg.y), 95))
        ymin = (np.percentile((100 - df_avg.y), 1))
        yspread = ymax - ymin
        ax2.set_ylim(ymin - (.05 * yspread), ymax + (.05 * yspread))
        
    ax.set_xlim(400, 900)
    ax.yaxis.tick_right()
    ax.set_zorder(1)    
    ax.patch.set_visible(False)
    
    axes[0].set_xlabel('Time (s)', fontsize = 'large')
    axes[0].set_ylabel('Potential (mV)', fontsize = 'large')
    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.001, 0.15, 0.02, 0.7])
    # fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    # cbar_ax.set_ylabel('Wavelength (nm) - 15 nm FWHM', rotation=90, fontsize = 'medium', labelpad = 20)
    # cbar_ax.yaxis.tick_left()
    # cbar_ax.yaxis.set_label_position("left")
    title = particle.name + ' OCP Photovoltage'
    if angle != 0:
        title = particle.name + ' OCP Photovoltage ' + str(angle) + 'deg'
    # fig.suptitle(title, fontsize = 'large', horizontalalignment='center', x = 0.47, y = 0.94)
    
    
    axes[1].set_xlabel('Wavelength (nm)', fontsize = 'large')
    axes[1].set_ylabel('Photovoltage (mV/mW)', fontsize = 'large', rotation = 270, labelpad = 60)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_label_position("right")    
    
    axes[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    axes[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    axes[1].hlines(xmin = 400, xmax = 900, y = 0, color = 'black', zorder = 0, linestyle = 'dashed')
    
    plt.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
        group.create_dataset(name = 'OCP Photovoltage_jpg_%d', data = fig_array)
        
        
# plot_ca_pec(particles[1])
# plot_ocp_photovoltage(particles[2])

# N = (power * 1) * (wavelength/h * 2.998e17)



#%% Co-TAPP-SMe Plot OCP Photovoltage spectra

fig, ax = plt.subplots(1, 1, figsize=[18,12])
ax2 = ax.twinx()

# fig.subplots_adjust(wspace = 0.25)

for particle in particles:

    if particle not in cotapp_particles and particle not in control_particles:
        continue

    spectra = particle.ocp_spectra_voltage
    power_dict = particle.power_dict
    size = particle.size
    
    df_avg = particle.df_avg

    if particle not in control_particles:
        label = str(particle.size) + str(' nm')
    else:
        label = particle.name    

    # Plot PV
    
    offset = 0.0
        
    wavelengths = []
    pvs = []
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        ## Plot
        color = color_dict[particle.size]
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 200, marker = marker, linewidth = 5, zorder = 3)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 15, elinewidth = 3, capthick = 3, ecolor = color, zorder = 2)
            wavelengths.append(spectrum.wavelength)
            pvs.append((np.mean(spectrum.pec)) * 10**3)
    ax.plot(wavelengths, pvs, color = color, zorder = 2, linewidth = 5, label = label)    
    pvs = np.array(pvs)
    wavelengths = np.array(wavelengths)
    # ax.fill_between(wavelengths, pvs, where = pvs > 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
    # ax.fill_between(wavelengths, pvs, where = pvs < 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
        
    # Plot DF
    if df_avg is not None:
        ax2.plot(df_avg.x, df_avg.y_norm, color = color, alpha = 0.3, zorder = 1)
        # ymax = (np.percentile((100 - df_avg.y), 95))
        # ymin = (np.percentile((100 - df_avg.y), 1))
        # yspread = ymax - ymin
        # ax2.set_ylim(ymin - (.05 * yspread), ymax + (.05 * yspread))
        
ax.set_xlim(400, 900)
ax.yaxis.tick_right()
ax.set_zorder(1)    
ax.patch.set_visible(False)
ax.legend()


title = 'Co-TAPP-SMe MLAgg OCP Photovoltage'
fig.suptitle(title, fontsize = 'large', horizontalalignment = 'center')

ax2.set_ylabel('Scattering (a.u.)', fontsize = 'large', labelpad = 5)
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()

ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Photovoltage (mV/mW)', fontsize = 'large', labelpad = 60, rotation = 270)
ax.yaxis.set_label_position("right")

ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.hlines(xmin = 400, xmax = 900, y = 0, color = 'black', zorder = 0, linestyle = 'dashed')

## Save
if save == True:
    save_dir = get_directory(str(size) + ' nm')
    fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
    group.create_dataset(name = title + '_jpg_%d', data = fig_array)


#%% BPDT Plot OCP Photovoltage spectra

fig, ax = plt.subplots(1, 1, figsize=[16,12])
ax2 = ax.twinx()

# fig.subplots_adjust(wspace = 0.25)

for particle in particles:

    if particle not in bpdt_particles and particle not in control_particles:
        continue

    spectra = particle.ocp_spectra_voltage
    power_dict = particle.power_dict
    size = particle.size
    
    df_avg = particle.df_avg

    if particle not in control_particles:
        label = str(particle.size) + str(' nm')
    else:
        label = particle.name

        
    # Plot PV
    
    offset = 0.0
        
    wavelengths = []
    pvs = []
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        ## Plot
        color = color_dict[particle.size]
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 200, marker = marker, linewidth = 5, zorder = 3)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 15, elinewidth = 3, capthick = 3, ecolor = color, zorder = 2)
            wavelengths.append(spectrum.wavelength)
            pvs.append((np.mean(spectrum.pec)) * 10**3)
    ax.plot(wavelengths, pvs, color = color, zorder = 2, linewidth = 5, label = label)    
    pvs = np.array(pvs)
    wavelengths = np.array(wavelengths)
    # ax.fill_between(wavelengths, pvs, where = pvs > 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
    # ax.fill_between(wavelengths, pvs, where = pvs < 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
        
    # Plot DF
    if df_avg is not None:
        ax2.plot(df_avg.x, df_avg.y, color = color, alpha = 0.3, zorder = 1)
        # ymax = (np.percentile((100 - df_avg.y), 95))
        # ymin = (np.percentile((100 - df_avg.y), 1))
        # yspread = ymax - ymin
        # ax2.set_ylim(ymin - (.05 * yspread), ymax + (.05 * yspread))
        
ax.set_xlim(400, 900)
ax.yaxis.tick_right()
ax.set_zorder(1)    
ax.patch.set_visible(False)
ax.legend()


title = 'BPDT MLAgg OCP Photovoltage'
fig.suptitle(title, fontsize = 'large', horizontalalignment = 'center')

ax2.set_ylabel('Scattering (a.u.)', fontsize = 'large', labelpad = 5)
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()

ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Photovoltage (mV/mW)', fontsize = 'large', labelpad = 60, rotation = 270)
ax.yaxis.set_label_position("right")

ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.hlines(xmin = 400, xmax = 900, y = 0, color = 'black', zorder = 0, linestyle = 'dashed')

## Save
if save == True:
    save_dir = get_directory(str(size) + ' nm')
    fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
    group.create_dataset(name = title + '_jpg_%d', data = fig_array)
    
    
#%% SAM Particles Plot OCP Photovoltage spectra

fig, ax = plt.subplots(1, 1, figsize=[18,12])
# ax2 = ax.twinx()

# fig.subplots_adjust(wspace = 0.25)

for particle in particles:

    if particle not in sam_particles and particle not in control_particles:
        continue
    
    spectra = particle.ocp_spectra_voltage
    power_dict = particle.power_dict
    size = particle.size
    print(particle.name)
    #%%
    df_avg = particle.df_avg

        
    # Plot PV
    
    offset = 0.0
        
    wavelengths = []
    pvs = []
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        ## Plot
        color = color_dict[particle.size]
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 3)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax.errorbar(spectrum.wavelength,  (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 2)
            wavelengths.append(spectrum.wavelength)
            pvs.append((np.mean(spectrum.pec)) * 10**3)
    ax.plot(wavelengths, pvs, color = color, zorder = 2, linewidth = 2, label = particle.name)    
    pvs = np.array(pvs)
    wavelengths = np.array(wavelengths)
    # ax.fill_between(wavelengths, pvs, where = pvs > 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
    # ax.fill_between(wavelengths, pvs, where = pvs < 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
        
    ## Plot DF
    # if df_avg is not None:
    #     ax2.plot(df_avg.x, df_avg.y_norm, color = color, alpha = 0.3, zorder = 1)
    #     ymax = (np.percentile((100 - df_avg.y), 95))
    #     ymin = (np.percentile((100 - df_avg.y), 1))
    #     yspread = ymax - ymin
        # ax2.set_ylim(ymin - (.05 * yspread), ymax + (.05 * yspread))
        
ax.set_xlim(400, 900)
# ax.yaxis.tick_right()
ax.set_zorder(1)    
ax.patch.set_visible(False)
ax.legend()


title = 'BPDT MLAgg OCP Photovoltage'
fig.suptitle(title, fontsize = 'xx-large', horizontalalignment = 'center')

# ax2.set_ylabel('Extinction (%)', fontsize = 'x-large', labelpad = 5)
# ax2.yaxis.set_label_position("left")
# ax2.yaxis.tick_left()

ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', labelpad = 20)
# ax.yaxis.set_label_position("right")

ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
ax.hlines(xmin = 400, xmax = 900, y = 0, color = 'black', zorder = 0, linestyle = 'dashed')

## Save
if save == True:
    save_dir = get_directory(str(size) + ' nm')
    fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
    group.create_dataset(name = title + '_jpg_%d', data = fig_array)


#%% Plot OCP Photovoltage spectra of diff molecules


color_dict_molecules = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'CotS' : 'orange', 'ITO' : 'grey', 'Au mirror' : 'goldenrod', 'TPDT' : 'pink'}


for size in sizes:

    fig, ax = plt.subplots(1, 1, figsize=[18,12])
    # ax2 = ax.twinx()
    
    # fig.subplots_adjust(wspace = 0.25)
    
    for particle in particles:
    
        if particle.size != size and particle not in control_particles:
            continue
    
        spectra = particle.ocp_spectra_voltage
        power_dict = particle.power_dict
        
        df_avg = particle.df_avg
    
            
        # Plot PV
        
        offset = 0.0
            
        wavelengths = []
        pvs = []
        for i, spectrum in enumerate(spectra):
            
            linetype = 'solid'
                
            ## Plot
            
            for molecule in color_dict_molecules.keys():
                if molecule in particle.name:
                    color = color_dict_molecules[molecule]
                    break
                
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 3)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax.errorbar(spectrum.wavelength,  (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 2)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3)
        ax.plot(wavelengths, pvs, color = color, zorder = 2, linewidth = 2, label = particle.name)    
        pvs = np.array(pvs)
        wavelengths = np.array(wavelengths)
        # ax.fill_between(wavelengths, pvs, where = pvs > 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
        # ax.fill_between(wavelengths, pvs, where = pvs < 0, interpolate = True, color = color, alpha = 0.25, zorder = 0)
            
        ## Plot DF
        # if df_avg is not None:
        #     ax2.plot(df_avg.x, df_avg.y_norm, color = color, alpha = 0.3, zorder = 1)
        #     ymax = (np.percentile((100 - df_avg.y), 95))
        #     ymin = (np.percentile((100 - df_avg.y), 1))
        #     yspread = ymax - ymin
            # ax2.set_ylim(ymin - (.05 * yspread), ymax + (.05 * yspread))
            
    ax.set_xlim(400, 900)
    # ax.yaxis.tick_right()
    ax.set_zorder(1)    
    ax.patch.set_visible(False)
    ax.legend()
    
    
    title = str(size) + 'nm MLAgg OCP Photovoltage'
    fig.suptitle(title, fontsize = 'xx-large', horizontalalignment = 'center')
    
    # ax2.set_ylabel('Extinction (%)', fontsize = 'x-large', labelpad = 5)
    # ax2.yaxis.set_label_position("left")
    # ax2.yaxis.tick_left()
    
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', labelpad = 20)
    # ax.yaxis.set_label_position("right")
    
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    ax.hlines(xmin = 400, xmax = 900, y = 0, color = 'black', zorder = 0, linestyle = 'dashed')
    
    ## Save
    if save == True:
        save_dir = get_directory(str(size) + ' nm')
        fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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
        group.create_dataset(name = title + '_jpg_%d', data = fig_array)



#%% Plot DF for diff molecules all same AuNP size

all_particles = cots_particles + cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles
molecules = ['CotS', 'Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
# sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'CotS' : 'orange'}
# my_cmap = plt.get_cmap('tab10')
# cmap = my_cmap
matplotlib.rcParams.update({'font.size': 36})


for size in sizes[0:1]:

    fig, ax = plt.subplots(1, 1, figsize=[16,12])
    ax.set_title(str(size) + 'nm MLAgg Dakrfield ', fontsize = 'large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
    ax.set_ylabel('Normalized Intensity', fontsize = 'large')

    for particle in all_particles:    

        print(particle.name)
        if particle.size != size:
            continue

        print('yes')
        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule
                print(molecule)
                print(particle.name)
                
        color = (color_dict_molecules[this_molecule])         
        
        ## Plot DF
        particle.df_avg.y_norm = (particle.df_avg.y - particle.df_avg.y.min())/(particle.df_avg.y.max() - particle.df_avg.y.min())
        particle.df_avg.y_smooth_norm = spt.butter_lowpass_filt_filt(particle.df_avg.y_norm, cutoff = 1000, fs = 40000)
        ax.plot(particle.df_avg.x, particle.df_avg.y_norm, color = color, alpha = 1, zorder = 0, label = this_molecule, linewidth = 4)

    ## Make pretty
    ax.set_xlim(450, 850)
    ax.set_ylim(-0.1,1.1)
    ax.legend(ncols = 2, fontsize = 'medium')
    
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + 'nm MLAgg DF'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
#%% Plot DF for same molecule at diff AuNP size

all_particles = cots_particles + cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles
molecules = ['CotS', 'Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
# sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'CotS' : 'orange'}
# my_cmap = plt.get_cmap('tab10')
# cmap = my_cmap
matplotlib.rcParams.update({'font.size': 36})


for molecule in molecules:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax.set_title(str(molecule) + ' MLAgg Dakrfield ', fontsize = 'large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
    ax.set_ylabel('Normalized Intensity', fontsize = 'large')

    for particle in all_particles:    

        if molecule not in particle.name:
            continue

        # for size in sizes:
        #     if size == particle.size:
        #         this_size = size
                
        color = (color_dict[particle.size])         
        
        ## Plot DF
        particle.df_avg.y_norm = (particle.df_avg.y - particle.df_avg.y.min())/(particle.df_avg.y.max() - particle.df_avg.y.min())
        particle.df_avg.y_smooth_norm = spt.butter_lowpass_filt_filt(particle.df_avg.y_norm, cutoff = 1000, fs = 40000)
        ax.plot(particle.df_avg.x, particle.df_avg.y_norm, color = color, alpha = 1, zorder = 0, label = str(particle.size) + ' nm', linewidth = 5)

    ## Make pretty
    ax.set_xlim(450, 850)
    ax.set_ylim(-0.1,1.1)
    ax.legend(ncols = 2, fontsize = 'medium')
    
    
    fig.tight_layout(pad = 1)
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(molecule) + 'nm MLAgg DF'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)


#%%

particle = particles[0]
offset = 0.015
my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-1, vmax=10)
fig, ax = plt.subplots(1, 1, figsize=[17,10])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'medium')
ax.set_ylabel('SERS CO2 - SERS N2 (au)', fontsize = 'medium')
for i, timescan in enumerate(particle.timescans):
    color = cmap(norm(i))
    spectrum = timescan.spectrum_v
    spectrum_2 = particles[1].timescans[i].spectrum_v
    ax.plot(spectrum.x, spectrum_2.y_norm - spectrum.y_norm + (i+1) * 0, color = color, zorder = 2, alpha = 1, label = str(spectrum.v) + 'V')
    # ax.text(s = str(spectrum.v) + 'V', x = 1510, y = (i-1) * offset, color = color, label = str(spectrum.v) + 'V')
# ax.text(s = '0V', x = 400, y = 2000, color = 'black')
ax.plot(particle.timescans[0].spectrum_v0.x, particles[1].timescans[0].spectrum_v0.y_norm - particles[0].timescans[0].spectrum_v0.y_norm, color = 'black', zorder = 3, linewidth = 4, linestyle = '-', label = '0V')
ax.legend(ncols =4, fontsize = 'small', loc = 'upper left')
ax.set_xlim(250, 1700)
ax.set_ylim(None, 0.15)


#%% Plot diff CA traces to stack for presentation

particle = bpdt_particles[1]
    
spectra = particle.ca_spectra
power_dict = particle.power_dict
size = particle.size
xlim = [40, 500]
df_avg = particle.df_avg

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]



fig, axes = plt.subplots(1, 2, figsize=[20,8], width_ratios = (2,1))
fig.subplots_adjust(wspace = 0.3)

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0 

# Plot Light Toggle CA on left

# for j, voltage in enumerate(voltages):
voltage = 0.4
ax = axes[0]

# ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
offset = 0.0

# if j < len(axes)-1:
#     ax.xaxis.set_ticklabels([])

i_xlim = [np.where(np.round(spectra[0].x, 1) == xlim[0])[0][0], np.where(np.round(spectra[0].x, 1) == xlim[1])[0][0]]   

for i, spectrum in enumerate(spectra):
    
    linetype = 'solid'
        
    if spectrum.voltage != voltage:
        continue

    if spectrum.wavelength > 40:# == 500 or spectrum.wavelength == 625 or spectrum.wavelength == 725 or spectrum.wavelength == 800:
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x[i_xlim[0]:i_xlim[1]], (((spectrum.y_smooth[i_xlim[0]:i_xlim[1]] - spectrum.y_smooth[np.where(np.round(spectrum.x, 1) == 50.0)]) * 10**6) + i * offset), label = spectrum.voltage, color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)

## Light on/off bars
ax.set_xlim(20, 500)
ylim = ax.get_ylim()    
ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = spectrum.toggle, color = 'grey', alpha = 0.2, zorder = 0)
ax.text(s = 'On', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 1) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
ax.text(s = 'Off', x = (spectrum.toggle) * (int(ax.get_xlim()[0]/spectrum.toggle) + 2) + (0.1 * spectrum.toggle), y = ylim[0] + .02*(ylim[1]-ylim[0]), fontsize = 'large')
ax.set_ylim(ylim)
    
    
# Plot PEC on right

# for j, voltage in enumerate(voltages):

ax = axes[1]

# ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
offset = 0.0

# if j < len(axes)-1:
#     ax.xaxis.set_ticklabels([])

wavelengths = []
ipces = []
for i, spectrum in enumerate(spectra):
    
    linetype = 'solid'
        
    if spectrum.voltage != voltage:
        continue

    
    ## Plot IPCE
    color = cmap(norm(spectrum.wavelength))  
    if spectrum.wavelength > 40:
        if np.mean(spectrum.pec) < 0:
            mfc = 'white'
        else:
            mfc = color     
        marker = 'o'                     
        ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
        ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
        ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
        wavelengths.append(spectrum.wavelength)
        ipces.append(np.mean(spectrum.ipce))
ax.plot(wavelengths, ipces, color = 'black', zorder = 0, linewidth = 2)

## Plot DF
if df_avg is not None:        
    ax2 = ax.twinx()
    ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.3, zorder = 0)
    ax2.set_ylabel('DF Intensity', fontsize = 'x-large', labelpad = 5)
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
ax.set_xlim(400, 900)
ax.yaxis.tick_right()

axes[0].set_xlabel('Time (s)', fontsize = 'x-large')
axes[0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
# fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.025, 0.15, 0.02, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Centre Wavelength (nm) - 15 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
cbar_ax.yaxis.tick_left()
cbar_ax.yaxis.set_label_position("left")
title = particle.name + ' CA Photocurrent'
fig.suptitle(title, fontsize = 'xx-large', horizontalalignment='center')


axes[1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
axes[1].set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 40)
axes[1].yaxis.set_label_position("right")
axes[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
axes[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
       
## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + title + '.svg', format = 'svg', bbox_inches='tight')
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


#%% Plot CV of MLAggs of various molecules

# all_particles = cots_particles + cotapp_particles + mtapp_particles + bpdt_particles + ito_particles
# molecules = ['CotS', 'Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT']
# sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'CotS' : 'orange'}
# my_cmap = plt.get_cmap('tab10')
# cmap = my_cmap
# voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


# for size in sizes:

#     fig, ax = plt.subplots(1, 1, figsize=[14,8])    

#     for particle in all_particles:    

#         if particle.size != size:
#             continue

#         for molecule in molecules:
#             if molecule in particle.name:
#                 this_molecule = molecule 
                
#         color = (color_dict[this_molecule]) 
    
#         spectrum = particle.cv_spectra[0]
#         power_dict = particle.power_dict
#         size = particle.size
        
        
#         # Plot CV spectra
        
#         ax.plot(spectrum.x, spectrum.y * 10**6, color = color, label = this_molecule)
    
#     particle = ito_particles[0]
#     spectrum = particle.cv_spectra[0]
#     ax.plot(spectrum.x, spectrum.y * 10**6, color = 'grey', label = 'ITO')    
            
#     ax.set_xlabel('Potential v. Ag/AgCl (V)', fontsize = 'large')
#     ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
#     # fig.subplots_adjust(right=0.9)
#     fig.suptitle(str(size) + ' nm MLAgg CV', fontsize = 'x-large', horizontalalignment='center')   
#     ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    
#     ax.legend()
    
# ## Save
# if save == True:
#     save_dir = get_directory()
#     fig.savefig(save_dir + str(size) + 'nm MLAgg CV'  + '.svg', format = 'svg', bbox_inches='tight')
#     plt.close(fig)


