# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter to compare IPCE spectra of different MLAggs from diff molecules and NP sizes

Need to import data from previous PEC processing scripts

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


skip_ca_particles = ['BPDT 70 nm MLAgg', 'TPDT 20 nm MLAgg']
skip_ocp_particles = ['BPDT 20 nm MLAgg', 'BPDT 70 nm MLAgg', 'TPDT 20 nm MLAgg', 'Bare FTO']


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
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-11-15_Compiled PEC Plots\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%%

# Calculate Au Absorption

# au = np.genfromtxt(r"C:\Users\il322\Desktop\Offline Data\Yakubovsky-117nm.txt", skip_header = 1)
# wln = au[:,0] * 1E-6 ## wln in m
# n = au[:,1]
# k = au[:,2]
# nk = n+(1j*k)

# ## Abs from alpha
# alpha = (4 * np.pi * k)/(wln) # alpha in 1/m
# A = (1 - np.exp(-1 * alpha * 60E-9)) * 100 # Abs = 100% * [1 - e^(-alpha * d)]; d = 60 nm

# ## Reflectance
# R = np.abs((1 - nk)/(1 + nk))**2 # R at top interface
# R2 = np.abs((1 - nk)/(1 + nk))**2 # R at bottom interface
# thick = particle.size * 1e-9
# T_1 = (1 - R) # I after 1st air-Au interface
# T_2 = (1 - R) * e**(-alpha * thick) # I after Au thin film of thickness = thick 
# T_3 = (1 - R2) * T_2 # I after 2nd Au-air interface
# A = T_1 - T_3 # Absorption in thin film approximation
# wln = wln *1E9
# Au = SERS.SERS_Spectrum(wln, A)
# ''' Note that A doesn't take into account absorption from cascading back reflections at Au interfaces'''


#%%

''' Comparing AuNP sizes of same molecule at different voltages'''


#%% Co-TAPP-SMe DF & IPCE v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

for voltage in voltages:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title('Co-TAPP-SMe MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    my_cmap = plt.get_cmap('tab10')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=90) 
    
    
    for i, particle in enumerate(cotapp_particles):
        
        size = particle.size
        color = cmap(color_dict[size])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
        
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ## Control samples
    for i, particle in enumerate(control_particles):
        
        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + 'Co-TAPP-SMe Photocurrent v AuNP size ' + str(voltage) + 'V'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        

#%% BPDT DF & IPCE v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

for voltage in voltages:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title('BPDT MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    my_cmap = plt.get_cmap('tab10')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=90) 
    
    
    for i, particle in enumerate(bpdt_particles):
        
        if particle.name in skip_ca_particles:
            continue
        
        size = particle.size
        color = cmap(color_dict[size])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
        
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ## Control samples
    for i, particle in enumerate(control_particles):
        
        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + 'BPDT Photocurrent v AuNP size ' + str(voltage) + 'V'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        

#%% TPDT DF & IPCE v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

for voltage in voltages:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title('TPDT MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    my_cmap = plt.get_cmap('tab10')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=90) 
    
    
    for i, particle in enumerate(tpdt_particles):
        
        if particle.name in skip_ca_particles:
            continue
        
        size = particle.size
        color = cmap(color_dict[size])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
        
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ## Control samples
    for i, particle in enumerate(control_particles):
        
        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + 'TPDT Photocurrent v AuNP size ' + str(voltage) + 'V'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)        
        
 
#%%

''' Comparing AuNP sizes of same molecule at OCP'''


#%% Co-TAPP-SMe DF & OCP Photovoltage v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}


fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax2 = ax.twinx()
ax.set_title('Co-TAPP-SMe MLAgg Photovoltage OCP', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=0, vmax=90) 


for i, particle in enumerate(cotapp_particles):
    
    if particle.name in skip_ocp_particles:
        continue
    
    size = particle.size
    color = cmap(color_dict[size])  
    
    ## Plot DF
    ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
    
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:
        
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            
## Control samples
for i, particle in enumerate(control_particles):
    
    # size = particle.size
    # color = cmap(color_dict[size])  
    
    if particle.name == '100 nm Au mirror':
        color = 'darkgoldenrod'
    else:
        color = 'grey'
    ax.plot(0, 0, color = color, label = particle.name)
    
    if particle.name in skip_ocp_particles:
        continue
           
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:

        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

## Make pretty
ax.set_xlim(400, 900)
ax.legend( loc = 'upper right')
scatt_plus = ax.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
scatt_minus = ax.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
df_line = ax2.plot(0, 0, color = 'black')
fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage +', 'PhotoVoltage -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))

fig.tight_layout()

## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'Co-TAPP-SMe Photovoltage v AuNP size OCP'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)
        

#%% BPDT DF & OCP Photovoltage v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}


fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax2 = ax.twinx()
ax.set_title('BPDT MLAgg Photovoltage OCP', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=0, vmax=90) 


for i, particle in enumerate(bpdt_particles):
    
    if particle.name in skip_ocp_particles:
        continue
    
    size = particle.size
    color = cmap(color_dict[size])  
    
    ## Plot DF
    ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
    
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:
        
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            
## Control samples
for i, particle in enumerate(control_particles):
    
    # size = particle.size
    # color = cmap(color_dict[size])  
    
    if particle.name in skip_ocp_particles:
        continue
    
    if particle.name == '100 nm Au mirror':
        color = 'darkgoldenrod'
    else:
        color = 'grey'
    ax.plot(0, 0, color = color, label = particle.name)
           
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:

        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

## Make pretty
ax.set_xlim(400, 900)
ax.legend( loc = 'upper right')
scatt_plus = ax.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
scatt_minus = ax.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
df_line = ax2.plot(0, 0, color = 'black')
fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage +', 'PhotoVoltage -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))

fig.tight_layout()

## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'BPDT Photovoltage v AuNP size OCP'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)
    
    
#%% TPDT DF & OCP Photovoltage v. Au NP size

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}


fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax2 = ax.twinx()
ax.set_title('TPDT MLAgg Photovoltage OCP', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=0, vmax=90) 


for i, particle in enumerate(tpdt_particles):
    
    if particle.name in skip_ocp_particles:
        continue
    
    size = particle.size
    color = cmap(color_dict[size])  
    
    ## Plot DF
    ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = str(size) + ' nm MLAgg')
    
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:
        
        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            
## Control samples
for i, particle in enumerate(control_particles):
    
    # size = particle.size
    # color = cmap(color_dict[size])  
    
    if particle.name == '100 nm Au mirror':
        color = 'darkgoldenrod'
    else:
        color = 'grey'
    ax.plot(0, 0, color = color, label = particle.name)
           
    ## Plot IPCE
    for spectrum in particle.ocp_spectra_voltage:

        if spectrum.wavelength > 40:
            if np.mean(spectrum.pec) < 0:
                mfc = 'white'
            else:
                mfc = color     
            marker = 'o'                     
            ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
            pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
            ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

## Make pretty
ax.set_xlim(400, 900)
ax.legend( loc = 'upper right')
scatt_plus = ax.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
scatt_minus = ax.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
df_line = ax2.plot(0, 0, color = 'black')
fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage +', 'PhotoVoltage -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))

fig.tight_layout()

## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'TPDT Photovoltage v AuNP size OCP'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)    
    

#%% DF & IPCE v. Molecule for each AuNP size and voltage

''' Compare different molecules at same AuNP size at different voltages'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


for size in sizes:
    
    for voltage in voltages:
    
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(size) + ' nm MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
       
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if particle.size != size:
                continue

            for molecule in molecules:
                if molecule in particle.name:
                    this_molecule = molecule            
            color = (color_dict[this_molecule])  
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = this_molecule)
            
            ## Plot IPCE
            for spectrum in particle.ca_spectra:
                
                if spectrum.voltage != voltage:
                    continue
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    
        # Control samples
        for i, particle in enumerate(control_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            # size = particle.size
            # color = cmap(color_dict[size])  
            
            if particle.name == '100 nm Au mirror':
                color = 'darkgoldenrod'
            else:
                color = 'grey'
            ax.plot(0, 0, color = color, label = particle.name)
                   
            ## Plot IPCE
            for spectrum in particle.ca_spectra:
                
                if spectrum.voltage != voltage:
                    continue
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black')
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(size) + ' nm MLAgg Photocurrent v Molecule ' + str(voltage) + 'V'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)


#%% DF & OCP photovoltage v. Molecule for each AuNP size

''' Compare different molecules at same AuNP size at OCP'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


for size in sizes:
    
    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(size) + ' nm MLAgg Photovoltage OCP', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
   
    
    for i, particle in enumerate(all_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule            
        color = (color_dict[this_molecule])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = this_molecule)
        
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ## Control samples
    for i, particle in enumerate(control_particles):
        
        if particle.name in skip_ocp_particles:
            continue

        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        for spectrum in particle.ca_spectra:
            
            if spectrum.voltage != voltage:
                continue
            
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)

    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage +', 'PhotoVoltage -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + ' nm MLAgg Photovoltage v Molecule OCP'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig) 


#%% Photocurrent v. voltage for each molecule and Au NP size

''' Plot photocurrent of each voltage for same molecule & size'''

all_particles = cotapp_particles + control_particles + bpdt_particles + tpdt_particles + mtapp_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('viridis')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5) 
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


for size in sizes:
    
    for molecule in molecules:  
        
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(molecule) + ' ' + str(size) + ' nm MLAgg Photocurrent ', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
       
        
        for voltage in voltages:
            
            color = cmap(norm(voltage)) 

            ax2.plot(0, 0, color = color, label = str(voltage) + ' V')

        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if particle.size != size:
                continue

            if molecule not in particle.name:
                continue
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = 'black', alpha = 0.5, zorder = 0)
            
            ## Plot IPCE
            for spectrum in particle.ca_spectra:
                
                color = cmap(norm(spectrum.voltage)) 
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    
        # ## Control samples
        # for i, particle in enumerate(control_particles):
            
            # if particle.name in skip_ca_particles:
                # continue
          
        #     # size = particle.size
        #     # color = cmap(color_dict[size])  
            
        #     if particle.name == '100 nm Au mirror':
        #         color = 'darkgoldenrod'
        #     else:
        #         color = 'grey'
        #     ax.plot(0, 0, color = color, label = particle.name)
                   
        #     ## Plot IPCE
        #     for spectrum in particle.ca_spectra:
                
        #         if spectrum.voltage != voltage:
        #             continue
                
        #         if spectrum.wavelength > 40:
        #             if np.mean(spectrum.pec) < 0:
        #                 mfc = 'white'
        #             else:
        #                 mfc = color     
        #             marker = 'o'                     
        #             ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
        #             ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
        #             ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black')
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(molecule) + ' ' + str(size) + ' nm Photocurrent v Voltage'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
            

#%%

''' Good after here'''


#%% Just plot Co-TAPP-SMe DF for diff Au sizes

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.set_title('Co-TAPP-SMe MLAgg Dakrfield ', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Normalized Scattering', fontsize = 'x-large')
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=0, vmax=90) 


for i, particle in enumerate(cotapp_particles):
    
    size = particle.size
    color = cmap(color_dict[size])  
    
    ## Plot DF
    particle.df_avg.y_norm = (particle.df_avg.y - particle.df_avg.y.min())/(particle.df_avg.y.max() - particle.df_avg.y.min())
    particle.df_avg.y_smooth_norm = spt.butter_lowpass_filt_filt(particle.df_avg.y_norm, cutoff = 1000, fs = 40000)
    ax.plot(particle.df_avg.x, particle.df_avg.y_smooth_norm, color = color, alpha = 1, zorder = 0, label = str(size) + ' nm MLAgg', linewidth = 6)
    
## Make pretty
ax.set_xlim(500, 900)
ax.set_ylim(0,1)
ax.legend( loc = 'upper right')


fig.tight_layout()

## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'Co-TAPP-SMe MLAgg DF'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Plot DF for diff molecules all same AuNP size

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70, 80]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap

for size in sizes:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax.set_title(str(size) + 'nm MLAgg Dakrfield ', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax.set_ylabel('Normalized Scattering Intensity', fontsize = 'x-large')

    for particle in all_particles:    

        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule 
                
        color = (color_dict[this_molecule])         
        
        ## Plot DF
        particle.df_avg.y_norm = (particle.df_avg.y - particle.df_avg.y.min())/(particle.df_avg.y.max() - particle.df_avg.y.min())
        particle.df_avg.y_smooth_norm = spt.butter_lowpass_filt_filt(particle.df_avg.y_norm, cutoff = 1000, fs = 40000)
        ax.plot(particle.df_avg.x, particle.df_avg.y_norm, color = color, alpha = 1, zorder = 0, label = this_molecule, linewidth = 4)

    ## Make pretty
    ax.set_xlim(450, 850)
    ax.set_ylim(-0.1,1.1)
    ax.legend( loc = 'upper right')
    
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + 'nm MLAgg DF'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)


#%% Plot UV-Vis of M-TAPP-SMe molecules

my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2021-04-05 All MTPP UV-Vis PL.h5")
my_csv = r"C:\Users\il322\Desktop\Offline Data\Co-TAPP Abs_processed.csv"

Co = np.genfromtxt(my_csv, delimiter=',')
Co = Co.transpose()
Co = spt.Spectrum(Co[0], Co[1])

H2 = my_h5['H2-MTPP MeOH MeCN UV-Vis']
Ni = my_h5['Ni-MTPP MeOH MeCN UV-Vis']
Zn = my_h5['Zn-MTPP MeOH MeCN UV-Vis']

H2 = spt.Spectrum(x = H2.attrs['wavelengths'], y = H2.attrs['yRaw'])
Ni = spt.Spectrum(x = Ni.attrs['wavelengths'], y = Ni.attrs['yRaw'])
Zn = spt.Spectrum(x = Zn.attrs['wavelengths'], y = Zn.attrs['yRaw'])

color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
plot_start = 200
fig, ax = plt.subplots(1,1, figsize = (14,10))
my_cmap = plt.get_cmap('Set1')
ax.plot(H2.x[300:], H2.y[300:], label = 'H2-TAPP', color = 'purple', zorder = 4, linewidth = 5)
ax.plot(Co.x[:400], Co.y[:400], label = 'Co-TAPP', color = 'blue', zorder = 2, linewidth = 5)
ax.plot(Ni.x[300:], Ni.y[300:], label = 'Ni-TAPP', color = 'red', zorder = 3, linewidth = 5)
ax.plot(Zn.x[300:], Zn.y[300:], label = 'Zn-TAPP', color = 'green', zorder = 1, linewidth = 5)

ax.plot(H2.x[:], H2.y[:], linestyle = 'dashed', color = 'purple', zorder = 1, linewidth = 5)
ax.plot(Co.x[:], Co.y[:], linestyle = 'dashed', color = 'blue', zorder = 1, linewidth = 5)
ax.plot(Ni.x[:], Ni.y[:], linestyle = 'dashed', color = 'red', zorder = 1, linewidth = 5)
ax.plot(Zn.x[:], Zn.y[:], linestyle = 'dashed', color = 'green', zorder = 1, linewidth = 5)

#ax.text(s = '633nm Raman\n   Excitation', x = 645, y = 0.12, fontsize = 'small', color = 'red')
# ax.text(s = '785nm Raman\n   Excitation', x = 745, y = 0.01, fontsize = 'small', color = 'darkorange')

# ax.vlines(x = 632.8, ymin = 0, ymax = 0.25, color = 'red', linewidth = 10, zorder = 0)
# ax.vlines(x = 785, ymin = -1, ymax = 0.25, color = 'darkorange', linewidth = 10, zorder = 0)

ax.set_xlim(450,850)
ax.set_ylim(-0.004,0.25)
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Absorption', fontsize = 'x-large')
ax.tick_params(axis='both', which='major', labelsize=25)
ax.legend(fontsize = 'x-large')
fig.suptitle('UV-Vis: 1$\mu$M M-TAPP in 1:1 MeOH:MeCN', fontsize = 'xx-large')

fig.tight_layout()

## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'UV-Vis'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Plot DF x UV-Vis

all_particles = cotapp_particles + mtapp_particles #+ bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70, 80]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
molecule_uv_dict = {'Co-TAPP-SMe' : Co, 'Ni-TAPP-SMe' : Ni, 'Zn-TAPP-SMe' : Zn, 'H2-TAPP-SMe' : H2}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap

for size in sizes:

    fig, ax = plt.subplots(1, 1, figsize=[12,9])
    ax.set_title(str(size) + 'nm MLAgg Dakrfield x UV-Vis', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax.set_ylabel('Norm DF x UV-Vis (a.u.)', fontsize = 'x-large')
    ax2 = ax.twinx()
    ax2.set_ylabel('Normalized DF Intensity', fontsize = 'x-large', rotation = 270, labelpad = 30)

    for particle in all_particles:    

        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule 
                
        color = (color_dict[this_molecule])         
        
        ## Get DF
        particle.df_avg.y_norm = (particle.df_avg.y - particle.df_avg.y.min())/(particle.df_avg.y.max() - particle.df_avg.y.min())
        particle.df_avg.y_smooth_norm = spt.butter_lowpass_filt_filt(particle.df_avg.y_norm, cutoff = 1000, fs = 40000)
        df = deepcopy(particle.df_avg)
        ax2.plot(df.x, df.y_norm, color = color, alpha = 0.2, zorder = 0, linewidth = 4)

        ## Get UV-Vis, interp to DF
        uv = molecule_uv_dict[this_molecule]
        uv_func = scipy.interpolate.interp1d(uv.x, uv.y)
        uv_interp = uv_func(df.x)
        ax2.plot(df.x, uv_interp, color = color, alpha = 0.2, linestyle = 'dashed')
        
        ## Plot UV-Vis x DF
        product = df.y_norm * uv_interp
        ax.plot(df.x, product, color = color, alpha = 1, label = this_molecule)
        

    ## Make pretty
    ax.set_xlim(450, 850)
    ax.set_ylim(-0.004,0.25)
    ax2.set_ylim(-0.1, 1.1)
    ax.legend( loc = 'upper right')
    df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
    uv_line = plt.plot([0], [0], color = 'black', linestyle = 'dashed', alpha = 0.2)
    product_line = plt.plot([0], [0], color = 'black', alpha = 1)
    fig.legend(handles = [product_line[0], df_line[0], uv_line[0]], labels = ['UV x DF', 'DF', 'UV'], ncols = 3, bbox_to_anchor=(0.68, 0.93))

    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + 'nm MLAgg DF x UV-Vis'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)


#%% Plot CV of MLAggs of various molecules

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


for size in sizes:

    fig, ax = plt.subplots(1, 1, figsize=[14,8])    

    for particle in all_particles:    

        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule 
                
        color = (color_dict[this_molecule]) 
    
        spectrum = particle.cv_spectra[0]
        power_dict = particle.power_dict
        size = particle.size
        
        
        # Plot CV spectra
        

        ax.plot(spectrum.x, spectrum.y * 10**6, color = color, label = this_molecule)
            
            
    ax.set_xlabel('Potential v. Ag/AgCl (V)', fontsize = 'large')
    ax.set_ylabel('Current ($\mu$A)', fontsize = 'large')
    # fig.subplots_adjust(right=0.9)
    fig.suptitle(str(size) + ' nm MLAgg CV', fontsize = 'x-large', horizontalalignment='center')   
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2,2), useOffset=False)
    
    ax.legend()
    
## Save
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + str(size) + 'nm MLAgg CV'  + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Plot IPCE v. molecule for each wavelength

# all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
# molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT', 'Au mirror', 'FTO']
# sizes = [20, 35, 57, 70, 80]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO': 'grey'}
# my_cmap = plt.get_cmap('tab10')
# cmap = my_cmap
# voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# wavelengths = []
# for spectrum in cotapp_particles[0].ca_spectra:
#     if spectrum.wavelength not in wavelengths:
#         wavelengths.append(spectrum.wavelength)
        

# for size in sizes:
    
#     for voltage in voltages:
    
#         for wavelength in wavelengths:    
    
#             fig, ax = plt.subplots(1, 1, figsize=[14,10])
#             ax.set_title(str(size) + ' nm MLAgg Photocurrent ' + str(voltage) + ' V @ ' + str(wavelength) + ' nm', fontsize = 'x-large')
#             ax.set_xlabel('MLAgg Molecule/Sample', fontsize = 'x-large')
#             ax.set_ylabel('IPCE @ ' + str(wavelength) + 'nm (%)', fontsize = 'x-large')
            
#             for i, particle in enumerate(all_particles):
                
#                 if particle.name in skip_ca_particles:
#                     continue
                
#                 if particle.size != size and particle not in control_particles:
#                     continue
    
#                 for j, molecule in enumerate(molecules):
#                     if molecule in particle.name:
#                         this_molecule = molecule            
#                 color = (color_dict[this_molecule])  
                
#                 ## Plot DF
#                 # ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = this_molecule)
                
#                 ## Plot IPCE
#                 for spectrum in particle.ca_spectra:
                    
#                     if spectrum.voltage != voltage:
#                         continue
                    
#                     if spectrum.wavelength != wavelength:
#                         continue
#                     if np.mean(spectrum.pec) < 0:
#                         mfc = 'white'
#                     else:
#                         mfc = color     
#                     marker = 'o'                     
#                     ax.scatter(str(this_molecule), np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
#                     ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
#                     ax.errorbar(this_molecule,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
               
#             ## Make pretty
#             # ax.set_xlim(400, 900)
#             # ax.legend( loc = 'upper right')
#             xlim = ax.get_xlim()
#             ylim = ax.get_ylim()
#             scatt_plus = plt.scatter(-1, -1, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
#             scatt_minus = plt.scatter(-1, -1, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
#             fig.legend(handles = [scatt_plus, scatt_minus], labels = ['IPCE +', 'IPCE -'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
            
#             fig.tight_layout()
            
#             ## Save
#             if save == True:
#                 save_dir = get_directory()
#                 fig.savefig(save_dir + str(size) + ' nm MLAgg Photocurrent v Molecule ' + str(voltage) + 'V ' + str(wavelength) + 'nm'  + '.svg', format = 'svg', bbox_inches='tight')
#                 plt.close(fig)


#%% Plot NORMALIZED IPCE v. molecule for each wavelength

# all_particles = cotapp_particles + mtapp_particles + bpdt_particles #+ control_particles
# molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT', 'Au mirror', 'FTO']
# sizes = [20, 35, 57, 70, 80]
# # sizes = [20]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO': 'grey'}
# my_cmap = plt.get_cmap('tab10')
# cmap = my_cmap
# voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# # voltages = [-0.4]
# wavelengths = []
# for spectrum in cotapp_particles[0].ca_spectra:
#     if spectrum.wavelength not in wavelengths:
#         wavelengths.append(spectrum.wavelength)
# # wavelengths = [500]

# for size in sizes:
    
#     for voltage in voltages:
    
#         for wavelength in wavelengths:    
    
#             fig, ax = plt.subplots(1, 1, figsize=[14,10])
#             ax.set_title(str(size) + ' nm MLAgg Normalized Photocurrent ' + str(voltage) + ' V @ ' + str(wavelength) + ' nm', fontsize = 'x-large')
#             ax.set_xlabel('MLAgg Molecule/Sample', fontsize = 'x-large')
#             ax.set_ylabel('Normalized IPCE @ ' + str(wavelength) + 'nm (a.u.)', fontsize = 'x-large')
            
#             for i, particle in enumerate(all_particles):
                
#                 if particle.name in skip_ca_particles:
#                     continue
                
#                 if particle.size != size and particle not in control_particles:
#                     continue
    
#                 for j, molecule in enumerate(molecules):
#                     if molecule in particle.name:
#                         this_molecule = molecule            
#                 color = (color_dict[this_molecule])  
                
#                 ## Plot DF
#                 # ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.5, zorder = 0, label = this_molecule)
                
#                 ## Normalize IPCE to area under curve (500 nm - 850 nm)
#                 all_wavelengths = []
#                 ipces = []
#                 for spectrum in particle.ca_spectra:
#                     if spectrum.voltage != voltage:
#                         continue
#                     if spectrum.wavelength >= 500:
#                         all_wavelengths.append(spectrum.wavelength)
#                         ipces.append(np.mean(spectrum.ipce))                    
#                 area = np.trapz(y = ipces, x = all_wavelengths)
#                 # ax2.text(x = 700, y = np.max(ipces), s = str(area), color = color)
                     
#                 ## Plot normalized IPCE
#                 for spectrum in particle.ca_spectra:
#                     if spectrum.voltage != voltage:
#                         continue
                    
#                     if spectrum.wavelength != wavelength:
#                         continue
#                     if np.mean(spectrum.pec) < 0:
#                         mfc = 'white'
#                     else:
#                         mfc = color     
#                     marker = 'o'                     
#                     ax.scatter(str(this_molecule), np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
#                     ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
#                     ax.errorbar(this_molecule,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
               
#             ## Make pretty
#             # ax.set_xlim(400, 900)
#             # ax.legend( loc = 'upper right')
#             xlim = ax.get_xlim()
#             ylim = ax.get_ylim()
#             scatt_plus = plt.scatter(-1, -1, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
#             scatt_minus = plt.scatter(-1, -1, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
#             fig.legend(handles = [scatt_plus, scatt_minus], labels = ['IPCE +', 'IPCE -'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
            
#             fig.tight_layout()
            
#             ## Save
#             if save == True:
#                 save_dir = get_directory()
#                 fig.savefig(save_dir + str(size) + ' nm MLAgg Normalized Photocurrent v Molecule ' + str(voltage) + 'V ' + str(wavelength) + 'nm'  + '.svg', format = 'svg', bbox_inches='tight')
#                 plt.close(fig)



#%% DF & IPCE v. Molecule for each AuNP size and voltage v2

''' Updated cosmetics'''

''' Compare different molecules at same AuNP size at different voltages'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT', 'Au mirror', 'FTO']
sizes = [20, 35, 57, 70]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# voltages = [-0.4]

for size in sizes:
    
    for voltage in voltages:
    
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(size) + ' nm MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
       
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if particle.size != size and particle not in control_particles:
                continue

            for molecule in molecules:
                if molecule in particle.name:
                    this_molecule = molecule            
            color = (color_dict[this_molecule])  
            
            ## Plot DF
            try:
                ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.3, zorder = 0, label = this_molecule, linewidth = 2)
            except:
                ax.plot(0, 0, color = color, label = particle.name)
            
            ## Plot IPCE
            wavelengths = []
            ipces = []
            for spectrum in particle.ca_spectra:
                
                if spectrum.voltage != voltage:
                    continue
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    wavelengths.append(spectrum.wavelength)
                    ipces.append(np.mean(spectrum.ipce))
            ax2.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = this_molecule)
    
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(size) + ' nm MLAgg Photocurrent v Molecule ' + str(voltage) + 'V v2'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)


#%% DF & IPCE v. Molecule for each AuNP size and voltage v2 NORMALIZED

''' Updated cosmetics & NORMALIZED to IPCE area under curve area'''

''' Compare different molecules at same AuNP size at different voltages'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles #+ control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT', 'Au mirror', 'FTO']
sizes = [20, 35, 57, 70]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# voltages = [-0.4]

for size in sizes:
    
    for voltage in voltages:
    
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(size) + ' nm MLAgg Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('Normalized IPCE (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if particle.size != size and particle not in control_particles:
                continue

            for molecule in molecules:
                if molecule in particle.name:
                    this_molecule = molecule            
            color = (color_dict[this_molecule])  
            
            ## Plot DF
            try:
                ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.3, zorder = 0, label = this_molecule, linewidth = 2)
            except:
                ax.plot(0, 0, color = color, label = particle.name)
            
            ## Normalize IPCE to area under curve (500 nm - 850 nm)
            wavelengths = []
            ipces = []
            for spectrum in particle.ca_spectra:
                if spectrum.voltage != voltage:
                    continue
                if spectrum.wavelength >= 500:
                    wavelengths.append(spectrum.wavelength)
                    ipces.append(np.mean(spectrum.ipce))                    
            area = np.trapz(y = ipces, x = wavelengths)
            # ax2.text(x = 700, y = np.max(ipces), s = str(area), color = color)
                 
            ## Plot normalized IPCE
            wavelengths = []
            norm_ipces = []
            for spectrum in particle.ca_spectra:
                if spectrum.voltage != voltage:
                    continue
                if spectrum.wavelength >= 500:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    wavelengths.append(spectrum.wavelength)
                    norm_ipces.append(np.mean(spectrum.ipce)/area)
            area = np.trapz(y = norm_ipces, x = wavelengths)
            ax2.plot(wavelengths, norm_ipces, color = color, zorder = 0, linewidth = 2, label = this_molecule)
        
      
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(size) + ' nm MLAgg Photocurrent v Molecule ' + str(voltage) + 'V v2 Norm'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
                    

#%% DF & OCP photovoltage v. Molecule for each AuNP size v2
#%% DF & OCP v. Molecule at each AuNP size v2

''' Compare different molecules at same AuNP size at OCP Updated cosmetics'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap


for size in sizes:
    
    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(size) + ' nm MLAgg Photovoltage OCP', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
   
    
    for i, particle in enumerate(all_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule            
        color = (color_dict[this_molecule])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.2, zorder = 0, label = this_molecule)
        
        ## Plot IPCE
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength > 40:   
                mfc = color  
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3/area)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = this_molecule)

    ## Control samples
    for i, particle in enumerate(control_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength > 40:
                mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3/area)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = particle.name)
        
    ## Make pretty
    ax.set_xlim(400, 900)
    ax2.hlines(y = 0, xmin = 400, xmax = 900, color = 'black', alpha = 1, linewidth = 2, zorder = 0)
    ax2.legend( loc = 'lower right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
    fig.legend(handles = [scatt_plus, df_line[0]], labels = ['PhotoVoltage', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.2))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + ' nm MLAgg Photovoltage v Molecule OCP v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig) 


#%% DF & OCP photovoltage v. Molecule for each AuNP size v2 NORMALIZED

''' Compare different molecules at same AuNP size at OCP Updated cosmetics NORMALIZED'''


all_particles = cotapp_particles + mtapp_particles + bpdt_particles + control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap


for size in sizes:
    
    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(size) + ' nm MLAgg Photovoltage OCP', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('Normalized Photovoltage (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
   
    
    for i, particle in enumerate(all_particles):
        
        if particle.name in skip_ocp_particles:
            continue

        if particle.size != size:
            continue

        for molecule in molecules:
            if molecule in particle.name:
                this_molecule = molecule            
        color = (color_dict[this_molecule])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.2, zorder = 0, label = this_molecule)
        
        ## Normalize PV to area under curve (500 nm - 850 nm)
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength >= 500:
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3)                    
        area = np.abs(np.trapz(y = pvs, x = wavelengths))
        # ax2.text(x = 700, y = np.max(pvs), s = str(area), color = color)
        
        ## Plot PV
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength >= 500:
                mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3/area, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3/area
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3/area, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3/area)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = this_molecule)

    # ## Control samples
    # for i, particle in enumerate(control_particles):
        
    #     if particle.name in skip_ocp_particles:
    #         continue
        
    #     # size = particle.size
    #     # color = cmap(color_dict[size])  
        
    #     if particle.name == '100 nm Au mirror':
    #         color = 'darkgoldenrod'
    #     else:
    #         color = 'grey'
    #     ax.plot(0, 0, color = color, label = particle.name)
               
    #     ## Plot IPCE
    #     wavelengths = []
    #     pvs = []
    #     for spectrum in particle.ocp_spectra_voltage:
    #         if spectrum.wavelength > 40:
    #             mfc = color     
    #             marker = 'o'                     
    #             ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
    #             pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
    #             ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    #             wavelengths.append(spectrum.wavelength)
    #             pvs.append((np.mean(spectrum.pec)) * 10**3/area)
    #     ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = particle.name)
        
    ## Make pretty
    ax.set_xlim(400, 900)
    ax2.hlines(y = 0, xmin = 400, xmax = 900, color = 'black', alpha = 1, linewidth = 2, zorder = 0)
    ax2.legend( loc = 'lower right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
    fig.legend(handles = [scatt_plus, df_line[0]], labels = ['PhotoVoltage', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.2))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(size) + ' nm MLAgg Normalized Photovoltage v Molecule OCP v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig) 
        

#%% DF & IPCE v. Au NP size for each molecule v2

''' IPCE v. AuNP size for each molecule, updated cosmetics'''

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles #+ control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# voltages = [-0.4]

for molecule in molecules:

    for voltage in voltages:

        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(molecule) + ' Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
        my_cmap = plt.get_cmap('tab10')
        cmap = my_cmap
        norm = mpl.colors.Normalize(vmin=0, vmax=90) 
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if molecule not in particle.name:
                continue
            
            size = particle.size
            color = cmap(color_dict[size])  
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.3, zorder = 0, label = this_molecule, linewidth = 2)
            
            ## Plot IPCE
            wavelengths = []
            ipces = []
            for spectrum in particle.ca_spectra:
                
                if spectrum.voltage != voltage:
                    continue
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    wavelengths.append(spectrum.wavelength)
                    ipces.append(np.mean(spectrum.ipce))
            ax2.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = str(size) + ' nm MLAgg')
    
        ## Control samples
        for i, particle in enumerate(control_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            # size = particle.size
            # color = cmap(color_dict[size])  
            
            if particle.name == '100 nm Au mirror':
                color = 'darkgoldenrod'
            else:
                color = 'grey'
            ax.plot(0, 0, color = color, label = particle.name)
                   
            ## Plot IPCE
            wavelengths = []
            ipces = []
            for spectrum in particle.ca_spectra:
                
                if spectrum.voltage != voltage:
                    continue
                
                if spectrum.wavelength > 40:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    wavelengths.append(spectrum.wavelength)
                    ipces.append(np.mean(spectrum.ipce))
            ax2.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = particle.name)
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(molecule) +' Photocurrent v AuNP size ' + str(voltage) + 'V v2'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
            
            
#%% DF & IPCE v. Au NP size for each molecule v2 NORMALIZED

''' IPCE v. AuNP size for each molecule, updated cosmetics NORMALIZED'''

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles #+ control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
# voltages = [-0.4]

for molecule in molecules:

    for voltage in voltages:

        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(molecule) + ' Photocurrent ' + str(voltage) + ' V', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('Normalized IPCE (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
        my_cmap = plt.get_cmap('tab10')
        cmap = my_cmap
        norm = mpl.colors.Normalize(vmin=0, vmax=90) 
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if molecule not in particle.name:
                continue
            
            size = particle.size
            color = cmap(color_dict[size])  
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.3, zorder = 0, label = this_molecule, linewidth = 2)
            
            ## Normalize IPCE to area under curve (500 nm - 850 nm)
            wavelengths = []
            ipces = []
            for spectrum in particle.ca_spectra:
                if spectrum.voltage != voltage:
                    continue
                if spectrum.wavelength >= 500:
                    wavelengths.append(spectrum.wavelength)
                    ipces.append(np.mean(spectrum.ipce))                    
            area = np.trapz(y = ipces, x = wavelengths)
            # ax2.text(x = 700, y = np.max(ipces), s = str(area), color = color)
                 
            ## Plot normalized IPCE
            wavelengths = []
            norm_ipces = []
            for spectrum in particle.ca_spectra:
                if spectrum.voltage != voltage:
                    continue
                if spectrum.wavelength >= 500:
                    if np.mean(spectrum.pec) < 0:
                        mfc = 'white'
                    else:
                        mfc = color     
                    marker = 'o'                     
                    ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                    ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
                    ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                    wavelengths.append(spectrum.wavelength)
                    norm_ipces.append(np.mean(spectrum.ipce)/area)
            area = np.trapz(y = norm_ipces, x = wavelengths)
            ax2.plot(wavelengths, norm_ipces, color = color, zorder = 0, linewidth = 2, label = str(size) + ' nm MLAgg')
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(molecule) +' Normalized Photocurrent v AuNP size ' + str(voltage) + 'V v2'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
            
            
#%% DF & OCP Photovoltage v. Au NP size for each molecule v2

'''OCP Photovoltage v. AuNP size updated cosmetics'''

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles #+ control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap

for molecule in molecules:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(molecule) + ' MLAgg Photovoltage OCP', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('Photovoltage (mV/mW)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    my_cmap = plt.get_cmap('tab10')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=90) 
    
    
    for i, particle in enumerate(all_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        if molecule not in particle.name:
            continue
        
        size = particle.size
        color = cmap(color_dict[size])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.2, zorder = 0, label = str(size) + ' nm MLAgg')
        
        ## Plot IPCE
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength > 40:
                mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = str(size) + ' nm MLAgg')
                
    ## Control samples
    for i, particle in enumerate(control_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        # size = particle.size
        # color = cmap(color_dict[size])  
        
        if particle.name == '100 nm Au mirror':
            color = 'darkgoldenrod'
        else:
            color = 'grey'
        ax.plot(0, 0, color = color, label = particle.name)
               
        ## Plot IPCE
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength > 40:
                mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = particle.name)
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax2.legend( loc = 'lower right')
    ax2.hlines(y = 0, xmin = 400, xmax = 900, color = 'black', alpha = 1, linewidth = 2, zorder = 0)
    scatt_plus = ax.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    # scatt_minus = ax.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = ax2.plot(0, 0, color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.2))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(molecule) + ' Photovoltage v AuNP size OCP v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)

#%% DF & OCP Normalized Photovoltage v. Au NP size for each molecule v2

'''OCP Photovoltage v. AuNP size updated cosmetics Normalized'''

color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}

all_particles = cotapp_particles + mtapp_particles + bpdt_particles + tpdt_particles #+ control_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
color_dict = {20 : 0, 35 : 2, 57 : 1, 70 : 3}
my_cmap = plt.get_cmap('tab10')
cmap = my_cmap

for molecule in molecules:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(molecule) + ' MLAgg Photovoltage OCP', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('Normalized Photovoltage (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    my_cmap = plt.get_cmap('tab10')
    cmap = my_cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=90) 
    
    
    for i, particle in enumerate(all_particles):
        
        if particle.name in skip_ocp_particles:
            continue
        
        if molecule not in particle.name:
            continue
        size = particle.size
        color = cmap(color_dict[size])  
        
        ## Plot DF
        ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color, alpha = 0.2, zorder = 0, label = str(size) + ' nm MLAgg')
        
        ## Normalize PV to area under curve (500 nm - 850 nm)
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength >= 500:
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3)                    
        area = np.abs(np.trapz(y = pvs, x = wavelengths))
        # ax2.text(x = 700, y = np.max(pvs), s = str(area), color = color)
        
        ## Plot PV
        wavelengths = []
        pvs = []
        for spectrum in particle.ocp_spectra_voltage:
            if spectrum.wavelength > 40:
                mfc = color     
                marker = 'o'                     
                ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3/area, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
                pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3/area
                ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3/area, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                pvs.append((np.mean(spectrum.pec)) * 10**3/area)
        ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = str(size) + ' nm MLAgg')
                
    # ## Control samples
    # for i, particle in enumerate(control_particles):
        
    #     if particle.name in skip_ocp_particles:
    #         continue

    #     # size = particle.size
    #     # color = cmap(color_dict[size])  
        
    #     if particle.name == '100 nm Au mirror':
    #         color = 'darkgoldenrod'
    #     else:
    #         color = 'grey'
    #     ax.plot(0, 0, color = color, label = particle.name)
               
    #     ## Plot IPCE
    #     wavelengths = []
    #     pvs = []
    #     for spectrum in particle.ocp_spectra_voltage:
    #         if spectrum.wavelength > 40:
    #             mfc = color     
    #             marker = 'o'                     
    #             ax2.scatter(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, facecolors = mfc, edgecolors = color, s = 100, marker = marker, linewidth = 3, zorder = 2)
    #             pv_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * 10**3
    #             ax2.errorbar(spectrum.wavelength, (np.mean(spectrum.pec)) * 10**3, yerr = pv_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
    #             wavelengths.append(spectrum.wavelength)
    #             pvs.append((np.mean(spectrum.pec)) * 10**3)
    #     ax2.plot(wavelengths, pvs, color = color, zorder = 0, linewidth = 2, label = particle.name)
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax2.legend( loc = 'lower right')
    ax2.hlines(y = 0, xmin = 400, xmax = 900, color = 'black', alpha = 1, linewidth = 2, zorder = 0)
    scatt_plus = ax.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    # scatt_minus = ax.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    df_line = ax2.plot(0, 0, color = 'black')
    fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['PhotoVoltage', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.2))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(molecule) + ' Normalized Photovoltage v AuNP size OCP v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)

#%% Photocurrent v. voltage for each molecule and Au NP size v2

''' Photocurrent v. voltage updated cosmetics'''

all_particles = cotapp_particles + control_particles + bpdt_particles + tpdt_particles + mtapp_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# sizes = [20]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('viridis')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5) 
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


# For MLAggs

for size in sizes:
    
    for molecule in molecules:  
        
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(molecule) + ' ' + str(size) + ' nm MLAgg Photocurrent ', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
        
        for i, particle in enumerate(all_particles):
            
            if particle.name in skip_ca_particles:
                continue
            
            if particle.size != size:
                continue

            if molecule not in particle.name:
                continue
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = 'black', alpha = 0.2, zorder = 0)
            
            ## Plot IPCE
            for voltage in voltages:    
                color = cmap(norm(voltage)) 
                wavelengths = []
                ipces = []
                for spectrum in particle.ca_spectra:
                    if spectrum.voltage != voltage:
                        continue
                    if spectrum.wavelength > 40:
                        if np.mean(spectrum.pec) < 0:
                            mfc = 'white'
                        else:
                            mfc = color     
                        marker = 'o'                     
                        ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                        ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                        ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                        wavelengths.append(spectrum.wavelength)
                        ipces.append(np.mean(spectrum.ipce))
                ax2.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V')
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(molecule) + ' ' + str(size) + ' nm Photocurrent v Voltage v2'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)


# For controls

for particle in control_particles:

    if particle.name in skip_ca_particles:
        continue
        
    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax.set_title(str(particle.name) + ' Photocurrent ', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        
    ## Plot IPCE
    for voltage in voltages:    
        color = cmap(norm(voltage)) 
        wavelengths = []
        ipces = []
        for spectrum in particle.ca_spectra:
            if spectrum.voltage != voltage:
                continue
            if spectrum.wavelength > 40:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                ipces.append(np.mean(spectrum.ipce))
        ax.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V')
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    fig.legend(handles = [scatt_plus, scatt_minus], labels = ['IPCE +', 'IPCE -'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(particle.name) + ' nm Photocurrent v Voltage v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)

#%% Photocurrent v. voltage for each molecule and Au NP size v2 NORMALIZED

''' Photocurrent v. voltage updated cosmetics NORMALIZED'''

all_particles = cotapp_particles + control_particles + bpdt_particles + tpdt_particles + mtapp_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT']
sizes = [20, 35, 57, 70]
# sizes = [20]
# color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink'}
my_cmap = plt.get_cmap('viridis')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5) 
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]


# For MLAggs

for size in sizes:
    
    for molecule in molecules:  
        
        fig, ax = plt.subplots(1, 1, figsize=[16,10])
        ax2 = ax.twinx()
        ax.set_title(str(molecule) + ' ' + str(size) + ' nm MLAgg Photocurrent ', fontsize = 'x-large')
        ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
        ax2.set_ylabel('Normalized IPCE (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
        ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
        
        for i, particle in enumerate(all_particles):
            
            # if particle.name in skip_ca_particles:
            #     continue
            
            if particle.size != size:
                continue

            if molecule not in particle.name:
                continue
            
            ## Plot DF
            ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = 'black', alpha = 0.2, zorder = 0)
            
            ## Plot normalized IPCE
            for voltage in voltages:    
                color = cmap(norm(voltage)) 
                
                ## Calculte IPCE area under curve 500 - 850nm
                wavelengths = []
                ipces = []
                for spectrum in particle.ca_spectra:
                    if spectrum.voltage != voltage:
                        continue
                    if spectrum.wavelength >= 500:                  
                        wavelengths.append(spectrum.wavelength)
                        ipces.append(np.mean(spectrum.ipce))
                area = np.abs(np.trapz(y = ipces, x = wavelengths))
                # ax2.text(s = str(area), x = 700, y = np.max(ipces), color = color)
                
                ## Plot normalized IPCE
                wavelengths = []
                norm_ipces = []
                for spectrum in particle.ca_spectra:
                    if spectrum.voltage != voltage:
                        continue
                    if spectrum.wavelength >= 475:
                        if np.mean(spectrum.pec) < 0:
                            mfc = 'white'
                        else:
                            mfc = color     
                        marker = 'o'                     
                        ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                        ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
                        ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                        wavelengths.append(spectrum.wavelength)
                        norm_ipces.append(np.mean(spectrum.ipce)/area)
                ax2.plot(wavelengths, norm_ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V')
        
        ## Make pretty
        ax.set_xlim(400, 900)
        ax2.legend( loc = 'upper right')
        scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
        df_line = plt.plot([0], [0], color = 'black', alpha = 0.2)
        fig.legend(handles = [scatt_plus, scatt_minus, df_line[0]], labels = ['IPCE +', 'IPCE -', 'Darkfield Scattering'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(molecule) + ' ' + str(size) + ' nm Normalized Photocurrent v Voltage v2'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)


# For controls

for particle in control_particles:

    if particle.name in skip_ca_particles:
        continue
        
    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax.set_title(str(particle.name) + ' Photocurrent ', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax.set_ylabel('Normalized IPCE (a.u.)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    
    ## Plot normalized IPCE
    for voltage in voltages:    
        color = cmap(norm(voltage)) 
        
        ## Calculte IPCE area under curve 500 - 850nm
        wavelengths = []
        ipces = []
        for spectrum in particle.ca_spectra:
            if spectrum.voltage != voltage:
                continue
            if spectrum.wavelength >= 500:                  
                wavelengths.append(spectrum.wavelength)
                ipces.append(np.mean(spectrum.ipce))
        area = np.abs(np.trapz(y = ipces, x = wavelengths))
        # ax.text(s = str(area), x = 700, y = np.max(ipces), color = color)
        
        ## Plot normalized IPCE
        wavelengths = []
        norm_ipces = []
        for spectrum in particle.ca_spectra:
            if spectrum.voltage != voltage:
                continue
            if spectrum.wavelength > 475:
                if np.mean(spectrum.pec) < 0:
                    mfc = 'white'
                else:
                    mfc = color     
                marker = 'o'                     
                ax.scatter(spectrum.wavelength, np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
                ax.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                wavelengths.append(spectrum.wavelength)
                norm_ipces.append(np.mean(spectrum.ipce)/area)
        ax.plot(wavelengths, norm_ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V')
    
    ## Make pretty
    ax.set_xlim(400, 900)
    ax.legend( loc = 'upper right')
    scatt_plus = plt.scatter(0, 0, facecolors = 'black', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    scatt_minus = plt.scatter(0, 0, facecolors = 'white', edgecolors = 'black', s = 100, marker = marker, linewidth = 3)
    fig.legend(handles = [scatt_plus, scatt_minus], labels = ['IPCE +', 'IPCE -'], ncols = 2, bbox_to_anchor=(0.68, 0.93))
    
    fig.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory()
        fig.savefig(save_dir + str(particle.name) + ' nm Normalized Photocurrent v Voltage v2'  + '.svg', format = 'svg', bbox_inches='tight')
        plt.close(fig)
        
        
#%% Photocurrent v. molecule v. voltage -0.4 & -0.2V

''' Photocurrent v. voltage v. molecule updated cosmetics for -0.4 & -0.2V ONLY'''

all_particles = cotapp_particles + control_particles + bpdt_particles + tpdt_particles + mtapp_particles
molecules = ['Co-TAPP-SMe', 'Ni-TAPP-SMe', 'Zn-TAPP-SMe', 'H2-TAPP-SMe', 'BPDT', 'TPDT', 'Au mirror', 'FTO']
sizes = [20, 35, 57, 70]
# sizes = [20]
color_dict = {'Co-TAPP-SMe' : 'blue', 'Ni-TAPP-SMe' : 'red', 'Zn-TAPP-SMe' : 'green', 'H2-TAPP-SMe' : 'purple', 'BPDT' : 'brown', 'TPDT' : 'pink', 'Au mirror' : 'darkgoldenrod', 'FTO' : 'grey'}
my_cmap = plt.get_cmap('viridis')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5) 
voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]
voltages = [-0.4, -0.2]


# For MLAggs

for size in sizes:

    fig, ax = plt.subplots(1, 1, figsize=[16,10])
    ax2 = ax.twinx()
    ax.set_title(str(size) + ' nm MLAgg Photocurrent v. Potential', fontsize = 'x-large')
    ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
    ax2.set_ylabel('IPCE (%)', fontsize = 'x-large', rotation = 270, labelpad = 30)
    ax.set_ylabel('Scattering Intensity', fontsize = 'x-large')
    
    for i, molecule in enumerate(molecules):  
        
        if molecule == 'Co-TAPP-SMe':
            linestyle = 'solid'
            zorder = 0
        else:
            linestyle = 'dashed'
            zorder = 1
            
        for particle in all_particles:
            
            if particle.name in skip_ca_particles:
                continue

            if particle.size != size and particle not in control_particles:
                continue

            if molecule not in particle.name:
                continue
            
            ## Plot DF
            # ax.plot(particle.df_avg.x, particle.df_avg.y - particle.df_avg.y.min(), color = color_dict[molecule], linestyle = linestyle, alpha = 0.2, zorder = 0)
            
            ## Plot IPCE
            offset = 0.025
            color = color_dict[molecule]
            for voltage in voltages:    
                if voltage == -0.4:
                    alpha = 0.1
                else:
                    alpha = 0.3
                wavelengths = []
                ipces = []
                for spectrum in particle.ca_spectra:
                    if spectrum.voltage != voltage:
                        continue
                    if spectrum.wavelength > 40:
                        if np.mean(spectrum.pec) < 0:
                            mfc = 'white'
                        else:
                            mfc = color     
                        marker = 'o'                     
                        # ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce), facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
                        # ipce_err = (np.std(spectrum.ipce)/np.sqrt(len(spectrum.ipce))) 
                        # ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce), yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
                        wavelengths.append(spectrum.wavelength)
                        ipces.append(np.mean(spectrum.ipce))
                # ax2.plot(wavelengths, ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V', linestyle = linestyle)
                ax2.fill_between(wavelengths, np.array(ipces) + (i*offset), (i*offset), color = color, alpha = alpha, zorder = zorder)
                ax2.text(s = str(molecule), x = 380, y = (i*offset), color = color)
            
            # ## Plot normalized IPCE
            # for voltage in voltages:    
            #     color = cmap(norm(voltage)) 
                
            #     ## Calculte IPCE area under curve 500 - 850nm
            #     wavelengths = []
            #     ipces = []
            #     for spectrum in particle.ca_spectra:
            #         if spectrum.voltage != voltage:
            #             continue
            #         if spectrum.wavelength >= 500:                  
            #             wavelengths.append(spectrum.wavelength)
            #             ipces.append(np.mean(spectrum.ipce))
            #     area = np.abs(np.trapz(y = ipces, x = wavelengths))
            #     # ax2.text(s = str(area), x = 700, y = np.max(ipces), color = color)
                
            #     ## Plot normalized IPCE
            #     wavelengths = []
            #     norm_ipces = []
            #     for spectrum in particle.ca_spectra:
            #         if spectrum.voltage != voltage:
            #             continue
            #         if spectrum.wavelength > 40:
            #             if np.mean(spectrum.pec) < 0:
            #                 mfc = 'white'
            #             else:
            #                 mfc = color     
            #             marker = 'o'                     
            #             ax2.scatter(spectrum.wavelength, np.mean(spectrum.ipce)/area, facecolors = mfc, edgecolors = color, s = 150, marker = marker, linewidth = 4, zorder = 2)
            #             ipce_err = (np.std(spectrum.ipce)/area/np.sqrt(len(spectrum.ipce))) 
            #             ax2.errorbar(spectrum.wavelength,  np.mean(spectrum.ipce)/area, yerr = ipce_err, capsize = 10, elinewidth = 2, capthick = 3, ecolor = color, zorder = 1)
            #             wavelengths.append(spectrum.wavelength)
            #             norm_ipces.append(np.mean(spectrum.ipce)/area)
            #     ax2.plot(wavelengths, norm_ipces, color = color, zorder = 0, linewidth = 2, label = str(voltage) + ' V', linestyle = linestyle)
        
        ## Make pretty
        ax.set_xlim(350, 870)
        # ax2.legend( loc = 'upper right')
        v2 = ax2.fill_between([0], [0], color = 'black', alpha = 0.3, zorder = zorder)
        v4 = ax2.fill_between([0], [0], color = 'black', alpha = 0.1, zorder = zorder)
        fig.legend(handles = [v4, v2], labels = ['-0.4 V', '-0.2 V'], ncols = 1, bbox_to_anchor=(0.9, 0.93))
        
        fig.tight_layout()
        
        ## Save
        if save == True:
            save_dir = get_directory()
            fig.savefig(save_dir + str(size) + ' nm Photocurrent v Voltage'  + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)