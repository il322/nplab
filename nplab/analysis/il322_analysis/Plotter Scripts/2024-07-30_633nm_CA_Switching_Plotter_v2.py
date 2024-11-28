# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:11:15 2024

@author: il322

Plotter for Co-TAPP-SMe 633nm SERS with CA Switching
Organized so that can save 'Particles' list to spydata
Added functionality to save timescan, ca, and plots to spydata and h5

Data: 2024-07-30_Co-TAPP-SMe_MLAgg_EChem_SERS_633nm.h5

(samples:
     2024-07-22_Co-TAPP-SMe_60nm_MLAgg_on_ITO_b)

"""

from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
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
from nplab import ArrayWithAttrs

from lmfit.models import GaussianModel


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

## Raw data h5 File
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-30_Co-TAPP-SMe_MLAgg_EChem_SERS_633nm.h5")

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-07-30_Processed_Data.h5")


#%% Plotting functions

def plot_timescan(particle, save = False):
    
    timescan = particle.timescan

    ## Plot timescan
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(timescan.Y),1)
    v_min = timescan.Y.min()
    v_max = np.percentile(timescan.Y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('Time (s)', fontsize = 'large')
    ax.set_yticks(t_plot, labels = t_plot * particle.chunk_size, fontsize = 'large', )
    # ax.set_xlim(300,1700)
    ax.set_title('633nm SERS Timescan' + 's\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot + 0.5, timescan.Y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '633nm SERS Timescan' + '.svg', format = 'svg')
        plt.close(fig)
        

def plot_timescan_ca(particle, save = False, save_to_h5 = False):
    
    timescan = particle.timescan
    ca = particle.ca

    ## Plot timescan
    fig, axes = plt.subplots(1, 2, figsize=[18,14], width_ratios=(1,4), sharey = True)
    ax = axes[1]
    ax2 = axes[0]
    t_plot = np.arange(0,len(timescan.Y),1) * timescan.chunk_size
    v_min = timescan.Y.min()
    v_max = np.percentile(timescan.Y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'x-large')
    ax2.set_ylabel('Time (s)', fontsize = 'x-large')
    # ax.set_yticks(t_plot)

    ax.set_title('633 nm 1 $\mu$W SERS Timescan', fontsize = 'x-large', pad = 20)
    pcm = ax.pcolormesh(timescan.x, t_plot + (0.5 * timescan.chunk_size), timescan.Y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'SERS Intensity', size = 'x-large', rotation = 270, labelpad=30)
    clb.ax.tick_params(labelsize='large')

    ax2.plot(ca.y, ca.x, zorder = 10)
    t_ticks = np.linspace(0, int(ca.x.max()), num = 11)
    ax2.set_yticks(t_ticks, labels = t_ticks.astype(int), fontsize = 'large')
    ax2.set_ylim(0, 600)
    # i_ticks = [int(np.round(ca.y.min(), 0)), 0, int(np.round(ca.y.min(), 0) * -1)] 
    i_ticks = [int(np.round(ca.y.min(), 0)), 0, int(np.round(ca.y.max(), 0))] 
    ax2.set_xticks(i_ticks, labels = i_ticks, fontsize = 'large')
    ax2.set_xlabel('Current ($\mu$A)', fontsize = 'x-large')
    ax2.tick_params(axis='both', which='major', labelsize='large')
    ax2.set_title('CA: [' + str(ca.v[0]) + ' V; ' + str(ca.v[1]) + ' V]', fontsize = 'x-large', pad = 20)
    
    switch_times = np.linspace(60, 600, 10)
    for switch in switch_times:
        plt.hlines(y = switch, xmin = -500, xmax = 2200, color = 'black', linestyle = '--', clip_on = False, linewidth = 5, alpha = 0.5, zorder = 8)
        if np.mod(switch, 120) == 0:
            v = 1
        else:
            v = 0
        ax2.text(s = str(ca.v[v]) + ' V', x = ax2.get_xlim()[0]*0.55, y = switch - 35, fontsize = 'large', horizontalalignment='right')
    ax.set_xlim(250,2200)   
    fig.suptitle(particle.name, fontsize = 'xx-large')
    plt.tight_layout()
    
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_633nm SERS Timescan CA Switch' + '.svg', format = 'svg')
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
        group.create_dataset(name = timescan.name + '_jpg_%d', data = fig_array)
        

def plot_timescan_background_ca(particle, save = False, save_to_h5 = False):
    
    timescan = particle.timescan
    ca = particle.ca

    ## Plot timescan
    fig, axes = plt.subplots(1, 2, figsize=[18,14], width_ratios=(1,4), sharey = True)
    ax = axes[1]
    ax2 = axes[0]
    t_plot = np.arange(0,len(timescan.Y),1) * timescan.chunk_size
    v_min = timescan.Baseline.min()
    v_max = np.percentile(timescan.Baseline, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'x-large')
    ax2.set_ylabel('Time (s)', fontsize = 'x-large')
    # ax.set_yticks(t_plot)

    ax.set_title('Background 633 nm 1 $\mu$W Timescan', fontsize = 'x-large', pad = 20)
    pcm = ax.pcolormesh(timescan.x, t_plot + (0.5 * timescan.chunk_size), timescan.Baseline, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'Intensity', size = 'x-large', rotation = 270, labelpad=30)
    clb.ax.tick_params(labelsize='large')

    ax2.plot(ca.y, ca.x, zorder = 10)
    t_ticks = np.linspace(0, int(ca.x.max()), num = 11)
    ax2.set_yticks(t_ticks, labels = t_ticks.astype(int), fontsize = 'large')
    ax2.set_ylim(0, 600)
    # i_ticks = [int(np.round(ca.y.min(), 0)), 0, int(np.round(ca.y.min(), 0) * -1)] 
    i_ticks = [int(np.round(ca.y.min(), 0)), 0, int(np.round(ca.y.max(), 0))] 
    ax2.set_xticks(i_ticks, labels = i_ticks, fontsize = 'large')
    ax2.set_xlabel('Current ($\mu$A)', fontsize = 'x-large')
    ax2.tick_params(axis='both', which='major', labelsize='large')
    ax2.set_title('CA: [' + str(ca.v[0]) + ' V; ' + str(ca.v[1]) + ' V]', fontsize = 'x-large', pad = 20)
    
    switch_times = np.linspace(60, 600, 10)
    for switch in switch_times:
        plt.hlines(y = switch, xmin = -490, xmax = 2200, color = 'black', linestyle = '--', clip_on = False, linewidth = 5, alpha = 0.5, zorder = 8)
        if np.mod(switch, 120) == 0:
            v = 1
        else:
            v = 0
        ax2.text(s = str(ca.v[v]) + ' V', x = ax2.get_xlim()[0]*0.55, y = switch - 35, fontsize = 'large', horizontalalignment='right')
    ax.set_xlim(250,2200)   
    fig.suptitle(particle.name, fontsize = 'xx-large')
    plt.tight_layout()
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_633nm Background Timescan CA Switch' + '.svg', format = 'svg')
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
        group.create_dataset(name = timescan.name + '_background_jpg_%d', data = fig_array)

    
my_cmap = plt.get_cmap('inferno')       
    
    
def plot_peak_areas(particle, save = False, save_to_h5 = False):
       
    timescan = particle.timescan
    
    time = np.arange(0, len(timescan.Y) * timescan.chunk_size, timescan.chunk_size, dtype = int)   
        
    colors = ['grey', 'purple', 'red', 'brown', 'darkgreen', 'darkblue', 'deeppink', 'yellow', 'cyan', 'orange', 'black']
    
    fig, axes = plt.subplots(10 + 1,1,figsize=[14,22], sharex = True)
    fig.suptitle('633 nm SERS Peak Area - CA Switch ' + str(particle.ca.v) + '\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Time (s)', size = 'x-large')
        
    peaks = timescan.peaks
    for i, peak in enumerate(timescan.peaks[0]):
        
        ## Peak area ratios to plot for y
        y = [] 
        ax = axes[i]    
                    
        for j in range(0, len(timescan.Y)):
            # if peaks[j][i].error:
            #     y.append(-1)
            # elif peaks[j][i].name == '1435/1420':
            #     y.append(peaks[j][i].area)
            # else:
            y.append((peaks[j][i].area - peaks[0][i].area)/timescan.peaks[0][i].area)          
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = time, y = y)
                              
        ## Plot peaks
        color = colors[i]
        if 'Avg' in particle.name:
            alpha = 0.5
        else:
            alpha = 1
        ax.plot(peak_spec.x, peak_spec.y * 100, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 4, alpha = alpha)
        ax.scatter(peak_spec.x[::2], peak_spec.y[::2] * 100, s = 200, label = '0.0 V', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
        ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2] * 100, s = 200, label = str(particle.ca.v[1]) + ' V', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)

   
        # Errorbars, if avg particle
        try:                
            y_error = []
            for j in range(0, len(timescan.Y)):
                if j == 0:
                    this_error = 0
                else:
                    this_error_subtraction = (peaks[j][i].area_error + peaks[0][i].area_error)
                    this_error = peak_spec.y[j] * ((this_error_subtraction/(peaks[j][i].area - peaks[0][i].area)) + (peaks[0][i].area_error/peaks[0][i].area))
                y_error.append(this_error)   
            # print(this_error)
            y_error = np.array(y_error)
            ax.errorbar(peak_spec.x, peak_spec.y * 100, yerr = y_error * 100, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
        except:
            y_error = None
            
                
        ## Plot background sum
        y = []
        for j in range(0, len(timescan.Y)):
            y.append(timescan.BaselineSum[j])
        y = np.array(y)
        background_spec = SERS.SERS_Spectrum(x = time, y = y)                             
        color = colors[len(colors)-1]
        axes[len(axes)-1].plot(background_spec.x, background_spec.y, label = 'Background Sum', color = color, alpha = 0.1, zorder = 4, linewidth = 4)
        axes[len(axes)-1].scatter(background_spec.x[::2], background_spec.y[::2], s = 200, label = '0.0 V', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
        axes[len(axes)-1].scatter(background_spec.x[1::2], background_spec.y[1::2], s = 200, label = str(particle.ca.v[1]) + ' V', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)

        # if 'Avg' in particle.name:
            
        #     fit_x = background_spec.x
        #     fit_y = exp_decay(fit_x,
        #                       particle.ms[-1],
        #                       particle.ts[-1],
        #                       particle.bs[-1])
        #     # axes[len(axes)-1].plot(fit_x, fit_y, color = color, zorder = 4, linewidth = 3, alpha = 0.5)
        ax.legend(loc = 'upper right')   
        ylim = ax.get_ylim()            
        
    axes[4].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'xx-large')
    axes[len(axes)-1].set_ylabel('Background', size = 'large')
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_CA Switch SERS Peak Area' + '.svg', format = 'svg')
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
        group.create_dataset(name = 'Peak_Area_jpg_%d', data = fig_array)
        
        
def plot_peak_positions(particle, save = False, save_to_h5 = False):
       
    timescan = particle.timescan
    
    time = np.arange(0, len(timescan.Y) * timescan.chunk_size, timescan.chunk_size, dtype = int)   
        
    colors = ['grey', 'purple', 'red', 'brown', 'darkgreen', 'darkblue', 'deeppink', 'yellow', 'cyan', 'orange', 'black']
    
    fig, axes = plt.subplots(10,1,figsize=[14,20], sharex = True)
    fig.suptitle('633 nm SERS Peak Position - CA Switch ' + str(particle.ca.v) + '\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Time (s)', size = 'x-large')
        
    peaks = timescan.peaks
    for i, peak in enumerate(timescan.peaks[0]):
        
        # if i in peaks_list:

        ## Peak area ratios to plot for y
        y = [] 
        ax = axes[i]    
                    
        for j in range(0, len(timescan.Y)):
            # if peaks[j][i].error:
            #     y.append(-1)
            # elif peaks[j][i].name == '1435/1420':
            #     y.append(peaks[j][i].area)
            # else:
            y.append(peaks[j][i].mu - timescan.peaks[0][i].mu)          
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = time, y = y)
                              
        ## Plot peaks
        color = colors[i]
        # if 'Avg' in particle.name:
            # alpha = 0.5
        # else:
        alpha = 1
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 4, alpha = alpha)
        ax.scatter(peak_spec.x[::2], peak_spec.y[::2], s = 200, label = '0.0 V', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
        ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2], s = 200, label = str(particle.ca.v[1]) + ' V', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)
        
        # ## ylim
        # if peaks[j][i].name == '1435/1420':
        #     pass
        # # else:
        # #     try:
        # #         ax.set_ylim(np.nanpercentile(peak_spec.y, 0.5), 1.05)
        # #     except: pass
        # if 'Avg' in particle.name:
            
        #     fit_x = peak_spec.x
        #     fit_y = exp_decay(fit_x,
        #                       particle.ms[i],
        #                       particle.ts[i],
        #                       particle.bs[i])
        #     ax.plot(fit_x, fit_y, color = color, zorder = 4, linewidth = 3, alpha = 1)

        ylim = ax.get_ylim()  

        # Errorbars, if avg particle
        try:                
            y_error = []
            for j in range(0, len(timescan.Y)):
                this_error = (peaks[j][i].mu_error) + (peaks[0][i].mu_error) 
                y_error.append(this_error)   
            y_error = np.array(y_error)[0]
            ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
        except:
            y_error = None
         
        ax.legend(loc = 'upper right')
    
        # ax.set_ylim(ylim)          
          
    axes[5].set_ylabel('$\Delta$ Peak Position (cm$^{-1}$)', size = 'x-large')
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_CA Switch SERS Peak Positions' + '.svg', format = 'svg')
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
        group.create_dataset(name = 'Peak_Position_jpg_%d', data = fig_array)
 
# Test plotting

# particle = particles[0]
# plot_timescan_ca(particle, save_to_h5 = True, save = True)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
        
# particle = particles[24]
# plot_timescan(particle, save = False)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
# plot_peak_areas(avg_particles[8], save = False)


#%% Functions to save timescan and ca data to h5 - should add these as functions to classes in SERS_tools

def save_timescan(particle, overwrite = False):
    
    timescan = particle.timescan

    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))

    ## Get timescan attributes, remove troublesome attributes (too large for h5 saving)
    try:
        timescan.__dict__.pop('dset')
    except: pass
    attrs = deepcopy(timescan.__dict__)
    attrs.pop('Y')
    attrs.pop('Y_raw')
    attrs.pop('y_raw')
    attrs.pop('rc_params')
    attrs.pop('peaks')

    ## Save Y raw timescan (Y-raw attribute is too large for h5 file attributes)
    group.create_dataset(name = timescan.name + '_raw_%d', data = timescan.Y_raw, attrs = attrs)

    ## Save Y processed timescan
    group.create_dataset(name = timescan.name + '_%d', data = timescan.Y, attrs = attrs)
    
    
def save_ca(particle, overwrite = False):
    
    ca = particle.ca
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    ## Get CA attributes, remove troublesome attributes
    try:
        ca.__dict__.pop('dset')
    except: pass
    attrs = deepcopy(ca.__dict__)
    attrs.pop('rc_params')
    
    ## Save 
    group.create_dataset(name = ca.name + '_%d', data = ca.y, attrs = attrs)
    
    
def save_peaks(particle):
    
    peaks = particle.timescan.peaks
    peaks = np.array(peaks)
    example_peak = peaks[0][0]
    
    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))
    
    
    ## For each peak attribute, create a dataset with size equal to number of peaks 
    for attr in example_peak.__dict__:    
        try:
            save_attr = np.zeros(shape = peaks.shape)
            
            for i, scan in enumerate(peaks):
                for j, peak in enumerate(scan):
                    save_attr[i][j] = peak.__dict__[attr]
                    
            group.create_dataset(name = 'Fitted Peak ' + str(attr), data = save_attr)
        except: pass
    
    # ## Get CA attributes, remove troublesome attributes
    # try:
    #     ca.__dict__.pop('dset')
    # except: pass
    # attrs = deepcopy(ca.__dict__)
    # attrs.pop('rc_params')
    
    # ## Save 
    # group.create_dataset(name = ca.name + '_%d', data = ca.y, attrs = attrs)
    
        

#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-07-30 Co-TAPP-SMe MLAgg EChem SERS Analysis\_' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Function to add & process SERS timescan & CA for each particle

# def process_timescan_ca(particle):
    
#     spectrum = particle.h5_address['Co-TAPP-SMe_633nm_SERS_CA_0']
#     ca = particle.h5_address['Co-TAPP-SMe_CA_Switch_x5_0']
#     spectrum = SERS.SERS_Timescan(spectrum)
#     spectrum.x = linear(spectrum.x, *cal_matrix)
#     spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
#     spectrum.truncate(truncate_range[0], truncate_range[1])
#     spectrum.calibrate_intensity(R_setup = R_setup,
#                                  dark_counts = dark_counts.y_smooth)

#     timescan = spectrum
#     particle.chunk_size = 20
#     timescan.chunk(particle.chunk_size)

#     ca_y = np.array(ca)[1]
#     ca_v = ca.attrs['Levels_v (V)']
#     ca = SERS.SERS_Spectrum(x = ca.attrs['Time (s)'], y = ca_y*1e6)
#     ca.v = ca_v
#     particle.ca = ca

#     for i, spectrum in enumerate(timescan.Y):
#           spectrum = spt.remove_cosmic_rays(spectrum, threshold = 9)
#           spectrum_baseline = spt.baseline_als(spectrum, 1e3, 1e-2, niter = 10)
#           spectrum_baselined = spectrum - spectrum_baseline
#           spectrum = spectrum_baselined
#           timescan.Y[i] = spectrum

#     particle.timescan = timescan


def process_timescan(particle, chunk_size = 20):
    
    ## Get timescan object
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    for key in keys:
        if 'SERS' in key:
            timescan = (my_h5[particle.scan][particle.particle_number][key])
            name = key
    
    ## Timescan calibration
    timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'])
    timescan.x = linear(timescan.x, *cal_matrix)
    timescan.x = spt.wl_to_wn(timescan.x, 632.8)
    timescan.truncate(truncate_range[0], truncate_range[1])
    timescan.calibrate_intensity(R_setup = R_setup,
                                 dark_counts = dark_counts.y_smooth)
    timescan.name = name
    
    ## Chunk timescan for improved S/N
    timescan.chunk(chunk_size)
    timescan.chunk_size = chunk_size
    timescan.Y = timescan.Y/chunk_size
    timescan.Baseline = deepcopy(timescan.Y)
    timescan.BaselineSum = np.zeros(len(timescan.Y))
    timescan.peaks = []
    
    ## Background subtract each spectrum
    for i, spectrum in enumerate(timescan.Y):
        spectrum = SERS.SERS_Spectrum(x = timescan.x, y = spectrum)
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        timescan.Y[i] = spectrum.y_baselined
        timescan.Baseline[i] = spectrum.baseline
        timescan.BaselineSum[i] = np.sum(timescan.Baseline[i])
        timescan.peaks.append([])
    
    particle.laser_power = timescan.laser_power
    timescan.__dict__.pop('dset')
    particle.timescan = timescan
    
    
def process_ca(particle):
    
    ca = my_h5[particle.scan][particle.particle_number]['Co-TAPP-SMe_CA_Switch_x5_0']
    ca_y = np.array(ca)[1]
    ca_v = ca.attrs['Levels_v (V)']
    ca = SERS.SERS_Spectrum(x = ca.attrs['Time (s)'], y = ca_y*1e6, particle_name = particle.name)
    ca.name = 'Co-TAPP-SMe_CA_Switch_x5_0'
    ca.v = ca_v
    particle.ca = ca  
    

#%% Peak fitting functions

def linear(x, m, b):
    return (m*x) + b


def quadratic(x, a, b, c):
    return (a*x**2 + b*x +c)


def exp_decay(x, m, t, b):
    return m * np.exp(-t * x) + b

       
def gaussian(x, height, mu, width, baseline):
    
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    
    return height * np.exp(-(((x - mu)**2)/(2 * sigma **2))) + baseline


def gaussian2(x, 
              height1, mu1, width1, baseline1,
              height2, mu2, width2, baseline2):
    
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))    
    
    y1 = height1 * np.exp(-(((x - mu1)**2)/(2 * sigma1 **2))) + baseline1
    y2 = height2 * np.exp(-(((x - mu2)**2)/(2 * sigma2 **2))) + baseline2
    
    return y1 + y2


def gaussian3(x, 
              height1, mu1, width1, baseline1,
              height2, mu2, width2, baseline2,
              height3, mu3, width3, baseline3):
    
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))    
    sigma3 = width3 / (2 * np.sqrt(2 * np.log(2)))
    
    y1 = height1 * np.exp(-(((x - mu1)**2)/(2 * sigma1 **2))) + baseline1
    y2 = height2 * np.exp(-(((x - mu2)**2)/(2 * sigma2 **2))) + baseline2
    y3 = height3 * np.exp(-(((x - mu3)**2)/(2 * sigma3 **2))) + baseline3
    
    return y1 + y2 + y3
    

def lorentzian(x, height, mu, fwhm):
    I = height
    x0 = mu
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot
    return y


#%% Peak fit function for single Gaussian regions  

def fit_gaussian_timescan(particle, fit_range, peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False):

    '''
    Fit single gaussian peak to given range of each spectrum in timescan
    Adds peaks to timescan.peaks[i]
    '''    
    
    timescan = particle.timescan

    for i, spectrum in enumerate(timescan.Y):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(timescan.x-fit_range[0]).argmin(), np.abs(timescan.x-fit_range[1]).argmin()+1]
        x = timescan.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum_smooth = spt.butter_lowpass_filt_filt(spectrum, cutoff = 3000, fs = 30000)
            y = spectrum_smooth[fit_range_index[0]:fit_range_index[1]]
        
    
        # Fit
        
        ## Initial guesses
        i_max = y.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height0 = y[i_max]
        mu0 = x[i_max] # centre x position of guessed peak
        baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
        i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
        width0 = sigma0 * 2 * np.sqrt(2*np.log(2))
        # sigma0 = 3
        p0 = height0, mu0, width0, baseline0
        
        ## Perform fit (height, mu, width, baseline)
        start = time.time()
        
        try:
            popt, pcov = curve_fit(f = gaussian, 
                                xdata = x,
                                ydata = y, 
                                p0 = p0,
                                bounds=(
                                    (0, x.min(), 0, y.min()),
                                    (height0 * 1.2, x.max(), (x.max() - x.min()) * 2, height0),),)
            
        except:
            popt = [height0, mu0, sigma0, baseline0]
            # print('\nFit Error')
          
        finish = time.time()
        runtime = finish - start
            
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        
        ## Get fit data
        height = popt[0]
        mu = popt[1]
        width = popt[2]
        baseline = popt[3]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt)
        residuals = y - y_fit
        residuals_sum = np.abs(residuals).sum()
        R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
        
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        
        ## Screening poor fits
        if R_sq < R_sq_thresh or np.isnan(R_sq):
            # print('\nPoor Fit')
            # print(particle.name)
            # print(spectrum.name)
            # print('R^2 = ' + str(np.round(R_sq, 3)))
            # print('Guess:', np.array(p0))
            # print('Fit:  ', popt)
            error = True
        else:
            error = False
        
        ## Add fit data to peak class
        this_peak = Peak(*popt)
        this_peak.area = area
        this_peak.sigma = sigma
        this_peak.name = peak_name
        this_peak.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        this_peak.R_sq = R_sq
        this_peak.residuals = SERS.SERS_Spectrum(x = x, y = residuals)
        this_peak.error = error        
        
        ## Add peak class to timescan class 'peaks' list
        particle.timescan.peaks[i].append(this_peak)
           
        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[14,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0] - 3, fit_range[1] + 3)
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            plt.show()


# Testing fit ranges

# particle = particles[24]
# fit_gaussian_timescan(particle = particle, fit_range = [390, 435], smooth_first = True, plot = True) ## grey
# fit_gaussian_timescan(particle = particle, fit_range = [990, 1040], smooth_first = False, plot = True) ## purple
# fit_gaussian_timescan(particle = particle, fit_range = [1170, 1250], smooth_first = True, plot = True) ## red
# fit_gaussian_timescan(particle = particle, fit_range = [1300, 1390], smooth_first = True, plot = True) ## green+blue - needs double or triple fit
# fit_gaussian_timescan(particle = particle, fit_range = [1610, 1650], smooth_first = True,  plot = True) ## orange   


#%% Peak fit function for triple Gaussian region - 1300-1390 block

def fit_gaussian3_timescan_1300(particle, fit_range = [1365, 1450], peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False, save = False):
    
    '''
    Fit triple gaussian peak to given range of each spectrum in timescan (highly tuned for 1365 - 1450cm-1 region)
    Adds peaks to timescan.peaks[i]
    '''       
    
    timescan = particle.timescan

    for i, spectrum in enumerate(timescan.Y):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(timescan.x-fit_range[0]).argmin(), np.abs(timescan.x-fit_range[1]).argmin()+1]
        x = timescan.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum_smooth = spt.butter_lowpass_filt_filt(spectrum, cutoff = 5000, fs = 30000)
            y = spectrum_smooth[fit_range_index[0]:fit_range_index[1]]

        # return x

        # Fit
        
        ## Initial guesses

        ### First peak
        x1 = x[:14]
        y1 = y[:14]
        i_max = y1.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height1 = y1[i_max]
        mu1 = x1[i_max] # centre x position of guessed peak
        width1 = (x1.max()-x1.min())/2
        baseline1 = 0
        
        ### Second peak
        x2 = x[14:26]
        y2 = y[14:26]
        i_max = y2.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height2 = y2[i_max]
        mu2 = x2[i_max] # centre x position of guessed peak
        width2 = 14
        baseline2 = 0
        
        ### Third peak
        # print(x[28])
        x3 = x[29:]
        y3 = y[29:]
        i_max = y3.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height3 = y3[i_max]
        mu3 = x3[i_max] # centre x position of guessed peak
        width3 = 13
        baseline3 = 0
        # baseline3 = (y3[0] + y3[len(y3)-1])/2 # height of baseline guess
        # i_half = np.argmax(3 >= (height3 + baseline3)/2) # Index of first argument to be at least halfway up the estimated bell
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        # sigma3 = (mu3 - x3[i_half]) / np.sqrt(2*np.log((height3 - baseline3)/(y3[i_half] - baseline3)))
        # width3 = sigma3 * 2 * np.sqrt(2*np.log(2))
        # sigma0 = 3
        # p0 = height0, mu0, width0, baseline0
        
        ## Perform fit (height, mu, width, baseline)
        start = time.time()
        
        ## Initial guesses
        
        ### First peak
        # height1 = y.max()/3
        # mu1 = 1399
        # width1 = 20
        # baseline1 = 0
        
        # ### Second peak
        # height2 = y.max()/1.7
        # mu2 = 1425
        # width2 = 15
        # baseline2 = 0
        
        # # ### Third peak
        # height3 = y.max()
        # mu3 = 1434
        # width3 = 15
        # baseline3 = 0
        
        p0 = [
                height1, mu1, width1, baseline1,
                height2, mu2, width2, baseline2,
                height3, mu3, width3, baseline3
             ]
        
        # lower_bounds = (
        #                 0, mu1 - 5, width1 - 5, y.min(),
        #                 0, mu2 - 5, width2 - 1, y.min(),
        #                 0, mu3 - 2, width3 - 1, y.min()
        #                 )
        
        # upper_bounds = (
        #                 height1 * 2, mu1 + 5, width1 + 2, 100,
        #                 height2 * 2, mu2 + 5, width2 + 1, 100,
        #                 height3 * 2, mu3 + 2, width3 + 1, 100
        #                 )

        lower_bounds = (
                        0, x1.min(), 0, y.min(),
                        height2 * 0.9, mu2 - 2, width2 - 5, y.min(),
                        height3 * 0.9, mu3 - 2, width3 - 5, y.min()
                        )
        
        upper_bounds = (
                        height2, x1.max(), x.max()-x.min(), 100,
                        height2 * 1.05, mu2 + 2, width2 + 5, 100,
                        height3 * 1.05, mu3 + 4, width3 + 5, 100
                        )        
        
        try:
            popt, pcov = curve_fit(f = gaussian3, 
                                xdata = x,
                                ydata = y, 
                                p0 = p0,
                                bounds=((lower_bounds),(upper_bounds),),)
            
        except:
            popt = p0
            # print('\nFit Error')
          
        finish = time.time()
        runtime = finish - start
        # print()
        # print('\n')
        # print(particle.name)
        # print(spectrum.name)
        # # print('R^2 = ' + str(np.round(R_sq, 3)))
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        
        
        # Get fit data
        
        ## Peak 1
        height = popt[0]
        mu = popt[1]
        width = popt[2]
        baseline = popt[3]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[0:4])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak1 = Peak(*popt[0:4])
        peak1.area = area
        peak1.sigma = sigma
        peak1.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak1.name = '1330'

        ## Peak 2
        height = popt[4]
        mu = popt[5]
        width = popt[6]
        baseline = popt[7]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[4:8])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak2 = Peak(*popt[4:8])
        peak2.area = area
        peak2.sigma = sigma
        peak2.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak2.name = '1355'

        ## Peak 3
        height = popt[8]
        mu = popt[9]
        width = popt[10]
        baseline = popt[11]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[8:12])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak3 = Peak(*popt[8:12])
        peak3.area = area
        peak3.sigma = sigma
        peak3.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak3.name = '1375'
        
        ## Sum & residuals
        y_fit = peak1.spectrum.y + peak2.spectrum.y + peak3.spectrum.y
        residuals = y - y_fit
        residuals_sum = np.abs(residuals).sum()
        R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
        ### R_sq is total R_sq for triple
        peak1.R_sq = R_sq
        peak2.R_sq = R_sq
        peak3.R_sq = R_sq
        
        # if peak_name is None:
        #     peak_name = str(int(np.round(np.mean(fit_range),0)))
        
        ## Screening poor fits
        if R_sq < R_sq_thresh or np.isnan(R_sq):
            # print('\nPoor Fit')
            # print(particle.name)
            # print(spectrum.name)
            # print('R^2 = ' + str(np.round(R_sq, 3)))
            # print('Guess:', np.array(p0))
            # print('Fit:  ', popt)
            error = True
        else:
            error = False
        peak1.error = error
        peak2.error = error
        peak3.error = error     
        
        ## Add peaks class to timescan class 'peaks' list
        particle.timescan.peaks[i].append(peak1)
        particle.timescan.peaks[i].append(peak2)        
        particle.timescan.peaks[i].append(peak3)

        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            # ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            plt.show()
            
        ## Save
        if save == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            
            save_dir = get_directory(particle.name + '\\Triple Fit')
            fig.savefig(save_dir + particle.name + '_' + 'Scan_' + str(i) + '_TripleFit' + '.svg', format = 'svg')
            plt.close(fig)


# Testing fit ranges

# particle = particles[1]

# fit_gaussian3_timescan_1300(particle = particle, fit_range = [1300, 1390], smooth_first = True, plot = True, save = False) ## green/blue

# for spectrum in particle.powerseries:
#     try:
#         baseline = spectrum.peaks[6].baseline/spectrum.peaks[5].baseline
#     except:
#         baseline = 0
#     ratio = Peak(height = spectrum.peaks[6].height/spectrum.peaks[5].height,
#                  mu = None,
#                  width = spectrum.peaks[6].width/spectrum.peaks[5].width,
#                  baseline = baseline)
#     ratio.area = spectrum.peaks[6].area/spectrum.peaks[5].area
#     ratio.sigma = spectrum.peaks[6].sigma/spectrum.peaks[5].sigma
#     ratio.name = '1435/1420'
#     if spectrum.peaks[6].error or spectrum.peaks[5].error:
#         ratio.error = True
#     else:
#         ratio.error = False
#     spectrum.peaks.append(ratio)

# plot_peak_areas(particle, save = False)


#%% Peak fit function for triple Gaussian region - 1500-1590 block

def fit_gaussian3_timescan_1500(particle, fit_range = [1500, 1590], peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False, save = False):
    
    '''
    Fit triple gaussian peak to given range of each spectrum in timescan (highly tuned for 1365 - 1450cm-1 region)
    Adds peaks to timescan.peaks[i]
    '''       
    
    timescan = particle.timescan

    for i, spectrum in enumerate(timescan.Y):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(timescan.x-fit_range[0]).argmin(), np.abs(timescan.x-fit_range[1]).argmin()+1]
        x = timescan.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum_smooth = spt.butter_lowpass_filt_filt(spectrum, cutoff = 5000, fs = 30000)
            y = spectrum_smooth[fit_range_index[0]:fit_range_index[1]]

        # return x

        # Fit
        
        ## Initial guesses

        ### First peak
        x1 = x[:14]
        y1 = y[:14]
        i_max = y1.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height1 = y1[i_max]
        mu1 = x1[i_max] # centre x position of guessed peak
        width1 = (x1.max()-x1.min())/2
        baseline1 = 0
        
        ### Second peak
        x2 = x[14:26]
        y2 = y[14:26]
        i_max = y2.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height2 = y2[i_max]
        mu2 = x2[i_max] # centre x position of guessed peak
        width2 = 14
        baseline2 = 0
        
        ### Third peak
        # print(x[28])
        x3 = x[29:]
        y3 = y[29:]
        i_max = y3.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height3 = y3[i_max]
        mu3 = x3[i_max] # centre x position of guessed peak
        width3 = 13
        baseline3 = 0
        # baseline3 = (y3[0] + y3[len(y3)-1])/2 # height of baseline guess
        # i_half = np.argmax(3 >= (height3 + baseline3)/2) # Index of first argument to be at least halfway up the estimated bell
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        # sigma3 = (mu3 - x3[i_half]) / np.sqrt(2*np.log((height3 - baseline3)/(y3[i_half] - baseline3)))
        # width3 = sigma3 * 2 * np.sqrt(2*np.log(2))
        # sigma0 = 3
        # p0 = height0, mu0, width0, baseline0
        
        ## Perform fit (height, mu, width, baseline)
        start = time.time()
        
        ## Initial guesses
        
        ### First peak
        # height1 = y.max()/3
        # mu1 = 1399
        # width1 = 20
        # baseline1 = 0
        
        # ### Second peak
        # height2 = y.max()/1.7
        # mu2 = 1425
        # width2 = 15
        # baseline2 = 0
        
        # # ### Third peak
        # height3 = y.max()
        # mu3 = 1434
        # width3 = 15
        # baseline3 = 0
        
        p0 = [
                height1, mu1, width1, baseline1,
                height2, mu2, width2, baseline2,
                height3, mu3, width3, baseline3
             ]
        
        # lower_bounds = (
        #                 0, mu1 - 5, width1 - 5, y.min(),
        #                 0, mu2 - 5, width2 - 1, y.min(),
        #                 0, mu3 - 2, width3 - 1, y.min()
        #                 )
        
        # upper_bounds = (
        #                 height1 * 2, mu1 + 5, width1 + 2, 100,
        #                 height2 * 2, mu2 + 5, width2 + 1, 100,
        #                 height3 * 2, mu3 + 2, width3 + 1, 100
        #                 )

        lower_bounds = (
                        0, x1.min(), 0, y.min(),
                        height2 * 0.9, mu2 - 2, width2 - 5, y.min(),
                        height3 * 0.9, mu3 - 2, width3 - 5, y.min()
                        )
        
        upper_bounds = (
                        height2, x1.max(), x.max()-x.min(), 100,
                        height2 * 1.05, mu2 + 2, width2 + 5, 100,
                        height3 * 1.05, mu3 + 4, width3 + 5, 100
                        )        
        
        try:
            popt, pcov = curve_fit(f = gaussian3, 
                                xdata = x,
                                ydata = y, 
                                p0 = p0,
                                bounds=((lower_bounds),(upper_bounds),),)
            
        except:
            popt = p0
            # print('\nFit Error')
          
        finish = time.time()
        runtime = finish - start
        # print()
        # print('\n')
        # print(particle.name)
        # print(spectrum.name)
        # # print('R^2 = ' + str(np.round(R_sq, 3)))
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        
        
        # Get fit data
        
        ## Peak 1
        height = popt[0]
        mu = popt[1]
        width = popt[2]
        baseline = popt[3]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[0:4])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak1 = Peak(*popt[0:4])
        peak1.area = area
        peak1.sigma = sigma
        peak1.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak1.name = '1520'

        ## Peak 2
        height = popt[4]
        mu = popt[5]
        width = popt[6]
        baseline = popt[7]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[4:8])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak2 = Peak(*popt[4:8])
        peak2.area = area
        peak2.sigma = sigma
        peak2.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak2.name = '1555'

        ## Peak 3
        height = popt[8]
        mu = popt[9]
        width = popt[10]
        baseline = popt[11]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[8:12])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak3 = Peak(*popt[8:12])
        peak3.area = area
        peak3.sigma = sigma
        peak3.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak3.name = '1575'
        
        ## Sum & residuals
        y_fit = peak1.spectrum.y + peak2.spectrum.y + peak3.spectrum.y
        residuals = y - y_fit
        residuals_sum = np.abs(residuals).sum()
        R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
        ### R_sq is total R_sq for triple
        peak1.R_sq = R_sq
        peak2.R_sq = R_sq
        peak3.R_sq = R_sq
        
        # if peak_name is None:
        #     peak_name = str(int(np.round(np.mean(fit_range),0)))
        
        ## Screening poor fits
        if R_sq < R_sq_thresh or np.isnan(R_sq):
            # print('\nPoor Fit')
            # print(particle.name)
            # print(spectrum.name)
            # print('R^2 = ' + str(np.round(R_sq, 3)))
            # print('Guess:', np.array(p0))
            # print('Fit:  ', popt)
            error = True
        else:
            error = False
        peak1.error = error
        peak2.error = error
        peak3.error = error     
        
        ## Add peaks class to timescan class 'peaks' list
        particle.timescan.peaks[i].append(peak1)
        particle.timescan.peaks[i].append(peak2)        
        particle.timescan.peaks[i].append(peak3)

        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            # ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            plt.show()
            
        ## Save
        if save == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            
            save_dir = get_directory(particle.name + '\\Triple Fit')
            fig.savefig(save_dir + particle.name + '_' + 'Scan_' + str(i) + '_TripleFit' + '.svg', format = 'svg')
            plt.close(fig)


# Testing fit ranges

# particle = particles[1]
# fit_gaussian3_timescan_1500(particle = particle, fit_range = [1500, 1600], smooth_first = True, plot = True, save = False) ## pink/yellow/cyan

# for spectrum in particle.powerseries:
#     try:
#         baseline = spectrum.peaks[6].baseline/spectrum.peaks[5].baseline
#     except:
#         baseline = 0
#     ratio = Peak(height = spectrum.peaks[6].height/spectrum.peaks[5].height,
#                  mu = None,
#                  width = spectrum.peaks[6].width/spectrum.peaks[5].width,
#                  baseline = baseline)
#     ratio.area = spectrum.peaks[6].area/spectrum.peaks[5].area
#     ratio.sigma = spectrum.peaks[6].sigma/spectrum.peaks[5].sigma
#     ratio.name = '1435/1420'
#     if spectrum.peaks[6].error or spectrum.peaks[5].error:
#         ratio.error = True
#     else:
#         ratio.error = False
#     spectrum.peaks.append(ratio)

# plot_peak_areas(particle, save = False)


#%%

'''
Run to here and then load in spyder save data
'''


#%%%

'''
Run data processing
'''


#%% New spectral calibration using neon calibration and comparing to calibrated spectra from cal_h5

## Truncation for all spectra
truncate_range = [250, 2300]

## Get ref (measured) neon peaks
neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
neon_ref.normalise()
neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.02)]
neon_ref_peaks = neon_ref_peaks[0:-3]
neon_ref_peaks_y = neon_ref.y_norm[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.02)]
neon_ref_peaks_y = neon_ref_peaks_y[0:-3]
neon_ref.__dict__.pop('dset')

## Get literature neon peaks
neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
neon_wls = neon_wls[14:]

'''Make peak matching more robust below'''
# Cut detected peaks that are too close together
delete_list = []
for i, peak in enumerate(neon_ref_peaks):
    if i < len(neon_ref_peaks)-1:        
        if neon_ref_peaks[i+1] - neon_ref_peaks[i] < 2:
            x = np.argmin((neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i])], neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i+1])]))
            delete_list.append(x+i)
neon_ref_peaks = np.delete(neon_ref_peaks, delete_list)    


## Assert same number of ref and lit neon peaks
assert len(neon_ref_peaks) == len(neon_wls)   

## Plot raw neon ref, raw neon ref peaks, neon lit peak positions
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_title('Neon Lamp Reference')
ax.set_xlabel('Wavelengths (nm)')
ax.set_ylabel('Counts')
ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
ax.scatter(neon_ref_peaks, neon_ref_peaks_y, color = 'red', zorder = 10, label = 'neon meas peaks raw', s = 100, marker = 'x')
ax.scatter(neon_wls, neon_ref_peaks_y, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks', s = 100, marker = 'x')
ax.set_xlim(630,770)
ax.legend()


## Fit neon ref peaks and neon lit peaks
popt, pcov = curve_fit(f = linear, 
                    xdata = neon_ref_peaks,
                    ydata = neon_wls)

cal_matrix = popt

neon_ref_cal = linear(neon_ref.x, *cal_matrix)


# Plot all sorts of stuff to check calibration

## Plot neon ref peaks v neon lit peaks and fitted calibration curve
plt.figure(figsize=(12,9), dpi=300)
plt.scatter(neon_wls, neon_ref_peaks, s = 100)
plt.plot(neon_wls, linear(neon_ref_peaks, *cal_matrix), '-', color = 'orange')
plt.xlabel('Neon wavelengths - Literature')
plt.ylabel('Neon wavelengths - Measured')
plt.title('Neon Calibration Curve')
# plt.figtext(0.5,0.3,'R$^{2}$: ' + str(R_sq))
plt.tight_layout()
plt.show()  

## Plot calibrated neon ref spectrum, astrosurf neon peaks, and thorlabs neon peaks
thorlabs_neon = cal_h5['ThorLabs Neon Spectrum']
thorlabs_neon = SERS.SERS_Spectrum(thorlabs_neon.attrs['x'], thorlabs_neon.attrs['y'])
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelengths (nm)')
ax.set_ylabel('Counts')
ax.set_title('Neon Lamp Calibration')
ax.plot(neon_ref_cal, neon_ref.y_norm, color = 'orange', label = 'neon meas fit')
ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
ax.scatter(neon_wls, neon_ref_peaks_y, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks', s = 100, marker = 'x')
ax.scatter(thorlabs_neon.x, thorlabs_neon.y, color = 'blue', label = 'thorlabs neon peaks', zorder = 10, s = 100, marker = 'x')
ax.vlines(thorlabs_neon.x, ymin = 0, ymax = thorlabs_neon.y, color = 'blue', alpha = 0.5, linewidth = 5)
ax.set_xlim(630,770)
ax.legend()

## Plot calibrated ref BPT v. BPT from cal_h5
### Process ref bpt
''' slight manual adjustment required'''
cal_matrix[1] += 1.4
bpt_ref = my_h5['ref_meas_0']['BPT_633nm_0']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)
bpt_ref.x = linear(bpt_ref.x, *cal_matrix)
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 632.8)
bpt_ref.truncate(truncate_range[0], truncate_range[1])
bpt_ref.normalise()
bpt_ref.__dict__.pop('dset')
### Process cal bpt 
bpt_cal = cal_h5['BPT NPoM 633nm SERS Calibrated']
bpt_cal = SERS.SERS_Spectrum(x = bpt_cal.attrs['calibrated_wn'], y = bpt_cal.attrs['y'])
bpt_cal.normalise()
### Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_title('Neon Lamp Calibration applied to BPT')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Counts')
ax.plot(bpt_ref.x, bpt_ref.y_norm, color = 'blue', alpha = 0.7, label = 'BPT meas calibrated')
ax.plot(bpt_cal.x, bpt_cal.y_norm, color = 'black', alpha = 0.7, label = 'BPT lit', zorder = 0)
# ax.set_xlim(950, 1650)
# ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas')
ax.legend()

## Plot calibrated ref Co-TAPP-SMe v. from cal_h5
### Process ref Co-TAPP-SMe
spectrum = my_h5['ref_meas_0']['Co-TAPP-SMe_OCP_1s_1uW_x60_1']
spectrum = SERS.SERS_Timescan(spectrum)
spectrum.x = linear(spectrum.x, *cal_matrix)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.chunk(60)
spectrum.normalise()
### Process cal Co-TAPP-SMe
cotapp_cal = cal_h5['Co-TAPP-SMe MLAgg 633nm SERS Calibrated']
cotapp_cal = SERS.SERS_Spectrum(x = cotapp_cal.attrs['x'], y = cotapp_cal.attrs['y'])
cotapp_cal.normalise()
### Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_title('Neon Lamp Calibration applied to Co-TAPP-SMe')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Counts')
ax.plot(spectrum.x, spectrum.Y_norm[0], label = 'Measured & Calibrated Co-TAPP-SMe')
ax.plot(cotapp_cal.x, cotapp_cal.y_norm, label = 'Co-TAPP-SMe Cal')
ax.legend()
# ax.set_xlim(950, 1650)


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas_0']['white_ref_x5_0']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = linear(white_ref.x, *cal_matrix)
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)

## Get white bkg (counts in notch region)
notch_range = [160, 180]
notch = SERS.SERS_Spectrum(white_ref.x, white_ref.y, title = 'Notch')
notch.truncate(notch_range[0], notch_range[1])
notch_cts = notch.y.mean()
notch.plot(title = 'White Scatter Notch')

## Truncate out notch (same as BPT ref), assign wn_cal
white_ref.truncate(start_x = truncate_range[0], end_x = truncate_range[1])


## Convert back to wl for efficiency calibration
white_ref.x = spt.wn_to_wl(white_ref.x, 632.8)


# Calculate R_setup

R_setup = cal.white_scatter_calibration(wl = white_ref.x,
                                    white_scatter = white_ref.y,
                                    white_bkg = notch_cts,
                                    plot = True,
                                    start_notch = None,
                                    end_notch = None,
                                    bpt_ref = bpt_ref)

''' 
Still issue with 'white background' of calculating R_setup
Right now, choosing white background ~400 (near notch counts) causes R_setup to be very low at long wavelengths (>900nm)
This causes very large background past 1560cm-1 BPT peak
Using a white_bkg of -100000 flattens it out...
'''    
    

#%% 633nm powerswitch dark counts

particle = my_h5['ref_meas_1']
dark_counts = particle['633_glass_1s_1uW_0']
dark_counts = SERS.SERS_Spectrum(dark_counts)
dark_counts.__dict__.pop('dset')
dark_counts.x = linear(dark_counts.x, *cal_matrix)
dark_counts.x = spt.wl_to_wn(dark_counts.x, 632.8)
dark_counts.truncate(truncate_range[0], truncate_range[1])
dark_counts.y_smooth = spt.butter_lowpass_filt_filt(dark_counts.y, cutoff = 1000, fs = 60000)
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts')
ax.plot(dark_counts.x, dark_counts.y, color = 'black')
ax.plot(dark_counts.x, dark_counts.y_smooth, color = 'red')

# Plot dark subtracted as test

particle = my_h5['ref_meas_0']

spectrum = particle['Co-TAPP-SMe_OCP_1s_1uW_x60_1']
spectrum = SERS.SERS_Timescan(spectrum)
spectrum.x = linear(spectrum.x, *cal_matrix)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.truncate(truncate_range[0], truncate_range[1])
spectrum.calibrate_intensity(R_setup = R_setup,
                             dark_counts = dark_counts.y_smooth)

particle = Particle()
particle.timescan = spectrum
particle.chunk_size = 30
spectrum.chunk(particle.chunk_size)
plot_timescan(particle)


#%% Testing S/N from chunking

particle = my_h5['ParticleScannerScan_1']['Particle_10']
timescan = particle['Co-TAPP-SMe_633nm_SERS_CA_0']
timescan = SERS.SERS_Timescan(timescan)
timescan.x = linear(timescan.x, *cal_matrix)
timescan.x = spt.wl_to_wn(timescan.x, 632.8)
timescan.truncate(truncate_range[0], truncate_range[1])
# timescan.calibrate_intensity(R_setup = 1,
#                               dark_counts = dark_counts.y_smooth)

for i, spectrum in enumerate(timescan.Y):
    timescan.Y[i] = spectrum- dark_counts.y_smooth



#%% Testing background subtraction & cosmic ray removal

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')

spectrum = SERS.SERS_Spectrum(x = particle.timescan.x, y = particle.timescan.Y[1])
    
spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

## Baseline
spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
spectrum.y_baselined = spectrum.y - spectrum.baseline
spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
## Plot raw, baseline, baseline subtracted
offset = 50000
spectrum.plot(ax = ax, plot_y = (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), linewidth = 1, color = 'blue', alpha = 1, label = 'Background subtracted', zorder = 2)
spectrum.plot(ax = ax, plot_y = spectrum.baseline- spectrum.y.min() + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'darkred', label = 'Background', linewidth = 1)    
spectrum.plot(ax = ax, plot_y = spectrum.y_cosmic + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'orange', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
fig.suptitle(particle.name)    
ax.legend()
# ax.set_xlim(1150, 1240)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)
    

#%% Test processing & plotting

particle = my_h5['ParticleScannerScan_1']['Particle_10']
spectrum = particle['Co-TAPP-SMe_633nm_SERS_CA_0']
ca = particle['Co-TAPP-SMe_CA_Switch_x5_0']
particle = Particle()
particle.name = 'test particle'
spectrum = SERS.SERS_Timescan(spectrum)
spectrum.x = linear(spectrum.x, *cal_matrix)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.truncate(truncate_range[0], truncate_range[1])
spectrum.calibrate_intensity(R_setup = R_setup,
                             dark_counts = dark_counts.y_smooth)

particle.timescan = spectrum
particle.timescan.chunk_size = 20
spectrum.chunk(particle.timescan.chunk_size)

ca_y = np.array(ca)[1]
ca_v = ca.attrs['Levels_v (V)']
ca = SERS.SERS_Spectrum(x = ca.attrs['Time (s)'], y = ca_y*1e6)
ca.v = ca_v
particle.ca = ca

for i, spectrum in enumerate(particle.timescan.Y):
      spectrum = spt.remove_cosmic_rays(spectrum, threshold = 9)
      spectrum_baseline = spt.baseline_als(spectrum, 1e3, 1e-2, niter = 10)
      spectrum_baselined = spectrum - spectrum_baseline
      spectrum = spectrum_baselined
      particle.timescan.Y[i] = spectrum

plot_timescan_ca(particle)


#%% Get all particles to analyze into Particle class with h5 locations and in a list

particles = []

scan_list = ['ParticleScannerScan_1']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)
           
            
    # Loop over particles in particle scan
    
    for particle in particle_list[0:27]:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'MLAgg_' + str(particle_scan) + '_' + particle
        this_particle.scan = str(particle_scan)
        this_particle.particle_number = str(particle)
        # this_particle.h5_address = my_h5[particle_scan][particle]
        particles.append(this_particle)
        

#%% Loop over all particles and process timescan & ca

print('\nProcessing spectra...')
for particle in tqdm(particles, leave = True):
    
    process_timescan(particle, chunk_size = 60)
    process_ca(particle)
    

#%% Loop over all particles and fit

print('\nPeak Fitting...')

for particle in tqdm(particles, leave = True):
    
    ## Clear previous fitted peaks
    for i, spectrum in enumerate(particle.timescan.Y):
        particle.timescan.peaks[i] = []
        
    ## Fit peaks
    fit_gaussian_timescan(particle = particle, fit_range = [390, 435], smooth_first = True, plot = False) ## grey
    fit_gaussian_timescan(particle = particle, fit_range = [990, 1040], smooth_first = False, plot = False) ## purple
    fit_gaussian_timescan(particle = particle, fit_range = [1170, 1250], smooth_first = True, plot = False) ## red
    fit_gaussian3_timescan_1300(particle = particle, fit_range = [1300, 1390], smooth_first = True, plot = False, save = False) ## brown/green/blue
    fit_gaussian3_timescan_1500(particle = particle, fit_range = [1500, 1600], smooth_first = True, plot = False, save = False) ## pink/yellow/cyan
    fit_gaussian_timescan(particle = particle, fit_range = [1610, 1650], smooth_first = True,  plot = False) ## orange   

    ## 1435/1425 ratio peak    
    # for i in range(0, len(particle.timescan.Y)):
    #     peaks = particle.timescan.peaks[i]
    #     try:
    #         baseline = peaks[6].baseline/peaks[5].baseline
    #     except:
    #         baseline = 0
    #     ratio = Peak(height = peaks[6].height/peaks[5].height,
    #                  mu = None,
    #                  width = peaks[6].width/peaks[5].width,
    #                  baseline = baseline)
    #     ratio.area = peaks[6].area/peaks[5].area
    #     ratio.sigma = peaks[6].sigma/peaks[5].sigma
    #     ratio.name = '1435/1420'
    #     if peaks[6].error or peaks[5].error:
    #         ratio.error = True
    #     else:
    #         ratio.error = False
    #     peaks.append(ratio)


# Report number of failed fits

count = 0
scans = 0
for particle in particles:
    for i in range(0, len(particle.timescan.Y)):
        scans += 1
        for peak in particle.timescan.peaks[i]:
            if peak.error: count += 1
print('\nFit Errors (%): ')
print(100*count/ (len(particles) * scans * len(particle.timescan.peaks[0])))
    

#%% Making average particles & calculating timescan

from copy import deepcopy

avg_particles = []
voltages = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]

# Loop over each voltage - one avg particle per voltage
for i, voltage in enumerate(voltages):
    ## Set up avg particle
    avg_particle = Particle()
    avg_particle.voltage = voltage
    avg_particle.name = 'MLAgg_Avg_' + str((voltage)) + 'V'
    
    ## Set up avg timescan
    keys = list(my_h5[particles[i*3].scan][particles[i*3].particle_number].keys())
    keys = natsort.natsorted(keys)
    for key in keys:
        if 'SERS' in key:
            timescan = my_h5[particles[i*3].scan][particles[i*3].particle_number][key]
    timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'] * particles[i*3].timescan.chunk_size)
    timescan.x = linear(timescan.x, *cal_matrix)
    timescan.x = spt.wl_to_wn(timescan.x, 632.8)
    timescan.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    timescan.__dict__.pop('dset')
    avg_particle.timescan = timescan
    
    avg_particle.timescan.Y = np.zeros(particles[i*3].timescan.Y.shape)
    avg_particle.timescan.Baseline = np.zeros(particles[i*3].timescan.Baseline.shape)
    # avg_particle.timescan.BaselineSum = np.zeros(particles[0].timescan.BaselineSum.shape)
    
    ## Add y-values to avg timescan
    counter = 0
    for particle in particles:
        if particle.ca.v[1] == avg_particle.voltage:
            counter += 1
            avg_particle.timescan.Y += particle.timescan.Y
            avg_particle.timescan.Baseline += particle.timescan.Baseline
            # avg_particle.timescan.BaselineSum += particle.timescan.BaselineSum
            avg_particle.timescan.chunk_size = particle.timescan.chunk_size          
            
    ## Divide
    avg_particle.timescan.Y = avg_particle.timescan.Y/counter
    avg_particle.timescan.Baseline = avg_particle.timescan.Baseline/counter
    # avg_particle.timescan.BaselineSum = avg_particle.timescan.BaselineSum/counter
    
    
    avg_particle.timescan.name = 'MLAgg Avg 633 nm Timescan'                
    avg_particles.append(avg_particle)
 
                      
#%% Get peak data for average particles

from copy import deepcopy

print('\nAveraging...')

for avg_particle in tqdm(avg_particles, leave = True):
    
    timescan = avg_particle.timescan
    
    ## Clear previous fitted peaks
    timescan.peaks = []
    for i, spectrum in enumerate(timescan.Y):
        timescan.peaks.append([])
    
    
    ## Set up peaks list
    for i, spectrum in enumerate(timescan.Y):
        
        timescan.BaselineSum = []
        timescan.BaselineSumError = []
        # spectrum.baseline_sum_recovery = []
        
        timescan.peaks[i] = deepcopy(particles[0].timescan.peaks[0])
        
        for peak in timescan.peaks[i]:
            peak.area = []
            peak.baseline = []
            peak.error = False
            peak.height = []
            peak.mu = []
            peak.sigma = []
            peak.width = []
    
    ## Add peaks
    counter = 0
    for particle in particles:
        if avg_particle.voltage == particle.ca.v[1]:
            counter += 1
            
            avg_particle.ca = particle.ca
            
            for i, spectrum in enumerate(timescan.Y):
                
                timescan.BaselineSum.append(particle.timescan.BaselineSum[i])
                timescan.BaselineSumError.append(0)
                
                for j, peak in enumerate(timescan.peaks[i]):
                    this_peak = deepcopy(particle.timescan.peaks[i][j])
                    peak.area.append(this_peak.area)
                    peak.baseline.append(this_peak.baseline)
                    peak.height.append(this_peak.height)
                    if this_peak.mu is not None:
                        peak.mu.append(this_peak.mu)
                    peak.sigma.append(this_peak.sigma)
                    peak.width.append(this_peak.width)

                    
    ## Calculate mean and error    
    for i, spectrum in enumerate(timescan.Y):
        
        timescan.BaselineSumError[i] = np.std(timescan.BaselineSum[i])/(np.size(timescan.BaselineSum[i]) ** 0.5)
        timescan.BaselineSum[i] = np.mean(timescan.BaselineSum[i])
        
        
        for peak in timescan.peaks[i]:
            
            peak.area_error = np.std(peak.area)/(np.size(peak.area) ** 0.5)
            peak.area = np.mean(peak.area)
            
            peak.baseline_error = np.std(peak.baseline)/(np.size(peak.baseline) ** 0.5)
            peak.baseline = np.mean(peak.baseline)
            
            peak.height_error = np.std(peak.height)/(np.size(peak.height) ** 0.5)
            peak.height = np.mean(peak.height)
            
            peak.mu_error = np.std(peak.mu)/(np.size(peak.mu) ** 0.5)
            peak.mu = np.mean(peak.mu)
            
            peak.sigma_error = np.std(peak.sigma)/(np.size(peak.sigma) ** 0.5)
            peak.sigma = np.mean(peak.sigma)
            
            peak.width_error = np.std(peak.width)/(np.size(peak.width) ** 0.5)
            peak.width = np.mean(peak.width)
            

#%%

'''
Run plotting & saving
Can import spyder data and run plotting from here 
'''


#%% Loop over all particles and plot

print('\nPlotting...')
for particle in tqdm(particles, leave = True):
    
    plot_timescan_ca(particle, save = False, save_to_h5 = True)
    plot_timescan_background_ca(particle, save = False, save_to_h5 = True)
    plot_peak_areas(particle, save = False, save_to_h5 = True)
    plot_peak_positions(particle, save = False, save_to_h5 = True)


#%% Loop over avg particles and plot

for particle in tqdm(avg_particles, leave = True):
    
    plot_timescan_ca(particle, save = False, save_to_h5 = True)
    plot_peak_areas(particle, save = False, save_to_h5 = True)
    plot_peak_positions(particle, save = False, save_to_h5 = True)
    plot_timescan_background_ca(particle, save = False, save_to_h5 = True)
    

#%% Loop over all particles and save data to h5

for particle in particles:
    save_ca(particle)
    save_timescan(particle)
    save_peaks(particle)
    
for particle in avg_particles:
    save_ca(particle)
    save_timescan(particle)
    save_peaks(particle)