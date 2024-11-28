# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:30:15 2024

@author: il322

Plotter for TiOx NPoM 633 nm SERS and control NPoMs
Organized so that can save 'Particles' list to spydata
Added functionality to save timescan, ca, and plots to spydata and h5

Copied from 2024-08-06_633nm_Powerswitching_Recovery_in_Air_Ploter.py

Data: 
    2024-07-26_TiO_NPoM_DF_633_Powerseries_Track.h5
    2024-08-02_Toluene_KCl_60nm_NPoM_633nm_Powerseries_DF_Track.h5

Samples:
    2024-07-25_TiO_60nm_NPoM
    2024-08-01_KCl_60nm_NPoM
    2024-08-01_Toluene_KCl_60nm_NPoM

"""

from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image
from pybaselines.polynomial import modpoly
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

## Raw data h5 Files
tiox_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-26_TiO_NPoM_DF_633_Powerseries_Track.h5")
control_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-08-02_Toluene_KCl_60nm_NPoM_633nm_Powerseries_DF_Track.h5")

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")


#%% Plotting functions


def plot_timescan(particle, save = False, save_to_h5 = False):
    
    timescan = particle.timescan
    title = timescan.name[timescan.name.find('Power'):]
    ## Plot timescan
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0, len(timescan.Y) * timescan.cycle_time, timescan.cycle_time)
    v_min = timescan.Y.min()
    v_max = np.percentile(timescan.Y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('Time (s)', fontsize = 'large')
    ax.set_yticks(t_plot, labels = np.round(t_plot * particle.chunk_size, 1), fontsize = 'large', )
    # ax.set_xlim(300,1700)
    ax.set_title(str(particle.name)  + '\n' + title + '\n' + str(np.round(timescan.laser_power * 1000, 0)) + ' $\mu$W', fontsize = 'x-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot + (timescan.cycle_time/2), timescan.Y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'SERS Intensity (cts/mW/s)', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + title + '.svg', format = 'svg')
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
        plt.close(fig)
        

def plot_nanocavity_powerseries(particle, save = False, save_to_h5 = False):
    
    powerseries = particle.powerseries
    
    fig, (ax) = plt.subplots(1, 1, figsize=[16,14])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('SERS Intensity (cts/mW/s)', fontsize = 'large')
    title = 'Nanocavity 633 nm SERS Powerseries'
    ax.set_title(str(particle.name)  + '\n' + title, fontsize = 'x-large', pad = 10)
    
    ## Set up colorbar  
    my_cmap = plt.get_cmap('plasma')
    cmap = my_cmap
    norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, fontsize = 'large', labelpad = 40)
    cbar_ax.yaxis.set_label_position("right")
    # cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
    cbar_ax.tick_params(which = 'major', length = 10, width = 1.5)
    cbar_ax.tick_params(which = 'minor', length = 6, width = 1)

    ## Get offset
    powerseries_y = []
    for i, timescan in enumerate(powerseries):
        powerseries_y.append(timescan.min_spectrum)
    offset = np.percentile(np.array(powerseries_y), 95)/2
    
    ## Plot nanocavity (min) spectrum
    for i, timescan in enumerate(powerseries):
        line = ax.plot(timescan.x, timescan.min_spectrum + (i * offset), color = cmap(norm(timescan.laser_power)))
        fill = ax.fill_between(timescan.x,
                        (timescan.min_spectrum + (i * offset) - (timescan.sem_spectrum * 10**0.5)), 
                        (timescan.min_spectrum + (i * offset) + (timescan.sem_spectrum * 10**0.5)), 
                        color = cmap(norm(timescan.laser_power)), 
                        alpha = 0.5)
        
        # ax.plot(timescan.x, (timescan.sem_spectrum.y * 10**0.5) + (i*3000), color = cmap(norm(timescan.laser_power)))
    
    ## Legend
    line[0].set_color('black')
    fill.set_color('black')
    ax.legend([line[0], fill], ['Nanocavity SERS', '$\sigma$ SERS'], fontsize = 'large')

    ## Axis limits and ticks
    # ax.set_xlim(300,1700)
    ax.set_ylim(-offset, offset * (len(powerseries_y) + 3))
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 10000), minor = True)
    ax.set_xticks(np.arange(200, ax.get_xlim()[1], 100), minor = True)
    ax.tick_params(which = 'major', length = 7, width = 1.5)
    ax.tick_params(which = 'minor', length = 5, width = 1)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + title + '.svg', format = 'svg')
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


def plot_powerswitch_recovery(particle, save = False, save_to_h5 = False):
    
    powerseries = particle.powerseries
    
    ## Add dark times into powerseries (for plotting)
    if particle.dark_time > 0:
        spectrum = SERS.SERS_Spectrum(x = powerseries[0].x, y = np.zeros(len(powerseries[0].y)))
        spectrum.y_baselined = spectrum.y        
        powerseries = np.insert(powerseries, [10], spectrum)
        powerseries = np.insert(powerseries, [10], spectrum)

    ## Get all specrta into single 2D array for timescan
    powerseries_y = np.zeros((len(powerseries), len(powerseries[0].y)))
    for i,spectrum in enumerate(powerseries):
        powerseries_y[i] = spectrum.y_baselined
    powerseries_y = np.array(powerseries_y)

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = powerseries_y.min()
    v_max = np.percentile(powerseries_y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'x-large')
    ax.tick_params(axis='both', which='major', labelsize='large')
    # ax.set_xlim(450,2100)
    if particle.dark_time > 0:
        ax.text(x=750,y=10.25,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + ' s', color = 'white', size='xx-large')
    ax.set_title('633 nm SERS Powerswitch Recovery - 1 / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'xx-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax = ax)
    clb.set_label(label = 'SERS Intensity', size = 'x-large', rotation = 270, labelpad=30)
    clb.ax.tick_params(labelsize='large')
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '633nm SERS Powerswitch Recovery' + '.svg', format = 'svg')
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
        group.create_dataset(name = '633nm SERS Powerswitch Recovery_jpg_%d', data = fig_array)


def plot_powerswitch_recovery_pl(particle, save = False, save_to_h5 = False):
    
    powerseries = particle.powerseries
    
    ## Add dark times into powerseries (for plotting)
    if particle.dark_time > 0:
        spectrum = SERS.SERS_Spectrum(x = powerseries[0].x, y = np.zeros(len(powerseries[0].y)))
        spectrum.pl = spectrum.y        
        powerseries = np.insert(powerseries, [10], spectrum)
        powerseries = np.insert(powerseries, [10], spectrum)

    ## Get all specrta into single 2D array for timescan
    powerseries_y = np.zeros((len(powerseries), len(powerseries[0].y)))
    for i,spectrum in enumerate(powerseries):
        powerseries_y[i] = spectrum.pl
    powerseries_y = np.array(powerseries_y)

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = powerseries_y.min()
    v_max = np.percentile(powerseries_y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'x-large')
    ax.tick_params(axis='both', which='major', labelsize='large')
    # ax.set_xlim(450,2100)
    if particle.dark_time > 0:
        ax.text(x=750,y=10.25,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + ' s', color = 'white', size='xx-large')
    ax.set_title('633 nm PL Powerswitch Recovery - 1 / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'xx-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax = ax)
    clb.set_label(label = 'Intensity', size = 'x-large', rotation = 270, labelpad=30)
    clb.ax.tick_params(labelsize='large')
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '633nm PL Powerswitch Recovery' + '.svg', format = 'svg')
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
        group.create_dataset(name = '633nm PL Powerswitch Recovery_jpg_%d', data = fig_array)
                
    
my_cmap = plt.get_cmap('inferno')       
    

def plot_powerseries_peak_areas(particle, save = False, save_to_h5 = False):
    
    powerseries = particle.powerseries
    
    fig, (ax) = plt.subplots(1, 1, figsize=[14,10])
    ax.set_xlabel('Scan No.', fontsize = 'large')
    ax.set_ylabel('Peak Area (cts/mW/s)', fontsize = 'large')
    title = 'TiOx SERS Peak Area v. Powerseries'
    ax.set_title(str(particle.name)  + '\n' + title, fontsize = 'x-large', pad = 10)

    ## Set up colorbar  
    my_cmap = plt.get_cmap('plasma')
    cmap = my_cmap
    norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, fontsize = 'large', labelpad = 40)
    cbar_ax.yaxis.set_label_position("right")
    # cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
    cbar_ax.tick_params(which = 'major', length = 10, width = 1.5)
    cbar_ax.tick_params(which = 'minor', length = 6, width = 1)
    
    for i, spectrum in enumerate(powerseries):
        peak = spectrum.peaks[0]
        if peak.error == True:
            continue
        else:
            ax.scatter(i, peak.area, color = cmap(norm(spectrum.laser_power)), s = 200, zorder = 2)
            try:
                ax.errorbar(x = i, y = peak.area, yerr = peak.area_sem, color = cmap(norm(spectrum.laser_power)), capsize = 7, elinewidth = 3, capthick = 3, zorder = 1)
            except: pass
        
    ## Axes limits and tick marks
    ax.set_xticks(np.arange(0, len(powerseries), 2), np.arange(0, len(powerseries), 2))
    ax.set_xlim(-1, len(powerseries))
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + 'Peak Area' + '.svg', format = 'svg')
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
    

def plot_powerseries_peak_positions(particle, save = False, save_to_h5 = False):
    
    powerseries = particle.powerseries
    
    fig, (ax) = plt.subplots(1, 1, figsize=[14,10])
    ax.set_xlabel('Scan No.', fontsize = 'large')
    ax.set_ylabel('Peak Position (cm$^{-1}$)', fontsize = 'large')
    title = 'TiOx SERS Peak Position v. Powerseries'
    ax.set_title(str(particle.name)  + '\n' + title, fontsize = 'x-large', pad = 10)

    ## Set up colorbar  
    my_cmap = plt.get_cmap('plasma')
    cmap = my_cmap
    norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, fontsize = 'large', labelpad = 40)
    cbar_ax.yaxis.set_label_position("right")
    # cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
    cbar_ax.tick_params(which = 'major', length = 10, width = 1.5)
    cbar_ax.tick_params(which = 'minor', length = 6, width = 1)
    
    for i, spectrum in enumerate(powerseries):
        peak = spectrum.peaks[0]
        if peak.error == True:
            continue
        else:
            ax.scatter(i, peak.mu, color = cmap(norm(spectrum.laser_power)), s = 200, zorder = 2)
            try:
                ax.errorbar(x = i, y = peak.mu, yerr = peak.mu_sem, color = cmap(norm(spectrum.laser_power)), capsize = 7, elinewidth = 3, capthick = 3, zorder = 1)
            except: pass
        
    ## Axes limits and tick marks
    ax.set_xticks(np.arange(0, len(powerseries), 2), np.arange(0, len(powerseries), 2))
    ax.set_xlim(-1, len(powerseries))
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + 'Peak Position' + '.svg', format = 'svg')
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
   

def plot_peak_areas_powerswitch_recovery(particle, save = False, save_to_h5 = False):
       
    powerseries = particle.powerseries
    
    scans = np.arange(len(powerseries))
        
    colors = ['grey', 'purple', 'violet', 'saddlebrown', 'red', 'tomato', 'salmon', 'green', 'lime', 'mediumspringgreen', 'gold', 'blue', 'royalblue', 'cyan', 'darkorange', 'black']
    
    fig, axes = plt.subplots(len(powerseries[0].peaks) + 1, 1, figsize=[14,32], sharex = True)
    fig.suptitle('633 nm SERS Peak Area -  Powerswitch - 1 / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Scan No.', size = 'x-large')
        
    
    # Loop over each peak and axis
    
    peaks = powerseries[0].peaks
    for i, peak in enumerate(peaks):
        
        ## Peak area ratios to plot for y
        y = [] 
        ax = axes[i]    
        
        ## Loop over spectra and data points            
        for spectrum in powerseries:
            if spectrum.peaks[i].error:
                y.append(nan)
            else:
                y.append(100*(spectrum.peaks[i].area - powerseries[0].peaks[i].area)/powerseries[0].peaks[i].area)          
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = scans, y = y)
                              
        ## Plot peaks
        color = colors[i]
        if 'Avg' in particle.name:
            alpha = 0.5
        else:
            alpha = 1
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 4, alpha = alpha)
        ax.scatter(peak_spec.x[::2], peak_spec.y[::2], s = 200, label = '1 $\mu$W', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
        ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2], s = 200, label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)

        ## Errorbars, if avg particle
        try:                
            y_error = []
            for j, spectrum in enumerate(powerseries):
                if j == 0:
                    this_error = 0
                else:
                    ## Error = propagated error (for % change) of peak areas std
                    subtraction = spectrum.peaks[i].area - powerseries[0].peaks[i].area
                    this_error_subtraction = (spectrum.peaks[i].area_std**2 + powerseries[0].peaks[i].area_std**2)**0.5
                    this_error = np.abs(peak_spec.y[j]) * ( (this_error_subtraction/subtraction)**2 + (powerseries[0].peaks[i].area_std/powerseries[0].peaks[i].area)**2 )**0.5
                y_error.append(this_error)   
            y_error = np.array(y_error)
            ax.errorbar(peak_spec.x, peak_spec.y , yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
        except:
            y_error = None
        
        ## Additions
        ylim = ax.get_ylim() 
        if particle.dark_time > 0:
            ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.7, linewidth = 15)
        ax.set_ylim(ylim)
        ax.legend(loc = 'upper right', facecolor = 'white', framealpha = 0.5)   
           
                      
    # Plot background sum
    
    y = []
    for spectrum in powerseries:
        y.append(100*(spectrum.pl_sum - powerseries[0].pl_sum)/powerseries[0].pl_sum)
    y = np.array(y)
    background_spec = SERS.SERS_Spectrum(x = scans, y = y)                             
    color = colors[len(colors)-1]
    ax = axes[len(axes)-1]
    ax.plot(background_spec.x, background_spec.y, label = 'PL Sum', color = color, alpha = 0.5, zorder = 4, linewidth = 4)
    ax.scatter(background_spec.x[::2], background_spec.y[::2], s = 200, label = '1 $\mu$W', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
    ax.scatter(background_spec.x[1::2], background_spec.y[1::2], s = 200, label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)
    ax.legend(loc = 'upper right')   
    ylim = ax.get_ylim() 
    if particle.dark_time > 0:
        ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.7, linewidth = 15)
    ax.set_ylim(ylim)
    ax.legend(loc = 'upper right', facecolor = 'white', framealpha = 0.5)   

    ## PL Sum error bars, if avg_particle
    try:                
        y_error = []
        for j, spectrum in enumerate(powerseries):
            if j == 0:
                this_error = 0
            else:
                ## Error = propagated error (for % change) of peak areas std
                subtraction = spectrum.pl_sum - powerseries[0].pl_sum
                this_error_subtraction = (spectrum.pl_sum_std**2 + powerseries[0].pl_sum_std**2)**0.5
                this_error = np.abs(background_spec.y[j]) * ( (this_error_subtraction/subtraction)**2 + (powerseries[0].pl_sum_std/powerseries[0].pl_sum)**2 )**0.5
            y_error.append(this_error)   
        y_error = np.array(y_error)
        ax.errorbar(background_spec.x, background_spec.y , yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
    except:
        y_error = None
    
    ## Labels
    axes[int(len(axes)/2)].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'xx-large')
    axes[len(axes)-1].set_ylabel('I$_{PL}$ (%$\Delta$)', size = 'xx-large')
    ax.set_xticks(scans[::2])
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Powerswitch Recovery SERS Peak Area' + '.svg', format = 'svg')
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
 
    
def plot_peak_positions_powerswitch_recovery(particle, save = False, save_to_h5 = False):
       
    powerseries = particle.powerseries
    
    scans = np.arange(len(powerseries))
        
    colors = ['grey', 'purple', 'violet', 'saddlebrown', 'red', 'tomato', 'salmon', 'green', 'lime', 'mediumspringgreen', 'gold', 'blue', 'royalblue', 'cyan', 'darkorange', 'black']
    
    fig, axes = plt.subplots(len(powerseries[0].peaks), 1, figsize=[14,32], sharex = True)
    fig.suptitle('633 nm SERS Peak Positions -  Powerswitch - 1 / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Scan No.', size = 'x-large')
        
    
    # Loop over peaks and axes
    
    peaks = powerseries[0].peaks
    for i, peak in enumerate(peaks):
        
        ## Peak area ratios to plot for y
        y = [] 
        ax = axes[i]    
        
        ## Loop over scans and datapoints            
        for spectrum in powerseries:
            if spectrum.peaks[i].error:
                y.append(nan)
            else:
                y.append(spectrum.peaks[i].mu - powerseries[0].peaks[i].mu)          
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = scans, y = y)
                              
        ## Plot peaks
        color = colors[i]
        if 'Avg' in particle.name:
            alpha = 0.5
        else:
            alpha = 1
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 4, alpha = alpha)
        ax.scatter(peak_spec.x[::2], peak_spec.y[::2], s = 200, label = '1 $\mu$W', marker = 'o', facecolors = 'none', edgecolors = color, zorder = 5, linewidth = 4)
        ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2], s = 200, label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, zorder = 5, linewidth = 4)

        ## Errorbars, if avg particle
        try:                
            y_error = []
            for j, spectrum in enumerate(powerseries):
                if j == 0:
                    this_error = 0
                else:
                    ## Error = propagated error (for difference) of peak mu std
                    this_error = ( (spectrum.peaks[i].mu_std)**2 + (powerseries[0].peaks[i].mu_std)**2 )**0.5
                y_error.append(this_error)   
            y_error = np.array(y_error)
            ax.errorbar(peak_spec.x, peak_spec.y , yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
        except:
            y_error = None
        
        ## Additions
        ylim = ax.get_ylim() 
        if particle.dark_time > 0:
            ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.7, linewidth = 15)
        ax.set_ylim(ylim)
        ax.legend(loc = 'upper right', facecolor = 'white', framealpha = 0.5)   
               
    ## Labeling
    axes[int(len(axes)/2)].set_ylabel('$\Delta$ Peak Position (cm$^{-1}$)', size = 'xx-large')
    ax.set_xticks(scans[::2])
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Powerswitch Recovery SERS Peak Positions' + '.svg', format = 'svg')
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
        group.create_dataset(name = 'Peak_Positions_jpg_%d', data = fig_array)
    
    
# Test plotting

# particle = particles[0]
# plot_timescan_ca(particle, save_to_h5 = True, save = True)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
        
# plot_peak_positions_powerswitch_recovery(avg_particles[2])

# particle = particles[1]
# plot_peak_positions_powerswitch_recovery(particle)
# plot_timescan(particle, save = False)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
# plot_peak_areas(avg_particles[8], save = False)
# plot_powerswitch_recovery_pl(particle)


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
    try:
        attrs.pop('Y')
    except: pass
    try:
        attrs.pop('Y_raw')
    except: pass
    try:
        attrs.pop('y_raw')
    except: pass
    try:
        attrs.pop('rc_params')
    except: pass
    try:
        attrs.pop('peaks')
    except:
        pass
    
    ## Save Y raw timescan (Y-raw attribute is too large for h5 file attributes)
    group.create_dataset(name = timescan.name + '_raw_%d', data = timescan.Y_raw, attrs = attrs)

    ## Save Y processed timescan
    group.create_dataset(name = timescan.name + '_%d', data = timescan.Y, attrs = attrs)
  
    
def save_powerswitch_recovery(particle, overwrite = False):
    
    powerseries = particle.powerseries

    ## Get group in h5 file (or create if it doesn't exist)
    try:
        group = save_h5[str(particle.name)]
    except:
        group = save_h5.create_group(str(particle.name))

    ## Get attributes, remove troublesome attributes (too large for h5 saving)
    for spectrum in powerseries:
        try:
            spectrum.__dict__.pop('dset')
        except: pass
        attrs = deepcopy(spectrum.__dict__)
        attrs.pop('rc_params')
        attrs.pop('peaks')

    # Create group to save raw spectra (Y-raw attribute is too large for h5 file attributes)
    # group.create_dataset(name = timescan.name + '_raw_%d', data = timescan.Y_raw, attrs = attrs)

    ## Save processed powerseries
    for spectrum in powerseries:
        group.create_dataset(name = spectrum.name + '_%d', data = spectrum.y, attrs = attrs)
    
    
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


def save_peaks_powerswitch_recovery(particle):
    
    ## Get all peaks from powerseries into single 2D array
    peaks = []
    powerseries = particle.powerseries
    for spectrum in powerseries:
        peaks.append(spectrum.peaks)
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


# save_powerswitch_recovery(particles[0])
# save_peaks_powerswitch_recovery(particles[0])

#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-07-25 TiOx NPoM 633nm SERS Powerseries Analysis\\' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Function to add & process SERS timescan, DF, & CA for each particle


def process_powerswitch_recovery(particle):
    
    # Get powerseries

    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(my_h5[particle.scan][particle.particle_number][key])

    # Process

    for i, spectrum in enumerate(powerseries):

        spectrum = SERS.SERS_Spectrum(spectrum)
        spectrum.__dict__.pop('dset')

        ## Calibrate x-axis
        spectrum.x = linear(spectrum.x, *cal_matrix)
        spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
        spectrum.truncate(truncate_range[0], truncate_range[1])
        
        ## Calibrate intensity
        spectrum.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y_smooth,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        
        ## Modpoly fit for PL background    
        pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
        spectrum.pl = pl
        ### Sum of PL background from 350 to 1900 1/cm
        spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
        
        ## Cosmic ray & ALS Baseline Subtraction
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        spectrum.normalise(norm_y = spectrum.y_baselined)
        
        spectrum.peaks = []
        
        powerseries[i] = spectrum

    particle.powerseries = powerseries
    particle.dark_time = np.round(my_h5[particle.scan][particle.particle_number]['dark_time_0'], 0)


def process_df(particle, plot = False, tinder = False, save = False):
    
    ## Get z-scan and images
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    z_scans = []
    images = []
    for key in keys:
        if 'z_scan' in key:
            z_scans.append(my_h5[particle.scan][particle.particle_number][key])
        if 'image' in key:
            images.append(my_h5[particle.scan][particle.particle_number][key])

    ## Process DF
    for i, scan in enumerate(z_scans):
        
        name = str(copy(scan.name))
        
        scan = df.Z_Scan(scan, z_min = scan.attrs['neg'], z_max = scan.attrs['pos'], z_trim = [8, -8])
        scan.truncate(400, 900)   
        scan.condense_z_scan()
        spectrum = scan.df_spectrum
        scan.df_spectrum = df.DF_Spectrum(scan.x, spectrum)
        scan.df_spectrum.find_critical_wln()
        image = images[i]
        save_file = None
        if save == True:
            save_file = get_directory(particle.name) + 'DF_' + str(i) + '.svg'
        scan.df_spectrum = df.df_screening(scan, scan.df_spectrum, image, plot = plot, tinder = tinder, brightness_threshold = 3, title = particle.name + '\n' + str(i), save_file = save_file)
        z_scans[i] = scan
        images[i] = image
        
    ## Add df objects to particle
    particle.z_scans = z_scans
    particle.images = images
    particle.crit_wlns = [z_scans[0].df_spectrum.crit_wln, z_scans[1].df_spectrum.crit_wln]
    particle.is_npom = [particle.z_scans[0].df_spectrum.is_npom, particle.z_scans[0].df_spectrum.is_npom]    


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
    

def process_kinetic_powerseries(particle):
    
    # Add each timescan to powerseries

    ## Get timescan object
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(my_h5[particle.scan][particle.particle_number][key])


    # Process each timescan in powerseries

    for i, timescan in enumerate(powerseries):

        name = str(copy(timescan.name))
        timescan = SERS.SERS_Timescan(timescan)
        timescan.__dict__.pop('dset')
        timescan.name = name[name.rfind('/') + 1:]

        ## Calibrate x-axis
        timescan.x = linear(timescan.x, *cal_matrix)
        timescan.x = spt.wl_to_wn(timescan.x, 632.8)
        timescan.truncate(truncate_range[0], truncate_range[1])
        
        ## Calibrate intensity
        timescan.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y_smooth,
                                      exposure = timescan.cycle_time,
                                      laser_power = timescan.laser_power)
        
        ## Modpoly fit for PL background    
        # pl = modpoly(timescan.y, x_data = timescan.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
        # timescan.pl = pl
        ### Sum of PL background from 350 to 1900 1/cm
        # timescan.pl_sum = np.sum(pl[np.argmin(np.abs(timescan.x - 350)) : np.argmin(np.abs(timescan.x - 1900))])
        
        ## Cosmic ray & ALS Baseline Subtraction
        # timescan.y = spt.remove_cosmic_rays(timescan.y, threshold = 9)
        # timescan.baseline = spt.baseline_als(timescan.y, 1e3, 1e-2, niter = 10)
        # timescan.y_baselined = timescan.y
        # timescan.normalise()
        
        ## Extract nanocavity timescan (using min_spectrum)
        timescan_y = timescan.Y.copy()
        pixel_baseline = np.zeros(len(timescan.x))
        pixel_min = np.zeros(len(timescan.x))
        pixel_sem = np.zeros(len(timescan.x))
        for pixel in range(0,len(timescan.x)):
            pixel_baseline[pixel] = np.polyfit(timescan.t, timescan_y[:,pixel], deg = 0)
            pixel_min[pixel] = np.min(timescan_y[:,pixel])
            pixel_sem[pixel] = np.std(timescan_y[:,pixel])/(len(timescan_y[:,pixel]) ** 0.5)
        timescan.nanocavity_spectrum = pixel_baseline
        timescan.min_spectrum = pixel_min
        timescan.sem_spectrum = pixel_sem
        
        powerseries[i] = timescan
        
    particle.powerseries = powerseries
    particle.chunk_size = 1


#%% Fitting functions

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


def fit_gaussian(spectrum, fit_range, peak_name = None, R_sq_thresh = 0.9, smooth_first = False, plot = False):    
    
    ## Get region of spectrum to fit
    fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
    x = spectrum.x[fit_range_index[0]:fit_range_index[1]]   
    
    ## Smooth, truncate y region
    y = spectrum.y_baselined
    if smooth_first == True:
        y = spt.butter_lowpass_filt_filt(y, cutoff = 2000, fs = 40000)
    y = y[fit_range_index[0]:fit_range_index[1]]
    y_raw = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
    

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
    
    error = False
    
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
        error = True
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
    if R_sq < R_sq_thresh or np.isnan(R_sq) or error == True:
        # print('\nPoor Fit')
        # print(particle.name)
        # print(spectrum.name)
        # print(peak_name)
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
       
    
    # Plot
    
    if plot == True: # or error == True:
        
        fig, ax = plt.subplots(1,1,figsize=[14,9])
        fig.suptitle(particle.name, fontsize = 'large')
        ax.set_title(spectrum.name + ' - ' + this_peak.name + ' cm$^{-1}$ Peak Fit', fontsize = 'large')
        ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
        ax.set_ylabel('SERS Intensity (cts/mW/s)')
        if smooth_first:
            ax.plot(x, y, color = 'grey', label = 'Smoothed') 
        ax.plot(x, y_raw, color = 'grey', linestyle = '--', label = 'Data')
        ax.plot(x, y_fit, label = 'Fit', color = 'orange')
        ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
        ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
        ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
        if error:
            ax.text(x = mu0, y = height0, s = 'ERROR')
        ax.set_xlim(fit_range[0] - 3, fit_range[1] + 3)
        ax.set_ylim(None, y.max()*1.2)
        ax.legend()
        plt.show()
        
    return this_peak


#%% Run to here then load in spydata after processing

'''
Run up to here then load in spydata after processing
'''


'''
Running data processing below
'''


#%% Start TiOx

'''
Start
TiOx
'''

my_h5 = tiox_h5


#%% New spectral calibration using BPT calibration and comparing to calibrated spectra from cal_h5

## Truncation for all spectra
truncate_range = [250, 2300]

# ## Get ref (measured) BPT peaks
# neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
# neon_ref = SERS.SERS_Spectrum(neon_ref)
# neon_ref.normalise()
# neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
# neon_ref_peaks = neon_ref_peaks[0:-4]
# neon_ref_peaks_y = neon_ref.y_norm[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
# neon_ref_peaks_y = neon_ref_peaks_y[0:-4]
# neon_ref.__dict__.pop('dset')

# ## Get literature neon peaks
# neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
# neon_wls = neon_wls[15:]

# '''Make peak matching more robust below'''
# # Cut detected peaks that are too close together
# delete_list = []
# for i, peak in enumerate(neon_ref_peaks):
#     if i < len(neon_ref_peaks)-1:        
#         if neon_ref_peaks[i+1] - neon_ref_peaks[i] < 1:
#             x = np.argmin((neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i])], neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i+1])]))
#             delete_list.append(x+i)
# neon_ref_peaks = np.delete(neon_ref_peaks, delete_list)    
# neon_ref_peaks_y = np.delete(neon_ref_peaks_y, delete_list)  

# ## Assert same number of ref and lit neon peaks
# assert(len(neon_ref_peaks) == len(neon_wls), print(len(neon_ref_peaks)))

# ## Plot raw neon ref, raw neon ref peaks, neon lit peak positions
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_title('Neon Lamp Reference')
# ax.set_xlabel('Wavelengths (nm)')
# ax.set_ylabel('Counts')
# ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
# ax.scatter(neon_ref_peaks, neon_ref_peaks_y, color = 'red', zorder = 10, label = 'neon meas peaks raw', s = 100, marker = 'x')
# ax.scatter(neon_wls, neon_ref_peaks_y, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks', s = 100, marker = 'x')
# ax.set_xlim(630,770)
# ax.legend()


# ## Fit neon ref peaks and neon lit peaks
# popt, pcov = curve_fit(f = linear, 
#                     xdata = neon_ref_peaks,
#                     ydata = neon_wls)

# cal_matrix = popt

# neon_ref_cal = linear(neon_ref.x, *cal_matrix)


# # Plot all sorts of stuff to check calibration

# ## Plot neon ref peaks v neon lit peaks and fitted calibration curve
# plt.figure(figsize=(12,9), dpi=300)
# plt.scatter(neon_wls, neon_ref_peaks, s = 100)
# plt.plot(neon_wls, linear(neon_ref_peaks, *cal_matrix), '-', color = 'orange')
# plt.xlabel('Neon wavelengths - Literature')
# plt.ylabel('Neon wavelengths - Measured')
# plt.title('Neon Calibration Curve')
# # plt.figtext(0.5,0.3,'R$^{2}$: ' + str(R_sq))
# plt.tight_layout()
# plt.show()  

# ## Plot calibrated neon ref spectrum, astrosurf neon peaks, and thorlabs neon peaks
# thorlabs_neon = cal_h5['ThorLabs Neon Spectrum']
# thorlabs_neon = SERS.SERS_Spectrum(thorlabs_neon.attrs['x'], thorlabs_neon.attrs['y'])
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Wavelengths (nm)')
# ax.set_ylabel('Counts')
# ax.set_title('Neon Lamp Calibration')
# ax.plot(neon_ref_cal, neon_ref.y_norm, color = 'orange', label = 'neon meas fit')
# ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
# ax.scatter(neon_wls, neon_ref_peaks_y, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks', s = 100, marker = 'x')
# ax.scatter(thorlabs_neon.x, thorlabs_neon.y, color = 'blue', label = 'thorlabs neon peaks', zorder = 10, s = 100, marker = 'x')
# ax.vlines(thorlabs_neon.x, ymin = 0, ymax = thorlabs_neon.y, color = 'blue', alpha = 0.5, linewidth = 5)
# ax.set_xlim(630,770)
# ax.legend()

cal_matrix = [1, 1] # Just doing this by eye with BPT bc no neon ref

## Plot calibrated ref BPT v. BPT from cal_h5
### Process ref bpt
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
# ax.set_xlim(200, 1000)
# ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas')
ax.legend()

## Plot calibrated ref Co-TAPP-SMe v. from cal_h5
### Process ref Co-TAPP-SMe
# spectrum = my_h5['ref_meas_0']['Co-TAPP-SMe_Air_100s_1uW_0']
# spectrum = SERS.SERS_Spectrum(spectrum)
# spectrum.x = linear(spectrum.x, *cal_matrix)
# spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
# # spectrum.chunk(60)
# spectrum.normalise()
# ### Process cal Co-TAPP-SMe
# cotapp_cal = cal_h5['Co-TAPP-SMe MLAgg 633nm SERS Calibrated']
# cotapp_cal = SERS.SERS_Spectrum(x = cotapp_cal.attrs['x'], y = cotapp_cal.attrs['y'])
# cotapp_cal.normalise()
# ### Plot
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_title('Neon Lamp Calibration applied to Co-TAPP-SMe')
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('Counts')
# ax.plot(spectrum.x, spectrum.y_norm, label = 'Measured & Calibrated Co-TAPP-SMe')
# ax.plot(cotapp_cal.x, cotapp_cal.y_norm, label = 'Co-TAPP-SMe Cal')
# ax.legend()
# ax.set_xlim(950, 1650)


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas_0']['white_ref_x5_0']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = linear(white_ref.x, *cal_matrix)
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)

## Get white bkg (counts in notch region)
notch_range = [160, 175]
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

particle = my_h5['PT_lab']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []

for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 80000)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


## Plot dark counts & smoothed dark counts
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts')
for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y, color = 'black')
    ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')


# Plot dark subtracted as test

particle = my_h5['ParticleScannerScan_1']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts Subtraction Test')

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    # spectrum.calibrate_intensity(R_setup = R_setup,
    #                               dark_counts = dark_powerseries[i].y_smooth,
    #                               exposure = spectrum.cycle_time,
    #                               laser_power = spectrum.laser_power)
    powerseries[i] = spectrum
    
i = 8
    
ax.plot(powerseries[i].x, powerseries[i].y, color = 'black')
ax.plot(powerseries[i].x, powerseries[i].y - dark_powerseries[i].y_smooth, color = 'blue')
ax.plot(powerseries[i].x, dark_powerseries[i].y_smooth, color = 'red')    
# ax.plot(powerseries[10].x, powerseries[5].y, color = 'black')
# ax.plot(powerseries[10].x, powerseries[10].y - dark_powerseries[10].y_smooth, color = 'blue')
# ax.plot(powerseries[10].x, dark_powerseries[10].y_smooth, color = 'red')  
# ax.set_ylim(300, 400)

#%% Testing background subtraction & cosmic ray removal

''' Skipping for now since TiOx peak is broad, gets caught in baseline'''

particle = my_h5['ParticleScannerScan_1']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    powerseries[i] = spectrum


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
spectrum = powerseries[2]
spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

## Baseline
spectrum.baseline = spt.baseline_als(spectrum.y, 1e6, 1e-1, niter = 10)
spectrum.y_baselined = spectrum.y - spectrum.baseline
spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
## Plot raw, baseline, baseline subtracted
offset = 50000
ax.plot(spectrum.x, (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
ax.plot(spectrum.x, spectrum.y_baselined + (i*offset), linewidth = 1, color = 'blue', alpha = 1, label = 'Background subtracted', zorder = 2)
ax.plot(spectrum.x, spectrum.baseline- spectrum.y.min() + (i*offset), color = 'darkred', label = 'Background', linewidth = 1)    
ax.plot(spectrum.x, spectrum.y_cosmic + (i*offset), color = 'orange', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
title = 'Background Subtraction & Cosmic Ray Test'
fig.suptitle(title + particle.name)    
ax.legend()
# ax.set_xlim(1150, 1240)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)
    

#%% Testing iterative polynomial fit for fitting PL background (Bart's method)

''' Skipping this for now - no PL and TiOx SERS peak is very broad

## Get powerseries
particle = my_h5['ParticleScannerScan_1']['Particle_40']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    ### Modpoly fit for PL background    
    pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    powerseries[i] = spectrum

## Plot
spectrum = powerseries[0]
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
offset = 50000
spectrum.plot(ax = ax, plot_y = spectrum.y)
spectrum.plot(ax = ax, plot_y = spectrum.pl)
fig.suptitle(particle.name)    
# ax.legend()
# ax.set_xlim(350, 1900)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)    
'''


#%% Test processing & plotting

particle = my_h5['ParticleScannerScan_1']['Particle_15']


# Add each timescan to powerseries

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


# Process timescan

for i, spectrum in enumerate(powerseries):

    name = str(copy(spectrum.name))
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')

    ## Calibrate x-axis
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    
    ## Calibrate intensity
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    
    ## Modpoly fit for PL background    
    # pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    # spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    # spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    ## Cosmic ray & ALS Baseline Subtraction
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
    # spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    # spectrum.y_baselined = spectrum.y
    # spectrum.normalise()
    spectrum.name = name[name.rfind('/') + 1:]
    
    spectrum.extract_nanocavity(plot = False)
    
    powerseries[i] = spectrum
    

particle = Particle()
particle.name = 'test'
particle.powerseries = powerseries
# particle.timescan = powerseries[0]
particle.chunk_size = 1

# for timescan in powerseries:
#     particle.timescan = timescan
#     plot_timescan(particle)


#%% Test nanocavity extraction

for timescan in particle.powerseries:
    
    particle.timescan = timescan
    plot_timescan(particle)
    
    # extract_nanocavity(self = timescan, plot = False)
    timescan_y = timescan.Y.copy()
    pixel_baseline = np.zeros(len(timescan.x))
    pixel_min = np.zeros(len(timescan.x))
    pixel_sem = np.zeros(len(timescan.x))
    for pixel in range(0,len(timescan.x)):
        pixel_baseline[pixel] = np.polyfit(timescan.t, timescan_y[:,pixel], deg = 0)
        pixel_min[pixel] = np.min(timescan_y[:,pixel])
        pixel_sem[pixel] = np.std(timescan_y[:,pixel])/(len(timescan_y[:,pixel]) ** 0.5)
    timescan.nanocavity_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_baseline)
    timescan.min_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_min)
    timescan.sem_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_sem)
    
    # fig, ax = plt.subplots(1,1,figsize=[12,9])
    # ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    # ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
    # ax.plot(timescan.x, timescan.y, color = 'black', label = 'Avg')
    # ax.plot(timescan.x, timescan.nanocavity_spectrum.y, color = 'red', alpha = 1, linestyle = '--', label = 'Extracted nanocavity (0-deg polyfit)')
    # ax.plot(timescan.x, timescan.min_spectrum.y, color = 'green', alpha = 0.5, label = 'Minimum pixel')
    # ax.plot(timescan.x, timescan.sem_spectrum.y, color = 'orange', label = 'Pixel SEM')
    # ax.set_title(timescan.name + '\nNanocavity Extraction')
    # ax.legend()
    

#%% Test peak fitting of single gaussian to TiOx region from extracted nanocavity spectra


# def fit_gaussian(spectrum, fit_range, peak_name = None, R_sq_thresh = 0.9, smooth_first = False, plot = False):    
    
#     ## Get region of spectrum to fit
#     fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
#     x = spectrum.x[fit_range_index[0]:fit_range_index[1]]   
    
#     ## Smooth, truncate y region
#     y = spectrum.y_baselined
#     if smooth_first == True:
#         y = spt.butter_lowpass_filt_filt(y, cutoff = 2000, fs = 40000)
#     y = y[fit_range_index[0]:fit_range_index[1]]
#     y_raw = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
    

#     # Fit
    
#     ## Initial guesses
#     i_max = y.argmax() # index of highest value - for guess, assumed to be Gaussian peak
#     height0 = y[i_max]
#     mu0 = x[i_max] # centre x position of guessed peak
#     baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
#     i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
#     # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
#     # half, and even if this point is a distant outlier the fit should still converge.
#     sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
#     width0 = sigma0 * 2 * np.sqrt(2*np.log(2))
#     # sigma0 = 3
#     p0 = height0, mu0, width0, baseline0
    
#     ## Perform fit (height, mu, width, baseline)
#     start = time.time()
    
#     error = False
    
#     try:
#         popt, pcov = curve_fit(f = gaussian, 
#                             xdata = x,
#                             ydata = y, 
#                             p0 = p0,
#                             bounds=(
#                                 (0, x.min(), 0, y.min()),
#                                 (height0 * 1.2, x.max(), (x.max() - x.min()) * 2, height0),),)
        
#     except:
#         popt = [height0, mu0, sigma0, baseline0]
#         error = True
#         print('\nFit Error')
      
#     finish = time.time()
#     runtime = finish - start
        
#     # print('Guess:', np.array(p0))
#     # print('Fit:  ', popt)
    
#     ## Get fit data
#     height = popt[0]
#     mu = popt[1]
#     width = popt[2]
#     baseline = popt[3]
#     sigma = width / (2 * np.sqrt(2 * np.log(2)))
#     area = height * sigma * np.sqrt(2 * np.pi)
#     y_fit = gaussian(x, *popt)
#     residuals = y - y_fit
#     residuals_sum = np.abs(residuals).sum()
#     R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
    
#     if peak_name is None:
#         peak_name = str(int(np.round(np.mean(fit_range),0)))
    
#     ## Screening poor fits
#     if R_sq < R_sq_thresh or np.isnan(R_sq) or error == True:
#         print('\nPoor Fit')
#         print(particle.name)
#         print(spectrum.name)
#         print(peak_name)
#         print('R^2 = ' + str(np.round(R_sq, 3)))
#         print('Guess:', np.array(p0))
#         print('Fit:  ', popt)
#         error = True
#     else:
#         error = False
    
#     ## Add fit data to peak class
#     this_peak = Peak(*popt)
#     this_peak.area = area
#     this_peak.sigma = sigma
#     this_peak.name = peak_name
#     this_peak.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
#     this_peak.R_sq = R_sq
#     this_peak.residuals = SERS.SERS_Spectrum(x = x, y = residuals)
#     this_peak.error = error        
       
    
#     # Plot
    
#     if plot == True: # or error == True:
        
#         fig, ax = plt.subplots(1,1,figsize=[14,9])
#         fig.suptitle(particle.name, fontsize = 'large')
#         ax.set_title(spectrum.name + ' - ' + this_peak.name + ' cm$^{-1}$ Peak Fit', fontsize = 'large')
#         ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#         ax.set_ylabel('SERS Intensity (cts/mW/s)')
#         if smooth_first:
#             ax.plot(x, y, color = 'grey', label = 'Smoothed') 
#         ax.plot(x, y_raw, color = 'grey', linestyle = '--', label = 'Data')
#         ax.plot(x, y_fit, label = 'Fit', color = 'orange')
#         ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
#         ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
#         ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
#         if error:
#             ax.text(x = mu0, y = height0, s = 'ERROR')
#         ax.set_xlim(fit_range[0] - 3, fit_range[1] + 3)
#         ax.set_ylim(None, y.max()*1.2)
#         ax.legend()
#         plt.show()
        
#     return this_peak

# particle = tiox_avg_particle
# particle = particles[5]

# for timescan in particle.powerseries:
    
#     timescan.peaks = []
    
#     spectrum = SERS.SERS_Spectrum(x = timescan.x, y = timescan.min_spectrum)
#     spectrum.name = timescan.name
#     spectrum.truncate(start_x = 575, end_x = 800)
#     baseline = spt.baseline_als(spectrum.y, lam = 1e4, p = 1e-3)
#     spectrum.y_baselined = spectrum.y - baseline    
#     # plt.plot(spectrum.x, spectrum.y)
#     # plt.plot(spectrum.x, baseline)
#     # plt.plot(spectrum.x, spectrum.y_baselined)
#     # plt.show()
    
#     peak = fit_gaussian(spectrum, fit_range = [575, 800], plot = True, peak_name = 'TiOx', smooth_first = True, R_sq_thresh = 0.9)
#     timescan.peaks.append(peak)


#%% Test DF processing and plotting

particle = my_h5['ParticleScannerScan_1']['Particle_51']
keys = list(particle.keys())
keys = natsort.natsorted(keys)

z_scans = []
images = []

for key in keys:
    if 'z_scan' in key:
        z_scans.append(particle[key])
    if 'image' in key:
        images.append(particle[key])

for i, scan in enumerate(z_scans):
    
    name = str(copy(scan.name))
    
    scan = df.Z_Scan(scan, z_min = scan.attrs['neg'], z_max = scan.attrs['pos'], z_trim = [8, -8])
    scan.truncate(400, 900)   
    scan.condense_z_scan()
    spectrum = scan.df_spectrum
    scan.df_spectrum = df.DF_Spectrum(scan.x, spectrum)
    scan.df_spectrum.find_critical_wln()
    image = images[i]
    scan.df_spectrum = df.df_screening(scan, scan.df_spectrum, image, plot = True)
    
    z_scans[i] = scan
    images[i] = image
    
particle.z_scans = z_scans
particle.images = images
particle.crit_wlns = [z_scans[0].df_spectrum.crit_wln, z_scans[1].df_spectrum.crit_wln]
particle.is_npom = [particle.z_scans[0].df_spectrum.is_npom, particle.z_scans[0].df_spectrum.is_npom]    

#%% Get all particles to analyze into Particle class with h5 locations and in a list

particles = []
sers_skip_list = [7, 8, 9, 13, 14, 16, 25, 33, 65, 68, 85, 107, 116, 117, 118, 119, 120, 121, 141, 144]
df_skip_list = [3, 7, 8, 9, 13, 14, 16, 25, 33, 98, 132, 141, 143, 144]
skip_list = []

scan_list = ['ParticleScannerScan_1']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)
            # print(particle)
    for particle in particle_list:
        if int(particle[particle.find('_')+1:]) in skip_list:
            particle_list.remove(particle)
            # print(particle)
            
    # Loop over particles in particle scan
    
    for particle in particle_list:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'TiOx 60nm NPoM ' + str(particle_scan) + '_' + particle
        this_particle.scan = str(particle_scan)
        this_particle.particle_number = str(particle)
        particles.append(this_particle)
        

#%% Loop over all particles and process kinetic powerseries & df

print('\nProcessing spectra...')
for particle in tqdm(particles, leave = True):
    
    process_kinetic_powerseries(particle)
    process_df(particle, plot = True, save = True, tinder = False)

## Run particle tinder once to get list of rejected particles
# rejected_0 = []
# rejected_1 = []
# for i, particle in enumerate(particles):
    
#     if particle.is_npom[0] == False:
#         rejected_0.append(i)
        
#     if particle.is_npom[1] == False:
#         rejected_1.append(i)


#%% Run peak fitting

print('\nPeak fitting...')
for particle in tqdm(particles, leave = True):
    
    for timescan in particle.powerseries:
        
        timescan.peaks = []
        
        spectrum = SERS.SERS_Spectrum(x = timescan.x, y = timescan.min_spectrum)
        spectrum.name = timescan.name
        spectrum.truncate(start_x = 575, end_x = 800)
        baseline = spt.baseline_als(spectrum.y, lam = 1e4, p = 1e-3)
        spectrum.y_baselined = spectrum.y - baseline    
        # plt.plot(spectrum.x, spectrum.y)
        # plt.plot(spectrum.x, baseline)
        # plt.plot(spectrum.x, spectrum.y_baselined)
        # plt.show()
        
        peak = fit_gaussian(spectrum, fit_range = [575, 800], plot = False, peak_name = 'TiOx', smooth_first = True, R_sq_thresh = 0.9)
        timescan.peaks.append(peak)

## Report fit errors
count = 0
scans = 0
for particle in particles:
    for i, spectrum in enumerate(particle.powerseries):
        scans += 1
        for peak in spectrum.peaks:
            if peak.error: count += 1
print('\nFit Errors (%): ')
print(100*count/(scans * len(spectrum.peaks)))

    
#%% Make avg particle

avg_particle = deepcopy(particles[0])
avg_particle.name = 'TiOx 60nm NPoM Average'


# Average the processed tiemscan data

for i, timescan in enumerate(avg_particle.powerseries):
    ## Set all timescan y-elements to 0
    timescan.Y = np.zeros(timescan.Y.shape)
    timescan.y = np.zeros(timescan.y.shape)
    timescan.nanocavity_spectrum = np.zeros(timescan.nanocavity_spectrum.shape)
    timescan.min_spectrum = np.zeros(timescan.min_spectrum.shape)
    timescan.sem_spectrum = np.zeros(timescan.sem_spectrum.shape)
    
    ## Add y-value elements from particle timescans
    counter = 0
    for particle in particles:
        particle.timescan = particle.powerseries[i]
        timescan.Y += particle.timescan.Y
        timescan.y += particle.timescan.y
        timescan.nanocavity_spectrum += particle.timescan.nanocavity_spectrum
        timescan.min_spectrum += particle.timescan.min_spectrum
        timescan.sem_spectrum += particle.timescan.sem_spectrum
        counter += 1
        
    ## Divide
    timescan.Y = timescan.Y/counter
    timescan.y = timescan.y/counter
    timescan.nanocavity_spectrum = timescan.nanocavity_spectrum/counter
    timescan.min_spectrum = timescan.min_spectrum/counter
    timescan.sem_spectrum = timescan.sem_spectrum/counter 
    

# Add peak data

for i, spectrum in enumerate(avg_particle.powerseries):
    
    ## Set up peaks list
    for spectrum in avg_particle.powerseries:
        spectrum.peaks = deepcopy(particles[0].powerseries[0].peaks)
        
        for peak in spectrum.peaks:
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
        counter += 1
        for i, spectrum in enumerate(avg_particle.powerseries):
            for j, peak in enumerate(spectrum.peaks):
                this_peak = deepcopy(particle.powerseries[i].peaks[j])
                peak.area.append(this_peak.area)
                peak.baseline.append(this_peak.baseline)
                peak.height.append(this_peak.height)
                peak.mu.append(this_peak.mu)
                peak.sigma.append(this_peak.sigma)
                peak.width.append(this_peak.width)
    
    ## Calculate std & sem for each peak value, ignore nans
    for spectrum in avg_particle.powerseries:
        for peak in spectrum.peaks:
            
            peak.area_std = np.nanstd(peak.area)
            peak.baseline_std = np.nanstd(peak.baseline)    
            peak.height_std = np.nanstd(peak.height)
            peak.mu_std = np.nanstd(peak.mu)
            peak.sigma_std = np.nanstd(peak.sigma)
            peak.width_std = np.nanstd(peak.width)
            
            peak.area_sem = peak.area_std/(counter**0.5)
            peak.baseline_sem = peak.baseline_std/(counter**0.5)    
            peak.height_sem = peak.height_std/(counter**0.5)
            peak.mu_sem = peak.mu_std/(counter**0.5)
            peak.sigma_sem = peak.sigma_std/(counter**0.5)
            peak.width_sem = peak.width_std/(counter**0.5)
            
    ## Calculate mean (ignoring nans)
    for spectrum in avg_particle.powerseries:
        for peak in spectrum.peaks:
            peak.area = np.nanmean(peak.area)
            peak.baseline = np.nanmean(peak.baseline)
            peak.height = np.nanmean(peak.height)
            peak.mu = np.nanmean(peak.mu)
            peak.sigma = np.nanmean(peak.sigma)
            peak.width = np.nanmean(peak.width)
        
particles.append(avg_particle)

#%% End TiOx

tiox_avg_particle = deepcopy(avg_particle)
tiox_particles = deepcopy(particles)

'''
End
TiOx
'''


#%% Start Toluene

'''
Start
Toluene
'''

my_h5 = control_h5


#%% New spectral calibration using BPT calibration and comparing to calibrated spectra from cal_h5

## Truncation for all spectra
truncate_range = [250, 2300]

## Get ref (measured) BPT peaks
neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
neon_ref.normalise()
neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
neon_ref_peaks = neon_ref_peaks[0:-4]
neon_ref_peaks_y = neon_ref.y_norm[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
neon_ref_peaks_y = neon_ref_peaks_y[0:-4]
neon_ref.__dict__.pop('dset')

## Get literature neon peaks
neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
neon_wls = neon_wls[15:]

'''Make peak matching more robust below'''
# Cut detected peaks that are too close together
delete_list = []
for i, peak in enumerate(neon_ref_peaks):
    if i < len(neon_ref_peaks)-1:        
        if neon_ref_peaks[i+1] - neon_ref_peaks[i] < 1:
            x = np.argmin((neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i])], neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i+1])]))
            delete_list.append(x+i)
neon_ref_peaks = np.delete(neon_ref_peaks, delete_list)    
neon_ref_peaks_y = np.delete(neon_ref_peaks_y, delete_list)  

## Assert same number of ref and lit neon peaks
assert(len(neon_ref_peaks) == len(neon_wls), print(len(neon_ref_peaks)))

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

cal_matrix[1] = 1.5

## Plot calibrated ref BPT v. BPT from cal_h5
### Process ref bpt
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
# ax.set_xlim(200, 1000)
# ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas')
ax.legend()

## Plot calibrated ref Co-TAPP-SMe v. from cal_h5
### Process ref Co-TAPP-SMe
# spectrum = my_h5['ref_meas_0']['Co-TAPP-SMe_Air_100s_1uW_0']
# spectrum = SERS.SERS_Spectrum(spectrum)
# spectrum.x = linear(spectrum.x, *cal_matrix)
# spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
# # spectrum.chunk(60)
# spectrum.normalise()
# ### Process cal Co-TAPP-SMe
# cotapp_cal = cal_h5['Co-TAPP-SMe MLAgg 633nm SERS Calibrated']
# cotapp_cal = SERS.SERS_Spectrum(x = cotapp_cal.attrs['x'], y = cotapp_cal.attrs['y'])
# cotapp_cal.normalise()
# ### Plot
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_title('Neon Lamp Calibration applied to Co-TAPP-SMe')
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('Counts')
# ax.plot(spectrum.x, spectrum.y_norm, label = 'Measured & Calibrated Co-TAPP-SMe')
# ax.plot(cotapp_cal.x, cotapp_cal.y_norm, label = 'Co-TAPP-SMe Cal')
# ax.legend()
# ax.set_xlim(950, 1650)


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas_0']['white_ref_x5_0']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = linear(white_ref.x, *cal_matrix)
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)

## Get white bkg (counts in notch region)
notch_range = [160, 175]
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

particle = my_h5['ref_meas_0']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []

for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 80000)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


## Plot dark counts & smoothed dark counts
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts')
for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y, color = 'black')
    ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')


# Plot dark subtracted as test

particle = my_h5['ParticleScannerScan_3']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts Subtraction Test')

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    # spectrum.calibrate_intensity(R_setup = R_setup,
    #                               dark_counts = dark_powerseries[i].y_smooth,
    #                               exposure = spectrum.cycle_time,
    #                               laser_power = spectrum.laser_power)
    powerseries[i] = spectrum
    
i = 5
    
ax.plot(powerseries[i].x, powerseries[i].y, color = 'black')
ax.plot(powerseries[i].x, powerseries[i].y - dark_powerseries[i].y_smooth, color = 'blue')
ax.plot(powerseries[i].x, dark_powerseries[i].y_smooth, color = 'red')    
ax.plot(powerseries[0].x, powerseries[0].y, color = 'black')
ax.plot(powerseries[0].x, powerseries[0].y - dark_powerseries[0].y_smooth, color = 'blue')
ax.plot(powerseries[0].x, dark_powerseries[0].y_smooth, color = 'red')  
# ax.set_ylim(300, 400)

#%% Testing background subtraction & cosmic ray removal

''' Skipping for now since TiOx peak is broad, gets caught in baseline'''

particle = my_h5['ParticleScannerScan_3']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    powerseries[i] = spectrum


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
spectrum = powerseries[2]
spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

## Baseline
spectrum.baseline = spt.baseline_als(spectrum.y, 1e6, 1e-1, niter = 10)
spectrum.y_baselined = spectrum.y - spectrum.baseline
spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
## Plot raw, baseline, baseline subtracted
offset = 50000
ax.plot(spectrum.x, (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
ax.plot(spectrum.x, spectrum.y_baselined + (i*offset), linewidth = 1, color = 'blue', alpha = 1, label = 'Background subtracted', zorder = 2)
ax.plot(spectrum.x, spectrum.baseline- spectrum.y.min() + (i*offset), color = 'darkred', label = 'Background', linewidth = 1)    
ax.plot(spectrum.x, spectrum.y_cosmic + (i*offset), color = 'orange', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
title = 'Background Subtraction & Cosmic Ray Test'
fig.suptitle(title + particle.name)    
ax.legend()
# ax.set_xlim(1150, 1240)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)
    

#%% Testing iterative polynomial fit for fitting PL background (Bart's method)

''' Skipping this for now - no PL and TiOx SERS peak is very broad

## Get powerseries
particle = my_h5['ParticleScannerScan_1']['Particle_40']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    ### Modpoly fit for PL background    
    pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    powerseries[i] = spectrum

## Plot
spectrum = powerseries[0]
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
offset = 50000
spectrum.plot(ax = ax, plot_y = spectrum.y)
spectrum.plot(ax = ax, plot_y = spectrum.pl)
fig.suptitle(particle.name)    
# ax.legend()
# ax.set_xlim(350, 1900)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)    
'''


#%% Test SERS processing & plotting

particle = my_h5['ParticleScannerScan_3']['Particle_15']


# Add each timescan to powerseries

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


# Process timescan

for i, spectrum in enumerate(powerseries):

    name = str(copy(spectrum.name))
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')

    ## Calibrate x-axis
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    
    ## Calibrate intensity
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    
    ## Modpoly fit for PL background    
    # pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    # spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    # spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    ## Cosmic ray & ALS Baseline Subtraction
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
    # spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    # spectrum.y_baselined = spectrum.y
    # spectrum.normalise()
    spectrum.name = name[name.rfind('/') + 1:]
    
    spectrum.extract_nanocavity(plot = False)
    
    powerseries[i] = spectrum
    

particle = Particle()
particle.name = 'test'
particle.powerseries = powerseries
# particle.timescan = powerseries[0]
particle.chunk_size = 1

for timescan in powerseries:
    particle.timescan = timescan
    plot_timescan(particle)


#%% Test nanocavity extraction

for timescan in particle.powerseries:
    
    particle.timescan = timescan
    plot_timescan(particle)
    
    # extract_nanocavity(self = timescan, plot = False)
    timescan_y = timescan.Y.copy()
    pixel_baseline = np.zeros(len(timescan.x))
    pixel_min = np.zeros(len(timescan.x))
    pixel_sem = np.zeros(len(timescan.x))
    for pixel in range(0,len(timescan.x)):
        pixel_baseline[pixel] = np.polyfit(timescan.t, timescan_y[:,pixel], deg = 0)
        pixel_min[pixel] = np.min(timescan_y[:,pixel])
        pixel_sem[pixel] = np.std(timescan_y[:,pixel])/(len(timescan_y[:,pixel]) ** 0.5)
    timescan.nanocavity_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_baseline)
    timescan.min_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_min)
    timescan.sem_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_sem)
    
    fig, ax = plt.subplots(1,1,figsize=[12,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
    ax.plot(timescan.x, timescan.y, color = 'black', label = 'Avg')
    ax.plot(timescan.x, timescan.nanocavity_spectrum.y, color = 'red', alpha = 1, linestyle = '--', label = 'Extracted nanocavity (0-deg polyfit)')
    ax.plot(timescan.x, timescan.min_spectrum.y, color = 'green', alpha = 0.5, label = 'Minimum pixel')
    ax.plot(timescan.x, timescan.sem_spectrum.y, color = 'orange', label = 'Pixel SEM')
    ax.set_title(timescan.name + '\nNanocavity Extraction')
    ax.legend()
    

#%% Get all particles to analyze into Particle class with h5 locations and in a list

particles = []
skip_list = [3, 4, 6, 7, 9, 10, 11, 12, 13, 60, 61, 62, 63, 64, 68, 169]

scan_list = ['ParticleScannerScan_3']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)
            # print(particle)
    for particle in particle_list:
        if int(particle[particle.find('_')+1:]) in skip_list:
            particle_list.remove(particle)
            # print(particle)
            
    # Loop over particles in particle scan
    
    for particle in particle_list:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'Toluene 60nm NPoM ' + str(particle_scan) + '_' + particle
        this_particle.scan = str(particle_scan)
        this_particle.particle_number = str(particle)
        particles.append(this_particle)
        

#%% Loop over all particles and process kinetic powerseries

print('\nProcessing spectra...')
for particle in tqdm(particles, leave = True):
    
    process_kinetic_powerseries(particle)

#%% Make avg particle

avg_particle = deepcopy(particles[0])
avg_particle.name = 'Toluene 60nm NPoM Average'


# Average the processed tiemscan data

for i, timescan in enumerate(avg_particle.powerseries):
    ## Set all timescan y-elements to 0
    timescan.Y = np.zeros(timescan.Y.shape)
    timescan.y = np.zeros(timescan.y.shape)
    timescan.nanocavity_spectrum = np.zeros(timescan.nanocavity_spectrum.shape)
    timescan.min_spectrum = np.zeros(timescan.min_spectrum.shape)
    timescan.sem_spectrum = np.zeros(timescan.sem_spectrum.shape)
    
    ## Add y-value elements from particle timescans
    counter = 0
    for particle in particles:
        particle.timescan = particle.powerseries[i]
        timescan.Y += particle.timescan.Y
        timescan.y += particle.timescan.y
        timescan.nanocavity_spectrum += particle.timescan.nanocavity_spectrum
        timescan.min_spectrum += particle.timescan.min_spectrum
        timescan.sem_spectrum += particle.timescan.sem_spectrum
        counter += 1
        
    ## Divide
    timescan.Y = timescan.Y/counter
    timescan.y = timescan.y/counter
    timescan.nanocavity_spectrum = timescan.nanocavity_spectrum/counter
    timescan.min_spectrum = timescan.min_spectrum/counter
    timescan.sem_spectrum = timescan.sem_spectrum/counter    
        
particles.append(avg_particle)


#%% End Toluene

toluene_avg_particle = deepcopy(avg_particle)
toluene_particles = deepcopy(particles)

'''
End
Toluene
'''


#%% Start KCl

'''
Start
KCl
'''

my_h5 = control_h5


#%% New spectral calibration using BPT calibration and comparing to calibrated spectra from cal_h5

## Truncation for all spectra
truncate_range = [250, 2300]

## Get ref (measured) BPT peaks
neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
neon_ref.normalise()
neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
neon_ref_peaks = neon_ref_peaks[0:-4]
neon_ref_peaks_y = neon_ref.y_norm[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.03)]
neon_ref_peaks_y = neon_ref_peaks_y[0:-4]
neon_ref.__dict__.pop('dset')

## Get literature neon peaks
neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
neon_wls = neon_wls[15:]

'''Make peak matching more robust below'''
# Cut detected peaks that are too close together
delete_list = []
for i, peak in enumerate(neon_ref_peaks):
    if i < len(neon_ref_peaks)-1:        
        if neon_ref_peaks[i+1] - neon_ref_peaks[i] < 1:
            x = np.argmin((neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i])], neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i+1])]))
            delete_list.append(x+i)
neon_ref_peaks = np.delete(neon_ref_peaks, delete_list)    
neon_ref_peaks_y = np.delete(neon_ref_peaks_y, delete_list)  

## Assert same number of ref and lit neon peaks
assert(len(neon_ref_peaks) == len(neon_wls), print(len(neon_ref_peaks)))

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

cal_matrix[1] = 1.5

## Plot calibrated ref BPT v. BPT from cal_h5
### Process ref bpt
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
# ax.set_xlim(200, 1000)
# ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas')
ax.legend()

## Plot calibrated ref Co-TAPP-SMe v. from cal_h5
### Process ref Co-TAPP-SMe
# spectrum = my_h5['ref_meas_0']['Co-TAPP-SMe_Air_100s_1uW_0']
# spectrum = SERS.SERS_Spectrum(spectrum)
# spectrum.x = linear(spectrum.x, *cal_matrix)
# spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
# # spectrum.chunk(60)
# spectrum.normalise()
# ### Process cal Co-TAPP-SMe
# cotapp_cal = cal_h5['Co-TAPP-SMe MLAgg 633nm SERS Calibrated']
# cotapp_cal = SERS.SERS_Spectrum(x = cotapp_cal.attrs['x'], y = cotapp_cal.attrs['y'])
# cotapp_cal.normalise()
# ### Plot
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_title('Neon Lamp Calibration applied to Co-TAPP-SMe')
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('Counts')
# ax.plot(spectrum.x, spectrum.y_norm, label = 'Measured & Calibrated Co-TAPP-SMe')
# ax.plot(cotapp_cal.x, cotapp_cal.y_norm, label = 'Co-TAPP-SMe Cal')
# ax.legend()
# ax.set_xlim(950, 1650)


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas_0']['white_ref_x5_0']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = linear(white_ref.x, *cal_matrix)
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)

## Get white bkg (counts in notch region)
notch_range = [160, 175]
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

particle = my_h5['ref_meas_0']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []

for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 80000)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


## Plot dark counts & smoothed dark counts
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts')
for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y, color = 'black')
    ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')


# Plot dark subtracted as test

particle = my_h5['ParticleScannerScan_4']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Dark Counts Subtraction Test')

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    # spectrum.calibrate_intensity(R_setup = R_setup,
    #                               dark_counts = dark_powerseries[i].y_smooth,
    #                               exposure = spectrum.cycle_time,
    #                               laser_power = spectrum.laser_power)
    powerseries[i] = spectrum
    
i = 5
    
ax.plot(powerseries[i].x, powerseries[i].y, color = 'black')
ax.plot(powerseries[i].x, powerseries[i].y - dark_powerseries[i].y_smooth, color = 'blue')
ax.plot(powerseries[i].x, dark_powerseries[i].y_smooth, color = 'red')    
ax.plot(powerseries[0].x, powerseries[0].y, color = 'black')
ax.plot(powerseries[0].x, powerseries[0].y - dark_powerseries[0].y_smooth, color = 'blue')
ax.plot(powerseries[0].x, dark_powerseries[0].y_smooth, color = 'red')  
# ax.set_ylim(300, 400)

#%% Testing background subtraction & cosmic ray removal

''' Skipping for now since TiOx peak is broad, gets caught in baseline'''

particle = my_h5['ParticleScannerScan_4']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    powerseries[i] = spectrum


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
spectrum = powerseries[2]
spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

## Baseline
spectrum.baseline = spt.baseline_als(spectrum.y, 1e6, 1e-1, niter = 10)
spectrum.y_baselined = spectrum.y - spectrum.baseline
spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
## Plot raw, baseline, baseline subtracted
offset = 50000
ax.plot(spectrum.x, (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
ax.plot(spectrum.x, spectrum.y_baselined + (i*offset), linewidth = 1, color = 'blue', alpha = 1, label = 'Background subtracted', zorder = 2)
ax.plot(spectrum.x, spectrum.baseline- spectrum.y.min() + (i*offset), color = 'darkred', label = 'Background', linewidth = 1)    
ax.plot(spectrum.x, spectrum.y_cosmic + (i*offset), color = 'orange', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
title = 'Background Subtraction & Cosmic Ray Test'
fig.suptitle(title + particle.name)    
ax.legend()
# ax.set_xlim(1150, 1240)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)
    

#%% Testing iterative polynomial fit for fitting PL background (Bart's method)

''' Skipping this for now - no PL and TiOx SERS peak is very broad

## Get powerseries
particle = my_h5['ParticleScannerScan_1']['Particle_40']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

for i, spectrum in enumerate(powerseries):

    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    ### Modpoly fit for PL background    
    pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    powerseries[i] = spectrum

## Plot
spectrum = powerseries[0]
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
offset = 50000
spectrum.plot(ax = ax, plot_y = spectrum.y)
spectrum.plot(ax = ax, plot_y = spectrum.pl)
fig.suptitle(particle.name)    
# ax.legend()
# ax.set_xlim(350, 1900)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)    
'''


#%% Test processing & plotting

particle = my_h5['ParticleScannerScan_4']['Particle_0']


# Add each timescan to powerseries

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


# Process timescan

for i, spectrum in enumerate(powerseries):

    name = str(copy(spectrum.name))
    spectrum = SERS.SERS_Timescan(spectrum)
    spectrum.__dict__.pop('dset')

    ## Calibrate x-axis
    spectrum.x = linear(spectrum.x, *cal_matrix)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.truncate(truncate_range[0], truncate_range[1])
    
    ## Calibrate intensity
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y_smooth,
                                  exposure = spectrum.cycle_time,
                                  laser_power = spectrum.laser_power)
    
    ## Modpoly fit for PL background    
    # pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
    # spectrum.pl = pl
    ### Sum of PL background from 350 to 1900 1/cm
    # spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
    
    ## Cosmic ray & ALS Baseline Subtraction
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
    # spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    # spectrum.y_baselined = spectrum.y
    # spectrum.normalise()
    spectrum.name = name[name.rfind('/') + 1:]
    
    spectrum.extract_nanocavity(plot = False)
    
    powerseries[i] = spectrum
    

particle = Particle()
particle.name = 'test'
particle.powerseries = powerseries
# particle.timescan = powerseries[0]
particle.chunk_size = 1

for timescan in powerseries:
    particle.timescan = timescan
    plot_timescan(particle)


#%% Test nanocavity extraction

for timescan in particle.powerseries:
    
    particle.timescan = timescan
    plot_timescan(particle)
    
    # extract_nanocavity(self = timescan, plot = False)
    timescan_y = timescan.Y.copy()
    pixel_baseline = np.zeros(len(timescan.x))
    pixel_min = np.zeros(len(timescan.x))
    pixel_sem = np.zeros(len(timescan.x))
    for pixel in range(0,len(timescan.x)):
        pixel_baseline[pixel] = np.polyfit(timescan.t, timescan_y[:,pixel], deg = 0)
        pixel_min[pixel] = np.min(timescan_y[:,pixel])
        pixel_sem[pixel] = np.std(timescan_y[:,pixel])/(len(timescan_y[:,pixel]) ** 0.5)
    timescan.nanocavity_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_baseline)
    timescan.min_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_min)
    timescan.sem_spectrum = SERS.SERS_Spectrum(timescan.x, pixel_sem)
    
    fig, ax = plt.subplots(1,1,figsize=[12,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
    ax.plot(timescan.x, timescan.y, color = 'black', label = 'Avg')
    ax.plot(timescan.x, timescan.nanocavity_spectrum.y, color = 'red', alpha = 1, linestyle = '--', label = 'Extracted nanocavity (0-deg polyfit)')
    ax.plot(timescan.x, timescan.min_spectrum.y, color = 'green', alpha = 0.5, label = 'Minimum pixel')
    ax.plot(timescan.x, timescan.sem_spectrum.y, color = 'orange', label = 'Pixel SEM')
    ax.set_title(timescan.name + '\nNanocavity Extraction')
    ax.legend()
    
    
#%% Get all particles to analyze into Particle class with h5 locations and in a list

particles = []
skip_list = [33, 40, 42, 49, 67, 76, 77, 79, 87, 89, 91, 105, 107, 116, 117, 121, 123, 125,
             146, 147, 152, 156, 160, 161, 162, 170, 175, 183, 190]

scan_list = ['ParticleScannerScan_4']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)
            # print(particle)
    for particle in particle_list:
        if int(particle[particle.find('_')+1:]) in skip_list:
            particle_list.remove(particle)
            # print(particle)
            
    # Loop over particles in particle scan
    
    for particle in particle_list:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'KCl 60nm NPoM ' + str(particle_scan) + '_' + particle
        this_particle.scan = str(particle_scan)
        this_particle.particle_number = str(particle)
        particles.append(this_particle)
        

#%% Loop over all particles and process kinetic powerseries

print('\nProcessing spectra...')
for particle in tqdm(particles, leave = True):
    
    process_kinetic_powerseries(particle)
    

#%% Make avg particle

avg_particle = deepcopy(particles[0])
avg_particle.name = 'KCl 60nm NPoM Average'


# Average the processed tiemscan data

for i, timescan in enumerate(avg_particle.powerseries):
    ## Set all timescan y-elements to 0
    timescan.Y = np.zeros(timescan.Y.shape)
    timescan.y = np.zeros(timescan.y.shape)
    timescan.nanocavity_spectrum = np.zeros(timescan.nanocavity_spectrum.shape)
    timescan.min_spectrum = np.zeros(timescan.min_spectrum.shape)
    timescan.sem_spectrum = np.zeros(timescan.sem_spectrum.shape)
    
    ## Add y-value elements from particle timescans
    counter = 0
    for particle in particles:
        particle.timescan = particle.powerseries[i]
        timescan.Y += particle.timescan.Y
        timescan.y += particle.timescan.y
        timescan.nanocavity_spectrum += particle.timescan.nanocavity_spectrum
        timescan.min_spectrum += particle.timescan.min_spectrum
        timescan.sem_spectrum += particle.timescan.sem_spectrum
        counter += 1
        
    ## Divide
    timescan.Y = timescan.Y/counter
    timescan.y = timescan.y/counter
    timescan.nanocavity_spectrum = timescan.nanocavity_spectrum/counter
    timescan.min_spectrum = timescan.min_spectrum/counter
    timescan.sem_spectrum = timescan.sem_spectrum/counter    
        
particles.append(avg_particle)


#%% End KCl

kcl_avg_particle = deepcopy(avg_particle)
kcl_particles = deepcopy(particles)

'''
End
KCl
'''


#%% Save processed data to spydata here

'''
Save processed data to spydata here
'''


#%% Import spydata and run from here

'''
Load in spydata then run from here
'''

save = True
save_to_h5 = True

## h5 File for saving processed data
if save_to_h5 == True:
    save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-07-25_TiOx_633nm_SERS_Powerseries Processed Data.h5")



#%% Loop over all particles, plot, and save timescan data to h5

print('\nPlotting spectra...')

# for particle in tqdm(tiox_particles, leave = True):
    
#     for timescan in particle.powerseries:
        
#         particle.timescan = timescan
#         plot_timescan(particle, save = save, save_to_h5 = save_to_h5)
#         if save_to_h5 == True:
#             save_timescan(particle)
            
#     plot_nanocavity_powerseries(particle, save = save, save_to_h5 = save_to_h5)
#     plot_powerseries_peak_areas(particle, save = save, save_to_h5 = save_to_h5)
#     plot_powerseries_peak_positions(particle, save = save, save_to_h5 = save_to_h5)
    
#     if save_to_h5 == True:
#         save_peaks_powerswitch_recovery(particle)
    
    
# for particle in tqdm(toluene_particles, leave = True):
    
#     for timescan in particle.powerseries:
        
#         particle.timescan = timescan
#         plot_timescan(particle, save = save, save_to_h5 = save_to_h5)
#         if save_to_h5 == True:
#             save_timescan(particle)
        
#     plot_nanocavity_powerseries(particle, save = save, save_to_h5 = save_to_h5)


# for particle in tqdm(kcl_particles, leave = True):
    
#     for timescan in particle.powerseries:
        
#         particle.timescan = timescan
#         plot_timescan(particle, save = save, save_to_h5 = save_to_h5)
#         if save_to_h5 == True:
#             save_timescan(particle)
        
#     plot_nanocavity_powerseries(particle, save = save, save_to_h5 = save_to_h5)


#%% Plot nanocavity spectra of avg_particles against each other

for i in range(0, len(toluene_avg_particle.powerseries)):

    fig, (ax) = plt.subplots(1, 1, figsize=[14,10])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('SERS Intensity (cts/mW/s)', fontsize = 'large')
    title = 'TiOx v Control NPoMs Nanocavity 633 nm SERS ' + str(i)
    ax.set_title('TiOx v Control NPoMs\nNanocavity 633 nm SERS' '\n' + str(np.round(tiox_avg_particle.powerseries[i].laser_power * 1000)) + ' $\mu$W', fontsize = 'x-large', pad = 10)
    
    
    ax.plot(tiox_avg_particle.powerseries[i].x, tiox_avg_particle.powerseries[i].min_spectrum, color = 'blue', label = 'TiOx')
    ax.fill_between(tiox_avg_particle.powerseries[i].x, 
                    tiox_avg_particle.powerseries[i].min_spectrum - tiox_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    tiox_avg_particle.powerseries[i].min_spectrum + tiox_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    color = 'blue', alpha = 0.2)
    
    ax.plot(toluene_avg_particle.powerseries[i].x, toluene_avg_particle.powerseries[i].min_spectrum, color = 'red', label = 'Toluene')
    ax.fill_between(toluene_avg_particle.powerseries[i].x, 
                    toluene_avg_particle.powerseries[i].min_spectrum - toluene_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    toluene_avg_particle.powerseries[i].min_spectrum + toluene_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    color = 'red', alpha = 0.2)
    
    ax.plot(kcl_avg_particle.powerseries[i].x, kcl_avg_particle.powerseries[i].min_spectrum, color = 'green', label = 'KCl')
    ax.fill_between(kcl_avg_particle.powerseries[i].x, 
                    kcl_avg_particle.powerseries[i].min_spectrum - kcl_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    kcl_avg_particle.powerseries[i].min_spectrum + kcl_avg_particle.powerseries[i].sem_spectrum * (10**0.5),
                    color = 'green', alpha = 0.2)

    ax.legend(fontsize = 'large')
    
    ## Save
    if save == True:
        save_dir = get_directory('Compiled')
        fig.savefig(save_dir + title + '.svg', format = 'svg')
        plt.close(fig)
      
    ## Save to h5 file as jpg
    if save_to_h5 == True:
        try:
            group = save_h5[str('Compiled')]
        except:
            group = save_h5.create_group(str('Compiled'))

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_array = np.array(canvas.renderer.buffer_rgba())
        group.create_dataset(name = title + '_jpg_%d', data = fig_array)
        plt.close(fig)