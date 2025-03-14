# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:30:15 2024

@author: il322

Plotter for Si Raman during Kymera shutter test

Copied from 2024-07-25_633nm_SERS_TiOx_NPoM_Plotter.py

Data: 
    2024-12-19_Kymera_Shutter_Si_test.h5

    


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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-12-19_Kymera_Shutter_Si_test.h5")

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
if save_to_h5 == True:
    save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-12-19_Kymera_Shutter_Si_Data.h5")


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
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-12-19 Kymera Shutter Test Si Analysis\\' + particle_name + '\\'
    
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
    

def lorentzian(x, height, mu, fwhm, baseline):
    I = height
    x0 = mu
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot + baseline
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
    # print(height0)
    
    mu0 = x[i_max] # centre x position of guessed peak
    # print(mu0)
    
    baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
    # print(baseline0)
    
    i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
    if i_half == i_max:
        i_half -= 1
    # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
    # half, and even if this point is a distant outlier the fit should still converge.
    sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
    width0 = sigma0 * 2 * np.sqrt(2*np.log(2))    
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
        print('\nFit Error')
      
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

# peak = fit_gaussian(spectrum = spectrum, fit_range = [480, 525], plot = True, smooth_first = False, R_sq_thresh=0.1)


#%% Run to here then load in spydata after processing

'''
Run up to here then load in spydata after processing
'''


'''
Running data processing below
'''


#%% No shutter cable

''' Processing data w/o shutter cable'''

group = my_h5['Si_no_shutter_cable_0']


#%% Dark counts


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []

for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    # spectrum.x = linear(spectrum.x)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    # spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 80000)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


## Plot dark counts & smoothed dark counts
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Dark Counts - No Shutter Cable')
for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y, color = 'black')
    ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')


#%% Test dark count subtraction


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    # spectrum.x = linear(spectrum.x)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    # spectrum.truncate(truncate_range[0], truncate_range[1])
    powerseries[i] = spectrum

    
# Plot dark count-subtracted spectrum

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Si Raman - No Shutter Cable')

## Color scale & colorbar
my_cmap = plt.get_cmap('plasma')
cmap = my_cmap
norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
cbar_ax.yaxis.set_label_position("right")
# cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

## Plot
for i, spectrum in enumerate(powerseries):
    color = cmap(norm(spectrum.laser_power))
    ax.plot(spectrum.x, (spectrum.y - dark_powerseries[i].y_smooth)/(spectrum.cycle_time * spectrum.laser_power), color = color, linestyle = ':')
    ax.plot(spectrum.x, spectrum.y/(spectrum.cycle_time * spectrum.laser_power), color = color, linestyle = '-')

## Labelling
solid = ax.plot([], [], color = 'black', linestyle = '-', label = 'Raw')
dotted = ax.plot([], [], color = 'black', linestyle = ':', label = 'Dark Counts Subtracted')
ax.legend()


#%% Plot Si peak


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.calibrate_intensity(dark_counts = dark_powerseries[i].y_smooth,
                                 laser_power = spectrum.laser_power,
                                 exposure = spectrum.cycle_time)
    spectrum.y_baselined = spectrum.y
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 9000, fs = 20000)
    powerseries[i] = spectrum

    
# Plot Si peak region

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Si Raman - No Shutter Cable')

## Color scale & colorbar
my_cmap = plt.get_cmap('plasma')
cmap = my_cmap
norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

## Plot
for i, spectrum in enumerate(powerseries):
    color = cmap(norm(spectrum.laser_power))
    ax.plot(spectrum.x, spectrum.y, color = color, alpha = 1)
    # ax.plot(spectrum.x, spectrum.y_smooth, color = color, alpha = 0.3, linestyle = 0.3)
    

ax.set_xlim(480, 525)


#%% Fit Si peak


for spectrum in powerseries:
    
    si_peak = fit_gaussian(spectrum = spectrum, fit_range = [455, 550], plot = True, smooth_first = False, R_sq_thresh=0.1, peak_name = 'Si')
    spectrum.peak = si_peak


#%% Plot Si peak stats

# Get peak attributes from all spectra in one array

heights = []
areas = []
baselines = []
widths = []
powers = []
times = []

for i, spectrum in enumerate(powerseries):
    heights.append(spectrum.peak.height)
    areas.append(spectrum.peak.area)
    baselines.append(spectrum.peak.baseline)
    widths.append(spectrum.peak.width)
    powers.append(spectrum.laser_power)
    times.append(spectrum.cycle_time)

    
# Plot

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Cycle Time (s)')
ax.set_ylabel('Si Peak Area')
ax.set_title('Si Raman - No Shutter Cable')
# ax2 = ax.twinx()
 
## Color scale & colorbar
# my_cmap = plt.get_cmap('plasma')
# cmap = my_cmap
# norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
# cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
# fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
# cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
# cbar_ax.yaxis.set_label_position("right")
# cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
# cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

# ax2.plot(times, baselines)
ax.plot(times, areas)
    

#%% No shutter powerseries

no_shutter_powerseries = deepcopy(powerseries)

#%% With shutter cable

''' Processing data with shutter cable'''

group = my_h5['Si_with_shutter_cable_0']


#%% Dark counts


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []

for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    # spectrum.x = linear(spectrum.x)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    # spectrum.truncate(truncate_range[0], truncate_range[1])
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1000, fs = 80000)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


## Plot dark counts & smoothed dark counts
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Dark Counts - With Shutter Cable')
for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y, color = 'black')
    ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')


#%% Test dark count subtraction


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    # spectrum.x = linear(spectrum.x)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    # spectrum.truncate(truncate_range[0], truncate_range[1])
    powerseries[i] = spectrum

    
# Plot dark count-subtracted spectrum

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Si Raman - With Shutter Cable')

## Color scale & colorbar
my_cmap = plt.get_cmap('plasma')
cmap = my_cmap
norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
cbar_ax.yaxis.set_label_position("right")
# cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

## Plot
for i, spectrum in enumerate(powerseries):
    color = cmap(norm(spectrum.laser_power))
    ax.plot(spectrum.x, (spectrum.y - dark_powerseries[i].y_smooth)/(spectrum.cycle_time * spectrum.laser_power), color = color, linestyle = ':')
    ax.plot(spectrum.x, spectrum.y/(spectrum.cycle_time * spectrum.laser_power), color = color, linestyle = '-')

## Labelling
solid = ax.plot([], [], color = 'black', linestyle = '-', label = 'Raw')
dotted = ax.plot([], [], color = 'black', linestyle = ':', label = 'Dark Counts Subtracted')
ax.legend()


#%% Plot Si peak


# Add all SERS spectra to powerseries list in order

particle = group
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.__dict__.pop('dset')
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.calibrate_intensity(dark_counts = dark_powerseries[i].y_smooth,
                                 laser_power = spectrum.laser_power,
                                 exposure = spectrum.cycle_time)
    spectrum.y_baselined = spectrum.y
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 9000, fs = 20000)
    powerseries[i] = spectrum

    
# Plot Si peak region

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Raman Intensity (cts/mW/s)')
ax.set_title('Si Raman - With Shutter Cable')

## Color scale & colorbar
my_cmap = plt.get_cmap('plasma')
cmap = my_cmap
norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

## Plot
for i, spectrum in enumerate(powerseries):
    color = cmap(norm(spectrum.laser_power))
    ax.plot(spectrum.x, spectrum.y, color = color, alpha = 1)
    # ax.plot(spectrum.x, spectrum.y_smooth, color = color, alpha = 0.3, linestyle = 0.3)
    

ax.set_xlim(480, 525)


#%% Fit Si peak


for spectrum in powerseries:
    
    si_peak = fit_gaussian(spectrum = spectrum, fit_range = [455, 550], plot = True, smooth_first = False, R_sq_thresh=0.1, peak_name = 'Si')
    spectrum.peak = si_peak


#%% Plot Si peak stats

# Get peak attributes from all spectra in one array

heights = []
areas = []
baselines = []
widths = []
powers = []
times = []

for i, spectrum in enumerate(powerseries):
    heights.append(spectrum.peak.height)
    areas.append(spectrum.peak.area)
    baselines.append(spectrum.peak.baseline)
    widths.append(spectrum.peak.width)
    powers.append(spectrum.laser_power)
    times.append(spectrum.cycle_time)

    
# Plot

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Cycle Time (s)')
ax.set_ylabel('Si Peak Area')
ax.set_title('Si Raman - With Shutter Cable')
# ax2 = ax.twinx()
 
## Color scale & colorbar
# my_cmap = plt.get_cmap('plasma')
# cmap = my_cmap
# norm = mpl.colors.LogNorm(vmin = 0.01, vmax = 3)
# cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
# fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
# cbar_ax.set_ylabel('Laser Power (mW)', rotation=270, labelpad = 20)
# cbar_ax.yaxis.set_label_position("right")
# cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
# cbar_ax.tick_params(which = 'minor', length = 4, width = 1)

# ax2.plot(times, baselines)
ax.plot(times, areas)
    

#%% With shutter powerseries

with_shutter_powerseries = deepcopy(powerseries)


#%% Plot shutter and no shutter comparison


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Cycle Time (s)')
ax.set_ylabel('Si Peak Area')
ax.set_title('Si Raman - Kymera Shutter Test')
ax2 = ax.twinx()
ax2.set_ylabel('Si Peak Baseline (cts/mW/s)', rotation = 270, labelpad = 20)

# No shutter

## Get peak attributes from all spectra in one array
heights = []
areas = []
baselines = []
widths = []
powers = []
times = []

for i, spectrum in enumerate(no_shutter_powerseries):
    heights.append(spectrum.peak.height)
    areas.append(spectrum.peak.area)
    baselines.append(spectrum.peak.baseline)
    widths.append(spectrum.peak.width)
    powers.append(spectrum.laser_power)
    times.append(spectrum.cycle_time)
    
## Plot
ax.plot(times, areas, color = 'blue', label = 'No shutter')
ax2.plot(times, baselines, color = 'blue', linestyle = ':', label = 'No shutter')
areas_no = deepcopy(areas)

# With shutter

## Get peak attributes from all spectra in one array
heights = []
areas = []
baselines = []
widths = []
powers = []
times = []

for i, spectrum in enumerate(with_shutter_powerseries):
    heights.append(spectrum.peak.height)
    areas.append(spectrum.peak.area)
    baselines.append(spectrum.peak.baseline)
    widths.append(spectrum.peak.width)
    powers.append(spectrum.laser_power)
    times.append(spectrum.cycle_time)
    
## Plot
ax.plot(times, areas, color = 'orange', label = 'With shutter')
ax2.plot(times, baselines, color = 'orange', linestyle = ':', label = 'With shutter')
areas_w = deepcopy(areas)    

ax.legend()
ax2.legend(loc = 'upper right')


#%% Testing BX-60 fit of laser powers

## Fit function from experimental script
from scipy import interpolate
def power_to_param2(power, spectrum, kind = 'cubic'):
    params = spectrum.x
    powers = spectrum.y
    curve = interpolate.interp1d(powers, params, kind=kind)
    return curve(power)


# High power

## Get power calibration curves as spectra
group = my_h5['PowerControl']['power control_0']
spectrum = group['powers']
x = spectrum.attrs['parameters']
spectrum = SERS.SERS_Spectrum(x = x, y = np.array(spectrum))
spectrum.__dict__.pop('dset')
spectrum.y_baselined = spectrum.y
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 9000, fs = 20000)
powers0 = deepcopy(spectrum)

## Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Filter wheel position')
ax.set_ylabel('Laser power (mW)')
ax.set_title('Power calibration curves (high power)')

## Plot raw calibration curve
ax.plot(powers0.x, powers0.y, label = 'Raw calibration curve')

## Plot filter wheel values from param_to_power2 v. laser powers used in powerseries for cubic and linear
for spectrum in no_shutter_powerseries:
    if spectrum.filter_slider == 0:
        ax.scatter(power_to_param2(spectrum.laser_power, powers0, kind = 'cubic'), spectrum.laser_power, s = 30, color = 'red', zorder = 2, label = 'CUBIC Powers from fit during expt')
        ax.scatter(power_to_param2(spectrum.laser_power, powers0, kind = 'linear'), spectrum.laser_power, s = 30, color = 'black', zorder = 2, label = 'LINEAR Powers from fit during expt')

handles = ax.get_legend_handles_labels()[0][0:3]
labels = ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels)


# Low power

## Get power calibration curves as spectra
group = my_h5['PowerControl']['power control_1']
spectrum = group['powers']
x = spectrum.attrs['parameters']
spectrum = SERS.SERS_Spectrum(x = x, y = np.array(spectrum))
spectrum.__dict__.pop('dset')
spectrum.y_baselined = spectrum.y
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 9000, fs = 20000)
powers1 = deepcopy(spectrum)

## Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Filter wheel position')
ax.set_ylabel('Laser power (mW)')
ax.set_title('Power calibration curves (low power)')

## Plot raw calibration curve
ax.plot(powers1.x, powers1.y, label = 'Raw calibration curve')

## Plot filter wheel values from param_to_power2 v. laser powers used in powerseries for cubic and linear
for spectrum in no_shutter_powerseries:
    if spectrum.filter_slider == 1:
        ax.scatter(power_to_param2(spectrum.laser_power, powers1, kind = 'cubic'), spectrum.laser_power, s = 30, color = 'red', zorder = 2, label = 'CUBIC Powers from fit during expt')
        ax.scatter(power_to_param2(spectrum.laser_power, powers1, kind = 'linear'), spectrum.laser_power, s = 30, color = 'black', zorder = 2, label = 'LINEAR Powers from fit during expt')

handles = ax.get_legend_handles_labels()[0][0:3]
labels = ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels)


#%% Plot difference in filter_wheel position for linear and cubic fit v. cycle time

## Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Cycle time (s)')
ax.set_ylabel('$\Delta$ Filter wheel position (Cubic - Linear)')
ax.set_title('Difference in power calibration fits')

for spectrum in no_shutter_powerseries:
    if spectrum.filter_slider == 0:
        cubic = power_to_param2(spectrum.laser_power, powers0, kind = 'cubic')
        linear = power_to_param2(spectrum.laser_power, powers0, kind = 'linear')
    if spectrum.filter_slider == 1:
        cubic = power_to_param2(spectrum.laser_power, powers1, kind = 'cubic')
        linear = power_to_param2(spectrum.laser_power, powers1, kind = 'linear')
    ax.scatter(spectrum.cycle_time, cubic - linear, s = 30, color = 'black', zorder = 2, label = 'CUBIC Powers from fit during expt')



#%% Try plotting actual filter_wheel position v. intended powerseries position


## Get target filter wheel positions
powers = []
for spectrum in no_shutter_powerseries:
    powers.append(spectrum.laser_power)
num_powers = len(powers)
wheel_positions = np.empty(num_powers)

for i, power in enumerate(powers):
    try:
        wheel_positions[i] = power_to_param2(power, powers1, kind = 'linear')
    except:
        wheel_positions[i] = power_to_param2(power, powers0, kind = 'linear')
print('Wheel positions: ')
print(wheel_positions)

## Plot
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Cycle time (s)')
ax.set_ylabel('$\Delta$ Filter wheel position (Actual - Target)')
ax.set_title('Wheel position accuracy')
ax2 = ax.twinx()
ax2.set_ylabel('Si Peak Area', rotation = 270, labelpad = 20)

for i, spectrum in enumerate(no_shutter_powerseries):
    ax.scatter(spectrum.cycle_time, spectrum.filter_wheel - wheel_positions[i], color = 'blue', label = 'No shutter')
ax2.plot(times, areas_no, color = 'blue', label = 'No shutter')
    
for i, spectrum in enumerate(with_shutter_powerseries):
    ax.scatter(spectrum.cycle_time, spectrum.filter_wheel - wheel_positions[i], color = 'orange', label = 'With shutter')
ax2.plot(times, areas_w, color = 'orange', label = 'With shutter')        
        
handles = [ax.get_legend_handles_labels()[0][0], ax.get_legend_handles_labels()[0][10]]
labels = [ax.get_legend_handles_labels()[1][0], ax.get_legend_handles_labels()[1][10]]
ax.legend(handles, labels, loc = 'upper left')
ax2.legend(loc = 'upper right')

ylim = ax.get_ylim()
ax.set_ylim(ylim[1], ylim[0])