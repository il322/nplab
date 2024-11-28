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

Copied from 2024-08-28_MLAgg_Photocurrent_Plotter.py

Data: 2024-09_04_TPDT_MLAggs_FTO

(samples: 2024-09_04_TPDT_MLAggs_FTO)

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
save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-09-04_Processed_TPDT_PEC_Data.h5")

## PEC h5 Files (one per MLAgg)

h5_20 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-06_TPDT_20nm_MLAgg_Photocurrent.h5")
h5_35 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-08_TPDT_35nm_MLAgg_Photocurrent.h5")

h5_dict = {20 : h5_20, 35 : h5_35}


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-09-04_TPDT_MLAgg_FTO Photocurrent Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def cottrell(x, a, b, c):
    
    return (a/(sqrt(pi*x + c))) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c

def reciprocal(x, a, b, c):
    
    return (a/(x+b)) + c

def linear(x, a, b):
    
    return a*x + b
    

#%% Darkfield

#%% DF data groups

df_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-05_TPDT_MLAgg_DF_SERS_Map.h5")

df_dict = {20 : "/ref_meas_0/TPDT_Strath20nm_MLAgg_SERS_map_0", 
           35 : "/ref_meas_0/TPDT_Strath20nm_MLAgg_SERS_map_0/grid_9_17/TPDT_Strath35nm_MLAgg_SERS_map_0", 
           57 : "/PT_lab/TPDT_Strath57nm_MLAgg_SERS_map_1", 
           70 : "/PT_lab/TPDT_Strath70nm_MLAgg_SERS_map_0", 
           80 : "/PT_lab/TPDT_Strath80nm_MLAgg_SERS_map_0"}

sizes = [20, 35, 57, 70, 80]


#%% Get all DF spectra


df_spectra_dict = {20 : [], 
                   35 : [], 
                   57 : [], 
                   70 : [], 
                   80 : []}

df_avg_dict = {20 : [], 
                   35 : [], 
                   57 : [], 
                   70 : [], 
                   80 : []}

for size in sizes:
    
    df_spectra = df_spectra_dict[size]
    df_avg = df_avg_dict[size]
    scan = df_h5[df_dict[size]]
    
    for key in list(scan.keys()):
        
        group = scan[key]
        
        for key in list(group.keys()):

            if 'spec' in key:
                spectrum = group[key]
                spectrum = df.DF_Spectrum(spectrum)
                spectrum.y = (spectrum.y - spectrum.background)/(spectrum.reference - spectrum.background)
                spectrum.truncate(400, 900)
                if spectrum.y.min() < 0:
                    continue
                if spectrum.y[0] > 0.05:
                    print(key)
                df_spectra.append(spectrum)
                df_avg.append(spectrum.y)

    df_avg = df.DF_Spectrum(spectrum.x, np.mean(df_avg, axis = 0), smooth_first = False)
    df_avg_dict[size] = df_avg
    

#%% Plot all df spectra

colors = ['blue', 'darkgreen', 'orange', 'red', 'black']

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Darkfield Scattering', fontsize = 'x-large')
fig.suptitle('TPDT MLAgg on FTO Darkfield', fontsize = 'x-large')

for i, size in enumerate(sizes):
    
    df_spectra = df_spectra_dict[size]
    
    # for spectrum in df_spectra[:]:
    #     ax.plot(spectrum.x, spectrum.y, color = colors[i], alpha = 0.01)

    spectrum = df_avg_dict[size]    
    ax.plot(spectrum.x, spectrum.y, color = colors[i], alpha = 1, label = str(size) + ' nm')

ax.legend()

## Save
save = True
if save == True:
    save_dir = get_directory('DF Compiled')
    fig.savefig(save_dir + 'TPDT MLAgg DF' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)

save_to_h5 = True
if save_to_h5 == True:
    group = save_h5

    canvas = FigureCanvas(fig)
    canvas.draw()
    fig_array = np.array(canvas.renderer.buffer_rgba())
    group.create_dataset(name = 'DF_Compiled_jpg_%d', data = fig_array)
    
    
#%% SERS

my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-05_TPDT_MLAgg_DF_SERS_Map.h5")

SERS_dict = {20 : 'BPDT_Strath20nm_MLAgg_SERS_map_0', 
           35 : "/PT_lab/TPDT_Strath35nm_MLAgg_SERS_map_0", 
           57 : "/PT_lab/TPDT_Strath57nm_MLAgg_SERS_map_2", 
           70 : "/PT_lab/TPDT_Strath70nm_MLAgg_SERS_map_1", 
           80 : "/PT_lab/TPDT_Strath80nm_MLAgg_SERS_map_1"}

#%% New spectral calibration using neon calibration and comparing to calibrated spectra from cal_h5

## Truncation for all spectra
truncate_range = [250, 2300]

## Get ref (measured) neon peaks
neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
neon_ref.normalise()
neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.007)]
neon_ref_peaks = neon_ref_peaks[1:-3]
neon_ref_peaks_y = neon_ref.y_norm[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.007)]
neon_ref_peaks_y = neon_ref_peaks_y[1:-3]
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

# cal_matrix[1] -= 0.3
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
notch_range = [150, 165]
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
    

#%% Get all SERS

SERS_spectra_dict = {20 : [], 
                   35 : [], 
                   57 : [], 
                   70 : [], 
                   80 : []}

SERS_avg_dict = {20 : [], 
                   35 : [], 
                   57 : [], 
                   70 : [], 
                   80 : []}

for size in sizes:
    if size == 20:
        continue
    
    SERS_spectra = SERS_spectra_dict[size]
    SERS_avg = SERS_avg_dict[size]
    scan = my_h5[SERS_dict[size]]
    
    for key in list(scan.keys()):
        
        group = scan[key]
        
        for key in list(group.keys()):

            if 'SERS' in key:
                spectrum = group[key]
                spectrum = SERS.SERS_Timescan(spectrum)
                
                ## Calibrate x-axis
                spectrum.x = linear(spectrum.x, *cal_matrix)
                spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
                spectrum.truncate(truncate_range[0], truncate_range[1])
                
                ## Calibrate intensity
                spectrum.calibrate_intensity(R_setup = 1,
                                              dark_counts = 0,
                                              exposure = spectrum.cycle_time,
                                              laser_power = spectrum.laser_power)
                
                spectrum.y = np.mean(spectrum.Y, axis = 0)
                
                
                SERS_spectra.append(spectrum)
                SERS_avg.append(spectrum.y)

    SERS_avg = SERS.SERS_Spectrum(spectrum.x, np.mean(SERS_avg, axis = 0))
    SERS_avg_dict[size] = SERS_avg
    

#%% Plot avg SERS

colors = ['blue', 'darkgreen', 'orange', 'red', 'black']

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'x-large')
ax.set_ylabel('Intensity (cts/mW/s)', fontsize = 'x-large')
fig.suptitle('TPDT MLAgg on FTO 633 nm SERS', fontsize = 'x-large')

for i, size in enumerate(sizes):
    
    if size == 20:
        continue
    
    SERS_spectra = SERS_spectra_dict[size]
    
    spectrum = SERS_avg_dict[size]    
    ax.plot(spectrum.x, spectrum.y, color = colors[i], alpha = 1, label = str(size) + ' nm')

ax.legend()

## Save
save = True
if save == True:
    save_dir = get_directory('SERS Compiled')
    fig.savefig(save_dir + 'TPDT MLAgg SERS' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)

save_to_h5 = True
if save_to_h5 == True:
    group = save_h5

    canvas = FigureCanvas(fig)
    canvas.draw()
    fig_array = np.array(canvas.renderer.buffer_rgba())
    group.create_dataset(name = 'SERS_Compiled_jpg_%d', data = fig_array)


        
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
    df_avg = particle.df_avg
    
    
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
    df_avg = particle.df_avg
    
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
            if spectrum.wavelength > 450:
                current = np.mean(spectrum.pec)
                if current < 0:
                    marker = '_'
                else:
                    marker = '+'
                ipce = (np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * power_dict[spectrum.wavelength]) * 100
                ax.scatter(spectrum.wavelength, ipce, color = color, s = 150, marker = marker, linewidth = 4)
                ipce_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * (10**3 * 1240 * 100 / (spectrum.wavelength * power_dict[spectrum.wavelength])) 
                # ax.errorbar(spectrum.wavelength,  ipce, yerr = ipce_err, marker = 'none', mfc = color, mec = color, linewidth = 1, markersize = 10, capsize = 0, elinewidth = 2, capthick = 0, ecolor = color, zorder = 1)
                # for current in spectrum.pec:
                    # ipce = (np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * power_dict[spectrum.wavelength]) * 100
                    # ax.scatter(spectrum.wavelength, ipce, color = color, s = 150, marker = marker, linewidth = 4)        
        
        ax2 = ax.twinx()
        ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
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
        
        
def plot_ocp_pec(particle, save = False, save_to_h5 = False):
    
    spectra = particle.ocp_spectra
    power_dict = particle.power_dict
    size = particle.size
    df_avg = particle.df_avg    
    
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
        print(spectrum.name)
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
            current = np.mean(spectrum.pec)
            if current < 0:
                marker = '_'
            else:
                marker = '+'
            ipce = (np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * power_dict[spectrum.wavelength]) * 100
            ax.scatter(spectrum.wavelength, ipce, color = color, s = 150, marker = marker, linewidth = 4)
            ipce_err = (np.std(spectrum.pec)/np.sqrt(len(spectrum.pec))) * (10**3 * 1240 * 100 / (spectrum.wavelength * power_dict[spectrum.wavelength])) 
            # ax.errorbar(spectrum.wavelength,  ipce, yerr = ipce_err, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 0, capthick = 2, ecolor = color, zorder = 1)
            # for current in spectrum.pec:
                # ipce = (np.abs(current) * 10**3 * 1240)/(spectrum.wavelength * power_dict[spectrum.wavelength]) * 100
                # ax.scatter(spectrum.wavelength, ipce, color = color, s = 150, marker = marker, linewidth = 4)

    ax2 = ax.twinx()
    ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
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
        
        linetype = 'solid'
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, ((spectrum.y_smooth) * 10**3 + i * offset), color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ## Light on/off bars
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
            current = np.mean(spectrum.pec)
            if current < 0:
                marker = '_'
            else:
                marker = '+'                
            ax.scatter(spectrum.wavelength, np.abs(current) * 10**3/power_dict[spectrum.wavelength], color = color, s = 150, marker = marker, linewidth = 4)
            # for current in spectrum.pec:
                # ax.scatter(spectrum.wavelength, current, color = color, s = 150, marker = marker, linewidth = 4)

    ax2 = ax.twinx()
    ax2.plot(df_avg.x, df_avg.y, color = 'black', alpha = 0.4, zorder = 0)
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


#%% EChem & Photocurrent functions


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


def process_ca(particle):
    
    spectra = particle.ca_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        voltage = voltage[0]
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.x += 1
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = name
        spectrum.__dict__.pop('dset')
        
        ## Baseline, smooth
        spectrum.truncate(20, None)    
        ## Cottrell fit baseline subtraction
        x = spectrum.x
        y = spectrum.y
        
        try:
            popt, pcov = curve_fit(f = cottrell, 
                                xdata = x,
                                ydata = y)
        except:
            # print('\nFit Error')
            # print('\n'+spectrum.name+'\n')
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
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 10 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 10 : dark_index])
            pec.append(light_current - dark_current)
        spectrum.pec = pec
        
        spectra[j] = spectrum
    
    particle.ca_spectra = spectra
    
    
def process_ocp(particle):

    spectra = particle.ocp_spectra ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
        
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.x += 0
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = name
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
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 10 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 10 : dark_index])
            pec.append(light_current - dark_current)
        spectrum.pec = pec
        
        spectra[j] = spectrum
        
    particle.ocp_spectra = spectra
    
    
def process_ocp_voltage(particle):

    spectra = particle.ocp_spectra_voltage ## Should be list of h5 groups with spectra you want
    
    for j, spectrum in enumerate(spectra):
            
        name = str(spectrum)
        name = name[name.find('"')+1:name.find('"', name.find('"')+1)]
        timestamp = spectrum.attrs['creation_timestamp']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1], particle_name = particle.name)
        spectrum.x += 0
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectrum.voltage = 'OCP'
        spectrum.__dict__.pop('dset')
    
        # Baseline, smooth
        # spectrum.truncate(20, None)    
        ## Cottrell fit baseline subtraction
        # x = spectrum.x
        # y = spectrum.y
        
        # try:
        #     popt, pcov = curve_fit(f = cottrell, 
        #                         xdata = x,
        #                         ydata = y)
        # except:
        #     print('\nFit Error')
        #     print('\n'+spectrum.name+'\n')
        #     popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]  

        # spectrum.truncate(100, None)

        # spectrum.y_baseline = cottrell(spectrum.x, *popt)
        spectrum.y_baseline = spt.baseline_als(spectrum.y, lam = 1e7, p = 1e-4, niter = 10)
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
        for i in range(0, toggles):
            dark_index = np.argmin(np.abs(spectrum.x - (spectrum.toggle * (2*i + 1))))
            light_index = np.argmin(np.abs(spectrum.x - (2*spectrum.toggle * (i + 1))))
            light_current = np.mean(spectrum.y_smooth[light_index - 10 : light_index])
            dark_current = np.mean(spectrum.y_smooth[dark_index - 10 : dark_index])
            pec.append(light_current - dark_current)
        spectrum.pec = pec
        
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
    

#%% Photocurrent

particles = []


# 20 nm MLAgg

this_particle = Particle()
this_particle.name = 'TPDT 20 nm MLAgg'
this_particle.size = 20
my_h5 = h5_dict[this_particle.size]
this_particle.df_avg = df_avg_dict[this_particle.size]
this_particle.power_calibration_group = my_h5['power_calibration_1']
skip_spectra = ["PEC_CA_450nm_50nmFWHM_toggle_50s_CA_-0.4V_0",
                "PEC_CA_475nm_50nmFWHM_toggle_50s_CA_-0.4V_0",
                "PEC_CA_850nm_50nmFWHM_toggle_50s_CA_0V_0",
                "PEC_CA_525nm_50nmFWHM_toggle_50s_CA_0V_0"]

## CA Spectra
group = my_h5['TPDT_20nm_MLAgg_0']['Potentiostat']
spectra = []
for key in group.keys():
    if 'PEC_CA' in key and key not in skip_spectra and '-0.4' not in key:        
            spectrum = group[key]
            spectra.append(spectrum)           
group = my_h5['TPDT_20nm_MLAgg_1']
for key in group.keys():
    if 'PEC_CA' in key and key not in skip_spectra:        
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ca_spectra = spectra


## CV Spectra
group = my_h5['TPDT_20nm_MLAgg_1']
spectra = []
for key in group.keys():
    if 'CV' in key and key not in skip_spectra:        
            spectrum = group[key]
            spectra.append(spectrum)    
this_particle.cv_spectra = spectra

## LSV Spectra
group = my_h5['TPDT_20nm_MLAgg_0']['Potentiostat']
spectra = []
for key in group.keys():
    if 'LSV' in key and key not in skip_spectra and 'test' not in key:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.lsv_spectra = spectra

## OCP Current spectra
group = my_h5['TPDT_20nm_MLAgg_1']
spectra = []
for key in group.keys():
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key and 'currents' in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra = spectra
            
## OCP voltage spectra
spectra = []
for key in group.keys():
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key and 'potential' in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra_voltage = spectra

particles.append(this_particle)


# 35 nm MLAgg

this_particle = Particle()
this_particle.name = 'TPDT 35 nm MLAgg'
this_particle.size = 35
my_h5 = h5_dict[this_particle.size]
this_particle.df_avg = df_avg_dict[this_particle.size]
this_particle.power_calibration_group = my_h5['power_calibration_0']
group = my_h5['TPDT_35nm_MLAgg_0']
skip_spectra = ["PEC_CA_450nm_50nmFWHM_toggle_50s_CA_-0.4V_1"]

## CA Spectra
spectra = []
for key in group.keys():
    if 'PEC_CA' in key and key not in skip_spectra and key[-1] != 1:        
            spectrum = group[key]
            spectra.append(spectrum)           
this_particle.ca_spectra = spectra

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
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key and 'currents' in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra = spectra
            
## OCP voltage spectra
spectra = []
for key in group.keys():
    
    if 'OCP' in key and key not in skip_spectra and 'test' not in key and 'potential' in key:
            spectrum = group[key]
            spectra.append(spectrum)
this_particle.ocp_spectra_voltage = spectra

particles.append(this_particle)


save = True

for particle in particles:
    
    process_power_calibration(particle)
    process_ca(particle)
    process_ocp(particle)
    process_ocp_voltage(particle)
    process_cv(particle)
    process_lsv(particle)
    
    if save == True:
        save_ca(particle)
        save_ocp(particle)
        save_cv(particle)
        save_lsv(particle)


for particle in particles:
    plot_ca_pec(particle, save = save, save_to_h5 = save)
    plot_cv(particle, save = save, save_to_h5 = save)
    plot_ocp_pec(particle, save = save, save_to_h5 = save)
    plot_lsv(particle, save = save, save_to_h5 = save)


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

  
