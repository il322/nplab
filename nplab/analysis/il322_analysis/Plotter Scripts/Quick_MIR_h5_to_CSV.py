# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 785nm SERS Kinetic Scan and SERS decay statistics

Data: 2024-03-28_785nm-Powerseries-KineticScan_NIR-Objective_Test.h5


(samples:
     2023-11-28_Co-TAPP-SMe_60nm_MLAgg_c)

"""

import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
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
from scipy.ndimage import gaussian_filter

from nplab import datafile
from nplab.analysis.general_spec_tools import spectrum_tools as spt
from nplab.analysis.general_spec_tools import npom_sers_tools as nst
from nplab.analysis.general_spec_tools import agg_sers_tools as ast
from nplab.analysis.SERS_Fitting import Auto_Fit_Raman as afr
from nplab.analysis.il322_analysis import il322_calibrate_spectrum as cal
from nplab.analysis.il322_analysis import il322_SERS_tools as SERS
from nplab.analysis.il322_analysis import il322_DF_tools as df

from lmfit.models import GaussianModel


#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2022-08-04_Ishaan.h5")
data_file = my_h5

#%% Function for neon calibration

# Input wavelengths from h5
# Output corrected wavelengths for use in plotting/fitting

def neon_calibration(wls_meas, neon_spec_meas, threshold = 0.3):
    
    # Neon wavelengths taken from http://www.astrosurf.com/buil/us/spe2/calib2/neon1.gif
    neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])

    # find peaks in measured neon spectrum
    neon_peaks_meas = find_peaks(neon_spec_meas, height = threshold*np.max(neon_spec_meas), distance = 10)
    wls_neon_peaks_meas = wls_meas[neon_peaks_meas[0]]
    
    neon_wls_list = np.zeros(len(wls_neon_peaks_meas))
    for i, meas_wl in enumerate(wls_neon_peaks_meas):
        neon_wls_list[i] = neon_wls[np.argmin(np.abs(neon_wls - meas_wl))]
        
    #print('measured wls: ' + str(np.round(wls_neon_peaks_meas,2)))
    #print('literature wls: ' + str(np.round(neon_wls_list,2)))

    # fit neon wavelengths (linear)
    slope_offset, wl_offset = np.polyfit(neon_wls_list, wls_neon_peaks_meas, 1, rcond=None, full=False, w=None, cov=False)

    # Calculate coefficienct of determination for neon fit    
    corr_matrix = np.corrcoef(wls_neon_peaks_meas, slope_offset*neon_wls_list + wl_offset)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    
    plt.figure(figsize=(4,3), dpi=300)
    plt.plot(neon_wls_list, wls_neon_peaks_meas, '.')
    plt.plot(neon_wls_list, slope_offset*neon_wls_list + wl_offset, '-')
    plt.xlabel('Neon wavelengths - Literature')
    plt.ylabel('Neon wavelengths - Measured')
    plt.figtext(0.5,0.3,'R$^{2}$: ' + str(R_sq))
    plt.tight_layout()
    plt.show()    
    
    #return R_sq
    return (wls_meas - wl_offset)/slope_offset

#%% White light correction

# Input white scatter measurement from h5
# Output R_setup matrix -> divide data intensities by R_setup to correct for spectrometer efficiency

white_scatter = data_file['reference_meas']['white_ref_0.01']
white_scatter_bg = 305
S_whitescatter = np.array(white_scatter) - np.array(white_scatter_bg)
S_whitescatter[755:823] = S_whitescatter[830] #remove zeros of notch filter
S_dkfd = np.loadtxt(r"S:\il322\MPhil\Python Scripts\Lab 2 4NTP Upconversion\spectralEfficiency.csv",delimiter=',')
spline = sp.interpolate.splrep(S_dkfd[...,0],S_dkfd[...,1], s=0)
S_dkfd_spline = sp.interpolate.splev(white_scatter.attrs['wavelengths'], spline, der=0)
R_setup = S_whitescatter/np.array(S_dkfd_spline)
R_setup = R_setup/max(R_setup) 


#%% Some measurement settings for calibration

neon_spectrum = data_file['reference_meas']['neon_ref_0.01']
laser_wavelength = 783.25
power = 1 # power in uW
integration_time = 10 # integration time in s


#%% Plot MIR toggle delay scan
#laser_ref = data_file['reference_meas']['785_laser_ref_5s']
group = 'CaF2_3um_NPoR_0_delayscan_MIRToggle_1uwVIS_0.0MIR_785_10s_1000ss'
item_list = list(data_file[group].items()) # list of data items (spectra, notes, snapshots) in target group as tuples 
item_list = [item[0] for item in item_list] # list of data item names (spectra, notes, snapshots) in target group as strings   
#print(item_list)

NPoR_delayscan = data_file[group]
particle = NPoR_delayscan

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []
for key in keys:
    if 'snapshot' not in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.laser_power = .001
    spectrum.cycle_time = 10
    # spectrum.x = spt.wl_to_wn(spectrum.x, 633)
    # spectrum.x = spectrum.x + coarse_shift
    # spectrum.x = spectrum.x * coarse_stretch
    # spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    Raman_shift = spt.wl_to_wn(spectrum.x, 785) + 200
    spectrum.x = Raman_shift
    spectrum.truncate(-1800, 1800)
    # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    spectrum.calibrate_intensity(R_setup = 1, dark_counts = 0, laser_power = 0.001, exposure = 10)
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    powerseries[i] = spectrum

particle.powerseries = powerseries

def plot_timescan_powerseries(particle, save = False):
    
    
    powerseries = particle.powerseries
    powerseries_y = particle.powerseries_y

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig3, (ax3) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = powerseries_y.min()
    v_max = np.percentile(powerseries_y, 99.9) + 5000
    cmap = plt.get_cmap('inferno')
    ax3.set_yticklabels([])
    ax3.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    # ax3.set_xlim(-1800,1800)
    ax3.set_title('MIR Upconversion NPoR Delay Scan (MIR On/Off)', fontsize = 'x-large', pad = 10)
    pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig3.colorbar(pcm, ax=ax3)
    clb.set_label(label = 'SERS Intensity (cts/mW/s)', size = 'large', rotation = 270, labelpad=30)
    ax3.set_ylabel('Pulse Delay (ps)', size = 'large')
    ax3.set_yticks(np.linspace(0, 100, 10))
    ax3.set_yticklabels(np.round(np.linspace((1000 *(2/7500) * -25), (1000 *(2/7500) * 25), 10)))
    
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig3.savefig(save_dir + particle.name + '633nm Powerswitch Timescan' + '.svg', format = 'svg')
        plt.close(fig)

powerseries_y = np.zeros((len(particle.powerseries), len(particle.powerseries[0].y)))
for i,spectrum in enumerate(particle.powerseries):
    powerseries_y[i] = spectrum.y_baselined
particle.powerseries_y = np.array(powerseries_y)
plot_timescan_powerseries(particle)

#%%

powerseries_y = particle.powerseries_y
np.savetxt(fname = r"C:\Users\il322\Desktop\Offline Data\NPoR_data.csv", X = powerseries_y, delimiter = ',')