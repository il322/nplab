# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:11:15 2024

@author: il322


Plotter for testing calibration with neon & BPT ref and getting Co-TAPP-SMe peak positions

Data: 2024-07-29_Co-TAPP-SMe_MLAgg_EChem_SERS_633nm.h5


(samples:
     2024-07-22_Co-TAPP-SMe_60nm_MLAgg_on_ITO_c)

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

from nplab import datafile
from nplab.utils.array_with_attrs import ArrayWithAttrs
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

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-29_Co-TAPP-SMe_MLAgg_EChem_SERS_633nm.h5")

## Save h5 file
# save_h5 = datafile.current(create_if_none=True)

#%% Spectral calibration

# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = my_h5['ref_meas_0']['BPT_633nm_0']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
coarse_shift = 100 # coarse shift to ref spectrum
coarse_stretch = 1 # coarse stretch to ref spectrum
notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 50, None] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

## Convert to wn
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 632.8)
bpt_ref.x = bpt_ref.x + coarse_shift
bpt_ref.x = bpt_ref.x * coarse_stretch

## No notch spectrum (use this truncation for all spectra!)
bpt_ref_no_notch = bpt_ref
bpt_ref_no_notch.truncate(start_x = truncate_range[0], end_x = truncate_range[1])

# Baseline, smooth, and normalize no notch ref for peak finding
bpt_ref_no_notch.y_baselined = bpt_ref_no_notch.y -  spt.baseline_als(y=bpt_ref_no_notch.y,lam=1e1,p=1e-4,niter=1000)
bpt_ref_no_notch.y_smooth = spt.butter_lowpass_filt_filt(bpt_ref_no_notch.y_baselined,
                                                        cutoff=1000,
                                                        fs = 11000,
                                                        order=2)
bpt_ref_no_notch.normalise(norm_y = bpt_ref_no_notch.y_smooth)

## Find BPT ref peaks
ref_wn = cal.find_ref_peaks(bpt_ref_no_notch, lit_spectrum = lit_spectrum, lit_wn = lit_wn, threshold = 0.07, distance = 1)

# ref_wn[3] = bpt_ref_no_notch.x[371]

## Find calibrated wavenumbers
wn_cal = cal.calibrate_spectrum(bpt_ref_no_notch, ref_wn, lit_spectrum = lit_spectrum, lit_wn = lit_wn, linewidth = 1, deg = 3)
bpt_ref.x = wn_cal


# Plot BPT lit and ref

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Counts')
ax.set_title('BPT NPoM Calibration')
ax.plot(lit_spectrum.x, lit_spectrum.y_norm, color = 'black', label = 'BPT lit')
ax.plot(bpt_ref.x, bpt_ref.y_norm, color = 'blue', label = 'BPT meas calibrated')
ax.legend()


# Confirm calibration with neon peak positions

neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
neon_ref.x = spt.wl_to_wn(neon_ref.x, 632.8)
neon_ref.x = neon_ref.x + coarse_shift
neon_ref.x = neon_ref.x * coarse_stretch
neon_ref.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
neon_ref.x = wn_cal
neon_ref.wls = spt.wn_to_wl(neon_ref.x, 632.8)
neon_ref.normalise()

## Neon wavelengths taken from http://www.astrosurf.com/buil/us/spe2/calib2/neon1.gif
neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
neon_wns = spt.wl_to_wn(neon_wls, 632.8)
neon_ys = []
for wn in neon_wns:
    i = np.where(np.round(neon_ref.x) == np.round(wn))
    print(i[0])
    neon_ys.append(neon_ref.y_norm[i[0]])

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Counts')
ax.set_title('BPT Calibration applied to neon lamp')

# ax.plot(bpt_ref.x, bpt_ref.y_norm, color = 'blue', alpha = 0.7, label = 'BPT meas')
# ax.plot(lit_spectrum.x, lit_spectrum.y_norm, color = 'black', alpha = 0.7, label = 'BPT lit', zorder = 0)
ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas calibrated')
ax.scatter(neon_wls, np.zeros(neon_wns.shape) + 0.6, color = 'black', zorder = 10)
ax.vlines(neon_wls, ymin = 0, ymax = 0.6, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks')
ax.set_xlim(640,None)
ax.legend()

## Not great

#%% Try calibrating from neon then confirming with BPT

## Get measured (ref) neon and peak positions
neon_ref = my_h5['ref_meas_0']['neon_lamp_0']
neon_ref = SERS.SERS_Spectrum(neon_ref)
## neon_wls = literature neon
### Neon wavelengths taken from http://www.astrosurf.com/buil/us/spe2/calib2/neon1.gif
neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
neon_ref.normalise()
neon_ref_peaks = neon_ref.x[spt.detect_maxima(neon_ref.y_norm, lower_threshold = 0.02)]

## Cut peaks out of desired wavelength range
neon_wls = neon_wls[15:]
neon_ref_peaks = neon_ref_peaks[0:-4]

## cut detected peaks that are too close together
delete_list = []
for i, peak in enumerate(neon_ref_peaks):
    if i < len(neon_ref_peaks)-1:    
        
        if neon_ref_peaks[i+1] - neon_ref_peaks[i] < 2:
            x = np.argmin((neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i])], neon_ref.y[np.where(neon_ref.x == neon_ref_peaks[i+1])]))
            delete_list.append(x+i)
neon_ref_peaks = np.delete(neon_ref_peaks, delete_list)        

## plot raw neon ref, raw neon ref peaks, neon lit peak positions
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_title('Lab 1 Neon lamp ref')
ax.set_xlabel('Wavelengths (nm)')
ax.set_ylabel('Counts')
ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
ax.scatter(neon_ref_peaks, np.zeros(neon_ref_peaks.shape) + 0.6, color = 'red', zorder = 10, label = 'neon meas peaks raw')
ax.scatter(neon_wls, np.zeros(neon_wls.shape) + 0.6, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks')
ax.set_xlim(630,770)
ax.legend()

## fit neon ref peaks and neon lit peaks
def linear(x, m, b):
    return (m*x) + b

def quadratic(x, a, b, c):
    return (a*x**2 + b*x +c)

popt, pcov = curve_fit(f = linear, 
                    xdata = neon_ref_peaks,
                    ydata = neon_wls)

neon_ref_cal = linear(neon_ref.x, *popt)

## plot neon ref peaks v neon lit peaks and fitted calibration curve
plt.figure(figsize=(12,9), dpi=300)
plt.plot(neon_wls, neon_ref_peaks, '.')
plt.plot(neon_wls, linear(neon_ref_peaks, *popt), '-')
plt.xlabel('Neon wavelengths - Literature')
plt.ylabel('Neon wavelengths - Measured')
# plt.figtext(0.5,0.3,'R$^{2}$: ' + str(R_sq))
plt.tight_layout()
plt.show()  

## thorlabs neon spectrum as extra lit neon confirmation
thorlabs_neon = np.genfromtxt(r"C:\Users\il322\Desktop\Offline Data\Thorlabs_Neon_Spectrum.csv", delimiter = ',')
thorlabs_neon = SERS.SERS_Spectrum(thorlabs_neon[:,0], thorlabs_neon[:,1])

## plot calibrated neon ref, raw neon ref, thorlabs neon lit, and lit peak position
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelengths (nm)')
ax.set_ylabel('Counts')
ax.set_title('Neon lamp calibration')
# ax.plot(bpt_ref.x, bpt_ref.y_norm, color = 'blue', alpha = 0.7, label = 'BPT meas')
# ax.plot(lit_spectrum.x, lit_spectrum.y_norm, color = 'black', alpha = 0.7, label = 'BPT lit', zorder = 0)
ax.plot(neon_ref_cal, neon_ref.y_norm, color = 'orange', label = 'neon meas fit')
ax.plot(neon_ref.x, neon_ref.y_norm, color = 'red', label = 'neon meas raw')
# ax.scatter(neon_ref_peaks, np.zeros(neon_ref_peaks.shape) + 0.6, color = 'red', zorder = 10, label = 'neon ref peaks raw')
ax.scatter(neon_wls, np.zeros(neon_wls.shape) + 0.6, color = 'black', zorder = 10, label = 'neon lit (astrosurf.com) peaks')
ax.scatter(thorlabs_neon.x, thorlabs_neon.y, color = 'blue', label = 'thorlabs neon peaks', zorder = 10)
ax.set_xlim(700,710)
ax.legend()

## Plot calibrated BPT to test
lit_spectrum = np.genfromtxt(r"C:\Users\il322\Desktop\Offline Data\nanocavity_spectrum_bpt.csv", delimiter = ',', skip_header=1)
lit_spectrum = SERS.SERS_Spectrum(lit_spectrum[:,1], lit_spectrum[:,0])
# lit_spectrum.y = lit_spectrum.y - spt.baseline_als(lit_spectrum.y, lam=1e2, p=1e-4)
lit_spectrum.normalise()
# lit_spectrum.x = spt.wn_to_wl(lit_spectrum.x, 632.8)
bpt_ref = my_h5['ref_meas_0']['BPT_633nm_0']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)
bpt_ref.x = linear(bpt_ref.x, *popt)
bpt_ref.normalise()
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 632.8)
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_title('Neon Lamp Calibration applied to BPT')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Counts')
ax.plot(spt.wl_to_wn(bpt_ref.x_raw, 632.8), bpt_ref.y_norm, color = 'red', alpha = 0.7, label = 'BPT meas raw')
ax.plot(bpt_ref.x, bpt_ref.y_norm, color = 'blue', alpha = 0.7, label = 'BPT meas calibrated')
ax.plot(lit_spectrum.x, lit_spectrum.y_norm, color = 'black', alpha = 0.7, label = 'BPT lit', zorder = 0)
# ax.set_xlim(950, 1650)
# ax.plot(neon_ref.wls, neon_ref.y_norm, color = 'orange', label = 'neon meas')
# ax.scatter(neon_wls, np.zeros(neon_wns.shape) + 0.6, color = 'black', zorder = 10)
# ax.set_xlim(640,None)
ax.legend()

cal_matrix = popt
truncate_range = [250, None]

# bpt_ref.truncate(truncate_range[0], truncate_range[1])
bpt_ref.calibrated_wn = bpt_ref.x
## Much better to use neon!!


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas_0']['white_ref_x5_0']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = linear(white_ref.x, *cal_matrix)
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)

## Get white bkg (counts in notch region)
notch_range = [155, 175]
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

## Get dark counts - skip for now as using powerseries
# dark_cts = my_h5['PT_lab']['whire_ref_x5']
# dark_cts = SERS.SERS_Spectrum(wn_cal_633, dark_cts[5], title = 'Dark Counts')
# # dark_cts.plot()
# plt.show()

''' 
Still issue with 'white background' of calculating R_setup
Right now, choosing white background ~400 (near notch counts) causes R_setup to be very low at long wavelengths (>900nm)
This causes very large background past 1560cm-1 BPT peak
Using a white_bkg of -100000 flattens it out...
'''    

#%% Use neon calibration to plot Co-TAPP-SMe MLAgg and get peak positions


## Dark counts
dark = my_h5['ref_meas_0']['633_glass_1s_1uW_0']
dark = SERS.SERS_Spectrum(dark)
dark.x = linear(dark.x, *cal_matrix)
dark.x = spt.wl_to_wn(dark.x, 632.8)
dark.truncate(truncate_range[0], truncate_range[1])

## Co-TAPP-SMe spectrum
spectrum = my_h5['ref_meas_0']['Co-TAPP-SMe_Air_1']
spectrum = SERS.SERS_Spectrum(spectrum)
spectrum.x = linear(spectrum.x, *cal_matrix)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.truncate(truncate_range[0], truncate_range[1])
spectrum.calibrate_intensity(R_setup = R_setup, 
                              dark_counts = dark.y)

## Further processing
spectrum.calibrated_wn = spectrum.x
spectrum.truncate(252, 2200)
spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
spectrum.y_baselined = spectrum.y - spectrum.baseline
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 15000)
maxima = spt.detect_maxima(spectrum.y_smooth, lower_threshold = 3000)
peaks = spectrum.x[maxima]
spectrum.peaks = peaks


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.set_title('Co-TAPP-SMe_MLAgg 633 nm SERS 0.5 uW')
# ax.plot(spectrum.x, spectrum.y_baselined + 20000, color = 'blue')
ax.plot(spectrum.x, spectrum.y_smooth, color = 'red')
ax.scatter(spectrum.x[maxima], spectrum.y_smooth[maxima], s = 100)
ax.set_xlim(None, 2100)
# ax.legend()


#%% Saving for calibrating new plotters

# save_h5.create_dataset(name = 'Co-TAPP-SMe MLAgg 633nm SERS Calibrated', data = spectrum.y, attrs = spectrum.__dict__)
# save_h5.create_dataset(name = 'BPT NPoM 633nm SERS Calibrated', data = bpt_ref.y, attrs = bpt_ref.__dict__)
# save_h5.create_dataset(name = 'ThorLabs Neon Spectrum', data = thorlabs_neon.y, attrs = thorlabs_neon.__dict__)
# neon_wls = np.array([585.249, 588.189, 594.483, 597.553, 603, 607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 692.947, 703.241, 717.394, 724.517, 743.89])
# save_h5.create_dataset(name = 'Astrosurf Neon Spectrum', data = neon_wls, attrs = thorlabs_neon.__dict__)

