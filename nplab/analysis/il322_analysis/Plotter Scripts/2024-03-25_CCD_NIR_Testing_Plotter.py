# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for NIR white scatter spectra with different optics


Data: 2024-03-18_785_Objective_Test.h5
      2024-03-25_CCD_NIR_Testing.h5
      2024-03-26_785nm_Powerseries_BPT_MLAgg.h5


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
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.stats import norm
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



#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-18_785_Objective_Test.h5")
my_h52 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-25_CCD_NIR_Testing.h5")
my_h53 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-26_785nm_Powerseries_BPT_MLAgg.h5")



#%% Plot white_scatt for new and old mirror

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Intensity (cts/s)', size = 'large')
    
spectrum = my_h52['AndorData']['white_scatt_BF_Grating4_850cnwln_200cn_15rows_-60C_0.001s_0']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum, label = 'New mirror')
ax.plot(spectrum.x, spectrum.y/0.001, label = 'Old mirror; BF; No Optics; -60C')

spectrum = my_h52['AndorData']['white_scatt_BF_Grating4_850cnwln_200cn_15rows_-60C_LensA_Lab8CCD_new_mirror_0.001s_0']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum)
ax.plot(spectrum.x, spectrum.y/0.001, label = 'New mirror; BF; No Optics; -60C')

spectrum = my_h53['ref_meas']['white_scatt_785']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum[2])
ax.plot(spectrum.x, spectrum.y/0.005, label = 'New mirror; DF; 20x 0.4NA; NIR BS; -90C')


ax.legend()
fig.suptitle('Lab 1 NIR Test - White Scatterer', fontsize = 'large')
plt.tight_layout()

# Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-25 Analysis\_'
# plt.savefig(save_dir + 'White Scatterer 785nm Optics Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot normalized white_scatt for new and old mirror

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Normalized Intensity', size = 'large')
    
spectrum = my_h52['AndorData']['white_scatt_BF_Grating4_850cnwln_200cn_15rows_-60C_0.001s_0']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum, label = 'New mirror')
ax.plot(spectrum.x, spectrum.y/spectrum.y.max(), label = 'Old mirror; BF; No Optics; -60C')

spectrum = my_h52['AndorData']['white_scatt_BF_Grating4_850cnwln_200cn_15rows_-60C_LensA_Lab8CCD_new_mirror_0.001s_0']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum)
ax.plot(spectrum.x, spectrum.y/spectrum.y.max(), label = 'New mirror; BF; No Optics; -60C')

spectrum = my_h53['ref_meas']['white_scatt_785']
spectrum = SERS.SERS_Spectrum(spectrum.attrs['wavelengths'], spectrum[2])
ax.plot(spectrum.x, spectrum.y/spectrum.y.max(), label = 'New mirror; DF; 20x 0.4NA; NIR BS; -90C')


ax.legend()
fig.suptitle('Lab 1 NIR Test - White Scatterer', fontsize = 'large')
plt.tight_layout()

# Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-25 Analysis\_'
# plt.savefig(save_dir + 'Normalized White Scatterer 785nm Optics Test' + '.svg', format = 'svg')
# plt.close(fig)
