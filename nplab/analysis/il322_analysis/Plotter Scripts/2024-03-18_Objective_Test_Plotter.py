# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Objective-test spectra


Data: 2024-03-18_785_Objective_Test.h5


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


#%%

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class Particle(): 
    def __init__(self):
        self.peaks = np.zeros((20,5)) 




#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-18_785_Objective_Test.h5")


#%% Get groups for each objective

x100_NA09 = my_h5['100X_0.9NA_Olympus']
x20_NA04 = my_h5['20x_0.4NA_Olympus_0']
x20_NA045 = my_h5['20x_0.45NA_Olympus_0']
x20_NA04_IR = my_h5['20x_0.4NA_IR_Olympus_0']
x40_Reflective = my_h5['LMM-40X-UVV_reflective_objective_0']
no_objective = my_h5['No_Objective_0']

group_list = [x100_NA09, x20_NA04, x20_NA045, x20_NA04_IR, x40_Reflective, no_objective]


#%% Plot BPT for each objective

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('SERS Intensity (cts/mW/s)', size = 'large')

for group in group_list:
    particle = group
    keys = list(particle.keys())
    keys = natsort.natsorted(keys)
    spectrum = None
    
    for key in keys:
        if 'BPT' in key and key[len(key)-1] == '0':
            
            power = key[key.find('mW')-4:key.find('mW')]
            power = float(power)
            spectrum = particle[key]
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.y = spectrum.y/power
            ax.plot(spectrum.x, spectrum.y, label = group.name)
    

ax.legend()
fig.suptitle('785nm BPT NPoM SERS - Objective Test', fontsize = 'large')
plt.tight_layout()

# Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'BPT 785nm SERS Objective Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot BPT normalized for each objective

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Normalized SERS Intensity', size = 'large')

for group in group_list:
    particle = group
    keys = list(particle.keys())
    keys = natsort.natsorted(keys)
    # spectrum = None
    
    for key in keys:
        if 'BPT' in key and key[len(key)-1] == '0':
            
            power = key[key.find('mW')-4:key.find('mW')]
            power = float(power)
            spectrum = particle[key]
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.truncate(820, None)
            spectrum.y = spectrum.y/power
            spectrum.normalise()
            ax.plot(spectrum.x, spectrum.y_norm, label = group.name)
    

ax.legend()
ax.set_ylim(0,1.4)
fig.suptitle('785nm BPT NPoM SERS - Objective Test', fontsize = 'large')
plt.tight_layout()

# Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'BPT 785nm SERS Normalized Objective Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot Si for each objective

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('SERS Intensity (cts/mW/s)', size = 'large')

for group in group_list:
    particle = group
    keys = list(particle.keys())
    keys = natsort.natsorted(keys)
    spectrum = None
    
    for key in keys:
        if 'Si_785' in key and key[len(key)-1] == '0':
            
            power = key[key.find('mW')-4:key.find('mW')]
            power = float(power)
            spectrum = particle[key]
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.y = spectrum.y/power
            ax.plot(spectrum.x, spectrum.y, label = group.name)
    

ax.legend()
fig.suptitle('785nm Si - Objective Test', fontsize = 'large')
plt.tight_layout()

# Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'Si 785nm Objective Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot Si normalized for each objective

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Normalized SERS Intensity', size = 'large')

for group in group_list:
    particle = group
    keys = list(particle.keys())
    keys = natsort.natsorted(keys)
    # spectrum = None
    
    for key in keys:
        if 'Si_785' in key and key[len(key)-1] == '0':
            
            power = key[key.find('mW')-4:key.find('mW')]
            power = float(power)
            spectrum = particle[key]
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.y = spectrum.y/power
            spectrum.normalise()
            ax.plot(spectrum.x, spectrum.y_norm, label = group.name)
    

ax.legend()
ax.set_ylim(0,1.4)
fig.suptitle('785nm Si - Objective Test', fontsize = 'large')
plt.tight_layout()

# # Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'Si 785nm Normalized Objective Test' + '.svg', format = 'svg')
# plt.close(fig)



#%% Plot white_scatt for all optics

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Intensity (cts/s)', size = 'large')

white = x100_NA09['white_scatt_785_100x_0.9NA_0.004s_x5_3']
x100_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x100_white.y = x100_white.y/0.004

white = x20_NA04['white_scatt_785_20x_0.4NA_0.004s_x5_0']
x20_NA04_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_white.y = x20_NA04_white.y/0.004

white = x20_NA045['white_scatt_785_20x_0.45NA_0.004s_x5_0']
x20_NA045_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA045_white.y = x20_NA045_white.y/0.004

white = x20_NA04_IR['white_scatt_785_20x_0.4NA_IR_0.004s_x5_0']
x20_NA04_IR_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_IR_white.y = x20_NA04_IR_white.y/0.004

white = x40_Reflective['white_scatt_785_BF_LMM-40x_0.0005s_0']
x40_Reflective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x40_Reflective_white.y = x40_Reflective_white.y/0.0005

white = no_objective['white_scatt_785_No_Objective_BF_0.001s_x5_0']
no_objective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_objective_white.y = no_objective_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_NoBS_0.001s_x5_1']
no_BS_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_BS_white.y = no_BS_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_0.001s_x5_0']
no_BS2_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_BS2_white.y = no_BS2_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_NoNotch_0.001s_x5_0']
no_notch_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_notch_white.y = no_notch_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_NoNotch_NoLens_Grating4_850cnwln_0.5s_x5_0']
no_lens_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_lens_white.y = no_lens_white.y/0.00095


white_list = [x100_white, x20_NA04_white, x20_NA045_white, x20_NA04_IR_white, x40_Reflective_white, no_objective_white, no_BS_white, no_BS2_white, no_notch_white, no_lens_white]
name_list = ['100x 0.9NA', '20x 0.4NA', '20x 0.45NA', '20x 0.4NA IR', '40x Reflective', 'No Objective', 'No Cam BS', 'No Laser BS', 'No Notch', 'No Lens', 'New Mirror']

for i, spectrum in enumerate(white_list[5:]):
    
    spectrum.truncate(800, None)
    # spectrum.y = spectrum.y/spectrum.y.max()
    ax.plot(spectrum.x, spectrum.y, label = name_list[i+5])
    
    

ax.legend()
fig.suptitle('White Scatterer - Optics Test', fontsize = 'large')
plt.tight_layout()

# # Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'White Scatterer 785nm Optics Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot white_scatt for all optics normalized

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Normalized Intensity', size = 'large')

white = x100_NA09['white_scatt_785_100x_0.9NA_0.004s_x5_3']
x100_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x100_white.y = x100_white.y/0.004

white = x20_NA04['white_scatt_785_20x_0.4NA_0.004s_x5_0']
x20_NA04_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_white.y = x20_NA04_white.y/0.004

white = x20_NA045['white_scatt_785_20x_0.45NA_0.004s_x5_0']
x20_NA045_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA045_white.y = x20_NA045_white.y/0.004

white = x20_NA04_IR['white_scatt_785_20x_0.4NA_IR_0.004s_x5_0']
x20_NA04_IR_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_IR_white.y = x20_NA04_IR_white.y/0.004

white = x40_Reflective['white_scatt_785_BF_LMM-40x_0.0005s_0']
x40_Reflective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x40_Reflective_white.y = x40_Reflective_white.y/0.0005

white = no_objective['white_scatt_785_No_Objective_BF_0.001s_x5_0']
no_objective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_objective_white.y = no_objective_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_NoBS_0.001s_x5_1']
no_BS_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_BS_white.y = no_BS_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_0.001s_x5_0']
no_BS2_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_BS2_white.y = no_BS2_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_NoNotch_0.001s_x5_0']
no_notch_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_notch_white.y = no_notch_white.y/0.001

white = no_objective['white_scatt_785_No_Objective_BF_2NoBS_NoNotch_NoLens_Grating4_850cnwln_0.5s_x5_0']
no_lens_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_lens_white.y = no_lens_white.y/0.5

white_list = [x100_white, x20_NA04_white, x20_NA045_white, x20_NA04_IR_white, x40_Reflective_white, no_objective_white, no_BS_white, no_BS2_white, no_notch_white, no_lens_white]
name_list = ['100x 0.9NA', '20x 0.4NA', '20x 0.45NA', '20x 0.4NA IR', '40x Reflective', 'No Objective', 'No Cam BS', 'No Laser BS', 'No Notch', 'No Lens', 'New Mirror']

for i, spectrum in enumerate(white_list[5:]):
    
    spectrum.truncate(800,None)
    # spectrum.normalise()
    spectrum.y_norm = spectrum.y/spectrum.y.max()
    
    # if spectrum == no_lens_white:
    #     spectrum.y_norm = spectrum.y/spectrum.y.max()
    
    ax.plot(spectrum.x, spectrum.y_norm, label = name_list[i+5])
    
    

ax.legend()
fig.suptitle('White Scatterer - Optics Test', fontsize = 'large')
plt.tight_layout()

# # Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'Normalized White Scatterer 785nm Optics Test' + '.svg', format = 'svg')
# plt.close(fig)

#%% Plot white_scatt normalized for each objective

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Normalized Intensity', size = 'large')

white = x100_NA09['white_scatt_785_100x_0.9NA_0.004s_x5_3']
x100_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x100_white.y = x100_white.y/0.004

white = x20_NA04['white_scatt_785_20x_0.4NA_0.004s_x5_0']
x20_NA04_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_white.y = x20_NA04_white.y/0.004

white = x20_NA045['white_scatt_785_20x_0.45NA_0.004s_x5_0']
x20_NA045_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA045_white.y = x20_NA045_white.y/0.004

white = x20_NA04_IR['white_scatt_785_20x_0.4NA_IR_0.004s_x5_0']
x20_NA04_IR_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_IR_white.y = x20_NA04_IR_white.y/0.004

white = x40_Reflective['white_scatt_785_BF_LMM-40x_0.0005s_0']
x40_Reflective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x40_Reflective_white.y = x40_Reflective_white.y/0.0005

white = no_objective['white_scatt_785_No_Objective_BF_0.001s_x5_0']
no_objective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_objective_white.y = no_objective_white.y/0.001

white_list = [x100_white, x20_NA04_white, x20_NA045_white, x20_NA04_IR_white, x40_Reflective_white, no_objective_white]

for i, spectrum in enumerate(white_list):
    
    spectrum.normalise()
    ax.plot(spectrum.x, spectrum.y_norm, label = group_list[i].name)
    
    
ax.set_ylim(0,1.4)
ax.legend()
fig.suptitle('White Scatterer - Objective Test', fontsize = 'large')
plt.tight_layout()

# # Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'Normalized White Scatterer 785nm Objective Test' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot white_scatt normalized for each objective and changes in BS

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelenght (nm)', size = 'large')
ax.set_ylabel('Normalized Intensity', size = 'large')

white = x100_NA09['white_scatt_785_100x_0.9NA_0.004s_x5_3']
x100_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x100_white.y = x100_white.y/0.004

white = x20_NA04['white_scatt_785_20x_0.4NA_0.004s_x5_0']
x20_NA04_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_white.y = x20_NA04_white.y/0.004

white = x20_NA045['white_scatt_785_20x_0.45NA_0.004s_x5_0']
x20_NA045_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA045_white.y = x20_NA045_white.y/0.004

white = x20_NA04_IR['white_scatt_785_20x_0.4NA_IR_0.004s_x5_0']
x20_NA04_IR_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x20_NA04_IR_white.y = x20_NA04_IR_white.y/0.004

white = x40_Reflective['white_scatt_785_BF_LMM-40x_0.0005s_0']
x40_Reflective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
x40_Reflective_white.y = x40_Reflective_white.y/0.0005

white = no_objective['white_scatt_785_No_Objective_BF_0.001s_x5_0']
no_objective_white = SERS.SERS_Spectrum(white.attrs['wavelengths'], white[2])
no_objective_white.y = no_objective_white.y/0.001

white_list = [x100_white, x20_NA04_white, x20_NA045_white, x20_NA04_IR_white, x40_Reflective_white, no_objective_white]

for i, spectrum in enumerate(white_list):
    
    spectrum.normalise()
    ax.plot(spectrum.x, spectrum.y_norm, label = group_list[i].name)
    
    
ax.set_ylim(0,1.4)
ax.legend()
fig.suptitle('White Scatterer - Objective Test', fontsize = 'large')
plt.tight_layout()

# # Save plot
# save_dir = r'C:\Users\il322\Desktop\Offline Data\2024-03-18 Analysis\_'
# plt.savefig(save_dir + 'Normalized White Scatterer 785nm Objective Test' + '.svg', format = 'svg')
# plt.close(fig)

