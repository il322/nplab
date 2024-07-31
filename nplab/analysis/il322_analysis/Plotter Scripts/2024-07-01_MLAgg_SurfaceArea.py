# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Initial attempt at script to use OpenCV to calculate area of MLAgg spot size from a photo of sample

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
import cv2
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


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-06-11 Photocurrent Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%%

## Read image
img = cv2.imread(r"C:\Users\il322\Desktop\Offline Data\2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_f_2.jpg")
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Crop to show just sample
crop_img = img[600:1100, 300:700]
# cv2.imshow('cropped image',crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# crop_img = cv2.blur(crop_img,(10,10))
crop_img = cv2.bilateralFilter(crop_img, 50, 100, 100)

## Find edges
edges = cv2.Canny(crop_img, 80, 60)

plt.subplot(121),plt.imshow(crop_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

