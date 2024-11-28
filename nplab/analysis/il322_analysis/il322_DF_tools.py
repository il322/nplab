# -*- coding: utf-8 -*-
"""
Created on Sun May 14 14:22:50 2023

@author: il322

Module with specific functions for processing and analysing DF spectra
Inherits spectral classes from spectrum_tools.py


To do:
    make function for plotting histogram
    add manual screening to plot screening df
    add rejection filter to particle scan loop
    think about global rc params
    
    Separately, make class for particle to store all types of data
"""

from copy import copy
import h5py
import os
import math
from math import log10, floor
import natsort
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from importlib import reload

from nplab.analysis.general_spec_tools import spectrum_tools as spt
from nplab.analysis.general_spec_tools import npom_df_pl_tools as df
from nplab.analysis.general_spec_tools import all_rc_params as arp


#%%

class Z_Scan(spt.Timescan):
    
    '''
    Object for handling NPoM DF z-scan datasets
    Inherits from 
    Contains functions for:
        Checking centering/focus of particles and/or collection path alignment
        Condensing z-stack into 1D spectrum, corrected for chromatic aberration
        Plotting z-scan
    '''
    
    def __init__(self, *args, dz = None, z_min = -3, z_max = 3, z_trim = [2,-2], 
                 particle_name = None, avg_scan = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.particle_name = particle_name
        
        ## Get dz
        if dz is None:
            self.z_min = z_min
            self.z_max = z_max
            dz = np.linspace(self.z_min, self.z_max, len(self.t_raw))     
        self.dz = dz

        ## Background and reference
        self.Y_raw = copy(self.Y)
        self.Y = self.Y - self.background
        self.Y = self.Y / (self.reference - self.background)
        self.Y = np.nan_to_num(self.Y, nan = 0, posinf = 0, neginf = 0) # remove nan's and inf's
        
        ## Trim
        self.dz = self.dz[z_trim[0]:z_trim[1]]
        self.t = self.t_raw[z_trim[0]:z_trim[1]]
        self.z_min = self.dz.min()
        self.z_max = self.dz.max()
        self.Y = self.Y[z_trim[0]:z_trim[1]]  
        
        self.avg_scan = np.mean(self.Y, axis = 0)
    
    
    def plot_z_scan(self, ax = None, rc_params = None, x_lim = None, 
                    cmap = 'inferno', title = 'Z-Scan', plot_centroid = False, colorbar = False, **kwargs):
        
        '''
        Plot z-scan on colormap
        '''

        if ax is None:
            fig, ax = plt.subplots(figsize = [14,10]) # if no axes provided, create some

        x = self.x
        z = self.dz
        Y = np.vstack(self.Y)

        pcm = ax.pcolormesh(x, z, Y, cmap = cmap, shading = 'auto', 
                      norm = mpl.colors.Normalize(vmin = 0, vmax = np.percentile(self.Y, 99.5)), rasterized = True, **kwargs)
        
        if colorbar == True:
            clb = plt.colorbar(pcm, ax = ax)
            clb.set_label(label = 'DF Intensity', rotation = 270, labelpad=30)            
            
        if plot_centroid == True:
            ax.plot(self.x, self.z_profile, color = 'mediumaquamarine')
            
        if x_lim is None:
            x_lim = [self.x.min() - 1, self.x.max() + 1]
        else:
            ax.set_xlim(x_lim)
            
        ax.set_ylim(z.min(), z.max())
        ax.set_ylabel('Focal Height ($\mathrm{\mu}$m)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(title)


    def check_centering(self, dz_interp_steps = 50, brightness_threshold = 4, 
                        plot = False, ax = None, print_progress = True, title = None, **kwargs):
        
        '''
        Checks whether the particle was correctly focused/centred during measurement
            (important for accuracy of absolute spectral intensities)

        Inspects the average z-profile of spectral intensity
        z-profile has one obvious intensity maximum if the particle is correctly focused/centred, and the collection path aligned
        If the z-profile deviates from this, the particle is flagged as unfocused/misaligned
        '''
        
        ## Average intensity at each z-position
        z_profile = np.average(self.Y, axis = 1)
        z_profile = z_profile - z_profile.min()

        ## Split profile into thirds
        dz_cont = np.linspace(self.z_min, self.z_max, dz_interp_steps)
        buffer = int(round(dz_interp_steps/4))
        z_profile_cont = np.interp(dz_cont, self.dz, z_profile)

        ## If centre brightness is brighter than background brightness of z-stack, particle is aligned
        i_edge = np.trapz(z_profile_cont[:buffer]) + np.trapz(z_profile_cont[-buffer:])
        i_mid = np.trapz(z_profile_cont[buffer:-buffer])
        relative_brightness = i_mid/i_edge
        self.aligned = relative_brightness > brightness_threshold # bool

        ## Plot
        if plot == True:  
            
            if ax is None:
                fig, ax = plt.subplots(figsize = [14, 10])# if no axes provided, create some
            
            ax.plot(dz_cont, z_profile_cont/z_profile_cont.max(), 'k', lw = 4)
            ax.vlines(dz_cont[buffer], 0 , 1)
            ax.vlines(dz_cont[-buffer], 0 , 1)
            ax.set_xlabel('Focal height (um)')
            ax.set_ylabel('Average intensity')
            status = 'Aligned' if self.aligned == True else 'Not Aligned'
            status = f'{status}: {relative_brightness:.2f}'
            ax.set_title(title)
            ax.text(s = status, x = ax.get_xlim()[0] + (ax.get_xlim()[1] * 0.05), y = 0.9, fontsize = 25)

    def condense_z_scan(self, threshold = 0.01, plot = False, cosmic_ray_removal = False, **kwargs):
        
        '''
        Condenses z-scan into 1D DF spectrum
        Z scan is thresholded and the centroid is taken for each wavelength
        '''

        Y_T = self.Y.T

        # Get indicies of brightest z position for each wavelength
        max_indices = np.array([wl_scan.argmax() for wl_scan in Y_T]) # finds index of brightest z position for each wavelength
        max_indices_smooth = spt.butter_lowpass_filt_filt(max_indices, cutoff = 900, fs = 80000) # smooth

        Y_thresh = spt.remove_nans(self.Y, noisy_data = True).astype(np.float64)
        Y_thresh = (Y_thresh - Y_thresh.min(axis = 0))/(Y_thresh.max(axis = 0) - Y_thresh.min(axis = 0))
        Y_thresh -= threshold
        Y_thresh *= (Y_thresh > 0) #Normalise and Threshold array
        ones = np.ones([Y_thresh.shape[1]])
        z_positions = np.array([ones*n for n in np.arange(Y_thresh.shape[0])]).astype(np.float64)

        centroid_indices = np.sum((Y_thresh*z_positions), axis = 0)/np.sum(Y_thresh, axis = 0) #Find Z centroid position for each wavelength
        centroid_indices = spt.remove_nans(centroid_indices)

        assert np.count_nonzero(np.isnan(centroid_indices)) == 0, 'All centroids are NaNs; try changing the threshold when calling condense_z_scan()'

        if plot == True:                           
            fig = plt.figure(figsize = (7, 12))
            ax_z = plt.subplot2grid((14, 1), (0, 0), rowspan = 8)
            plt.setp(ax_z.get_xticklabels(), visible = False)
            ax_df = plt.subplot2grid((14, 1), (8, 0), rowspan = 6, sharex = ax_z)
            self.plot_z_scan(ax_z, title = None)
    
        df_spectrum = []
        z_profile = []

        for n, centroid_index in enumerate(centroid_indices):
            #use centroid_index in z (as float) to obtain interpolated spectral intensity

            if 0 < centroid_index < len(self.dz) - 1:#if calculated centroid is within z-range
                lower = int(centroid_index)
                upper = lower + 1
                frac = centroid_index % 1
                yi = spt.linear_interp(Y_T[n][lower], Y_T[n][upper], frac)
                zi = spt.linear_interp(self.dz[lower], self.dz[upper], frac)
                
            else:
                #print('centroid shifted')
                if centroid_index <= 0:
                    yi = Y_T[n][0]     
                    zi = self.dz[0]
                elif centroid_index >= len(self.dz) - 1:
                    yi = Y_T[n][-1]     
                    zi = self.dz[-1]

            df_spectrum.append(yi)
            z_profile.append(zi)

        if cosmic_ray_removal == True:
            df_spectrum = spt.remove_cosmic_rays(df_spectrum, **kwargs)

        df_spectrum = spt.remove_nans(df_spectrum)

        self.df_spectrum = np.array(df_spectrum)
        self.z_profile = np.array(z_profile)

        if plot == True:
            ax_df.plot(self.x, df_spectrum, alpha = 0.6, label = 'Centroid')
            ax_z.plot(self.x, z_profile)

            ax_df.plot(self.x, np.average(self.Y, axis = 0), label = 'Avg')
            ax_df.set_xlim(400, 900)
            ax_df.legend(loc = 0, fontsize = 14, ncol = 2)
            ax_df.set_xlabel('Wavelength (nm)')

            plt.subplots_adjust(hspace = 0)
            plt.show()


#%%
       
class DF_Spectrum(df.NPoM_DF_Spectrum):
    
    '''
    Object containing xy data and functions for NPoM DF spectral analysis
    Inherits from "df.NPoM_DF_Spectrum" data class
    args can be y data, x and y data, h5 dataset or h5 dataset and its name
    '''
    
    def __init__(self, *args, particle_name = None, np_size = 80, lower_cutoff = None,
                 pl = False, doubles_threshold = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.particle_name = particle_name

        centre_trough_wl_dict = {80: 680, 70 : 630, 60 : 580, 50 : 550, 40 : 540}
        cm_min_wl_dict = {80: 580, 70 : 560, 60 : 540, 50 : 520, 40 : 500}
        self.centre_trough_wl = centre_trough_wl_dict[np_size]
        self.cm_min_wl = cm_min_wl_dict[np_size]
        self.np_size = np_size

        if lower_cutoff is not None:
            self.cm_min_wl = lower_cutoff

        self.pl = pl

        if self.y_smooth is None:
            self.y_smooth = spt.butter_lowpass_filt_filt(y = self.y)

        self.find_maxima(**kwargs)
        
    
    def plot_df(self, ax = None, smooth = True, x_lim = None, y_lim = None, title = None, **kwargs):
        
        '''
        Plots DF spectrum using self.x and self.y
        '''
        
        if ax is None:
            fig, ax = plt.subplots(figsize = [14,10]) # if no axes provided, create some

        ax.plot(self.x, self.y, color = 'tab:blue')
        
        if smooth == True:
            ax.plot(self.x, self.y_smooth, color = 'tab:orange')
        
        ax.set_ylabel('Darkfield Intensity')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(title)
 
        
    def find_critical_wln(self, xlim = None):
        
        '''
        Find critical wavelength from list of maxima (takes global maximum)
        '''
        if xlim is None:
            xlim = (self.x.min(), self.x.max())
        crit_wln = 0
        global_max = 0
        for maximum in self.maxima:
            if self.x[maximum] >= xlim[0] and self.x[maximum] <= xlim[1]:
                if self.y_smooth[maximum] > global_max:
                    global_max = self.y_smooth[maximum]
                    crit_wln = self.x[maximum]
                    self.crit_maximum = maximum
                
        self.crit_wln = crit_wln
            

#%%

def df_screening(z_scan, df_spectrum, image = None, tinder = False, plot = False, title = None, save_file = None, brightness_threshold = 4, **kwargs):
    
    '''
    Function to screen NPoMs based on DF data
    Can plot NPoM CWL.image, z_scan, and df_spectrum with maxima
    
    Parameters:
        z_scan: (Z_Scan)
        df_spectrum: (DF_Spectrum)
        image: (HDF5 image = None)
        tinder: (boolean = False) Set True for manual NPoM screening
        plot: (boolean = False) plot z_scan, df_spectrum with maxima, and CWL.image
        title: (string = None) title for plotting
        save_file: (string = None) path for saving figure. If none, does not save
        
    Output:
        df_spectrum: (DF_Spectrum) Return df_spectrum with updated .is_npom and .not_npom_because attributes
    '''
    
    # Plotting
    
    if plot == True:
        fig, axes = plt.subplots(4, 1, figsize=[8,24])
        plt.rc('font', size=18)
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]
        ax1.set_title('Z-Scan')
        ax4.set_title('Image')
        fig.suptitle(title)
        plt.tight_layout(pad = 2)
    
        ## Plot z-scan
        z_scan.plot_z_scan(ax = ax1, plot_centroid = True, **kwargs)
    
        ## Plot df spectrum (raw & smoothed) w/ maxima & crit wln
        df_spectrum.plot_df(ax=ax2, title = 'Condensed DF Spectrum')
        if len(df_spectrum.maxima) > 0:
            for maximum in df_spectrum.maxima:
                ax2.scatter(df_spectrum.x[maximum], df_spectrum.y_smooth[maximum], marker='x', s=250, color = 'black', zorder = 10)
            ax2.scatter(df_spectrum.crit_wln, np.max(df_spectrum.y_smooth[df_spectrum.maxima]), marker = '*', s = 800, color = 'yellow', edgecolor = 'black', linewidth = 2, zorder = 20)
        ax2.set_yticks(np.round(np.linspace(0, df_spectrum.y.max(), 5), 3))       

        ## Plot z-profile
        z_scan.check_centering(plot = True, ax = ax3, title = 'Average Z-Profile', brightness_threshold = brightness_threshold)

        ## Plot CWL image
        if image is not None:
            ax4.imshow(image, zorder = 0)
            xlim = ax4.get_xlim()
            ylim = ax4.get_ylim()
            ax4.vlines(np.mean(xlim), ylim[0], ylim[1], color = 'magenta', alpha = 0.3)
            ax4.hlines(np.mean(ylim), xlim[0], xlim[1], color = 'magenta', alpha = 0.3)

        ## Labelling and such
        ax1.set_xlabel('')
        ax2.set_xlim(ax1.get_xlim())

    # Run NPoM tests & print reasons why NPoM rejected
        
    ## Check if z-scan centred correctly (z_scan.check_centering())
    if hasattr(z_scan, 'aligned') == False:
        z_scan.check_centering(brightness_threshold = brightness_threshold)
    if z_scan.aligned == False:
        df_spectrum.is_npom = False
        df_spectrum.not_npom_because = 'Centering failed'
        if plot == True: 
            ax3.text(s='Centering Failed', x = ax3.get_xlim()[0] + (ax3.get_xlim()[1] * 0.05), y = 0.75, fontsize = 25)
    
    ## Test df spectrum via test_if_npom() 
    else:
        if hasattr(df_spectrum, 'is_npom') == False:
            df_spectrum.test_if_npom()
        if df_spectrum.is_npom == False:
            if plot == True: ax2.text(s='NPoM Test failed: ' + '\n' + df_spectrum.not_npom_because, x = ax2.get_xlim()[0] + 10, y = ax2.get_ylim()[1] * 0.75, fontsize = 25, zorder=20)

    
    # Plot and save
    
    if plot == True and save_file == None:
        plt.show()
    elif plot == True and save_file is not None:
        plt.savefig(save_file, format = 'svg')
    
    
    # Manual rejection
    
    if tinder == True:        
        ar = input('a/d = accept/decline: ').strip().lower()
        if ar == 'a':
            df_spectrum.is_npom = True
        if ar == 'd':
            ### If not already rejected by automatic screening
            if df_spectrum.is_npom == True:
                df_spectrum.is_npom = False
                df_spectrum.not_npom_because = 'Manually rejected'
    
    
    return df_spectrum


#%%

def plot_df_histogram(crit_wln_list, 
                      df_spectra_list = None, 
                      df_spectrum_x = None, 
                      num_bins = 31, 
                      bin_range = (550, 850),
                      df_avg_threshold = 1,
                      ax = None,
                      title = None,
                      ax_df_label = None,
                      **kwargs):

    '''
    Function for plotting darkfield critical wavelength histogram with avg df spectra
    
    Parameters:
        crit_wln_list: (1DArray float) array of critical wavelengths
        df_spectra_list: (2DArray float = None) array of dark field spectra, axis 0 corresponds to particles in crit_wln_list
        df_spectrum_x: (1D Array float = None) array of dark field x-axis for plotting
        num_bins: (int = 31) number of bins for histogram
        bin_range: (tuple = (550,850)) wavelength range for histogram
        df_avg_threshold: (int = 1) Number of counts in bin required to plot df average for that bin
        ax: (plt ax = None) axis for plotting. If None will create figure
        title: (str = None) title for histogram
        ax_df_label: (str = None) Label for secondary y-axis
        
    Need to add:
        Fitting histogram
        Title
        Saving
        RC Params
    '''
    
    
    # Binning data
    
    crit_wln_list = np.array(crit_wln_list)
    df_spectra_list = np.array(df_spectra_list)
    
    ## Set bins
    bins = np.linspace(bin_range[0], bin_range[1], num_bins)
    
    ## Find bin index for each crit_wln
    inds = np.digitize(crit_wln_list, bins[:-1], right = False) - 1
    
    ## Find counts in each bin
    hist = np.histogram(bins[inds], bins = bins)[0]
    bins = bins[:-1]
    
    
    # Plot histogram
    
    if ax is None:
        plt.rc('font', size=18, family='sans-serif')
        plt.rc('lines', linewidth=3)
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot()
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Frequency')
        if title is not None:
            fig.suptitle(title, fontsize = 'large')
        else:
            fig.suptitle('NPoM $\lambda_c$ Histogram', fontsize = 'large')
   
    else:
        if title is not None:
            plt.title(title, fontsize = 'large', pad = 1)
        else:
            plt.title('NPoM $\lambda_c$ Histogram', fontsize = 'large', pad = 1)
    
    my_cmap = plt.get_cmap("hsv")
    colors = (-bins + 800)/320 # Rainbow colormap where 550nm = violet and 800nm = ref
    ax.bar(bins, hist, width = ((max(bins)-min(bins))/(num_bins)), color=my_cmap(colors), align = 'edge', zorder=2)
    
    
    # Average df spectrum per bin
    
    if df_spectra_list is not None and df_spectrum_x is not None:
    
        bin_avg = np.zeros((len(hist), len(df_spectra_list[0])))     
        
        ## Secondary axis for df_spectrum
        ax_df = ax.twinx()
        if ax_df_label is not None:
            ax_df.set_ylabel(ax_df_label, rotation = 270, labelpad = 25)
            ax_df.yaxis.set_label_position("right")
        
        ## Loop over each bin
        for i in range(0, len(hist)):
            ### If bin frequency meets threshold 
            if hist[i] >= df_avg_threshold:
                #### Avg df_spectrum in bin
                bin_avg[i] = np.sum(df_spectra_list[np.where(inds == i)], axis = 0) / len(df_spectra_list[np.where(inds == i)])
                #### Plot normalized avg df spectrum
                ax_df.plot(df_spectrum_x, (bin_avg[i] - bin_avg[i].min())/(bin_avg[i].max() - bin_avg[i].min()), color=my_cmap(colors[i]))
    
        ax_df.set_ylim(0, 1.2)
        ax_df.set_yticks([])
        ax.set_zorder(ax_df.get_zorder() + 1)
        ax.patch.set_visible(False)
        
    ax.set_xlim(bin_range)
    
    
    # Fit & plot normal distribution
    
    mu, std = norm.fit(crit_wln_list)
    FWHM = 2.3548 * std
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p/p.max() * hist[np.where(bins < mu)[0].max()], color = 'black', linewidth = 2, linestyle = 'dashed')
    ax.text(s = 'FWHM:\n' + str(np.round(FWHM)) + 'nm', x = 470, y = ax.get_ylim()[1] * 0.8)
    
#%% Template of how to analyze your DF data (run this from your script)


# # Get particle scan & list of particles from h5
# my_h5 = h5py.File(r'C:\Users\il322\Desktop\Offline Data\2023-05-10_M-TAPP-SMe_NPoM\2023-05-22_M-TAPP-SME_80nm_NPoM_Track_DF_633nmPowerseries.h5')
# particle_scan = 'ParticleScannerScan_2'
# particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
# for particle in particle_list:
#     if 'Particle' not in particle:
#         particle_list.remove(particle)


# # Set lists for critical wavelength hist & rejected particles
# crit_wln_list = []
# df_spectra_list = []
# rejected = []


# # Loop over particles in particle scan

# for particle in particle_list:
#     particle_name = particle_scan + ': ' + particle
#     particle = my_h5[particle_scan][particle]
    
#     ## Get z_scan, df_spectrum, crit_wln of particle
#     try:
#         z_scan = particle['lab.z_scan_0']
#     except:
#         print(particle_name + ': Z-Scan not found')
#         continue
#     z_scan = Z_Scan(z_scan)
#     z_scan.condense_z_scan() # Condense z-scan into single df-spectrum
#     ### Smoothing necessary for finding maxima
#     z_scan.df_spectrum = DF_Spectrum(x = z_scan.x,
#                                       y = z_scan.df_spectrum, 
#                                       y_smooth = spt.butter_lowpass_filt_filt(z_scan.df_spectrum, cutoff = 1600, fs = 200000))
#     z_scan.df_spectrum.test_if_npom()
#     z_scan.df_spectrum.find_critical_wln()
        
#     ## Run DF screening of particle
#     image = particle['CWL.thumb_image_0']
#     z_scan.df_spectrum = df_screening(z_scan = z_scan,
#                                       df_spectrum = z_scan.df_spectrum,
#                                       image = image,
#                                       tinder = False,
#                                       plot = False,
#                                       title = particle_name)

#     ## Add crit_wln & df_spectrum to list for binning or reject
#     if z_scan.aligned == True and z_scan.df_spectrum.is_npom == True:
#         crit_wln_list.append(z_scan.df_spectrum.crit_wln)
#         df_spectra_list.append(z_scan.df_spectrum.y_smooth)
        
#     else:
#         rejected.append(particle_name + ' - ' + z_scan.df_spectrum.not_npom_because)

# ## Plot histogram
# crit_wln_list = np.array(crit_wln_list)
# df_spectra_list = np.array(df_spectra_list)   
# bin_range = (crit_wln_list.min(), crit_wln_list.max())
# plot_df_histogram(crit_wln_list, 
#                   df_spectra_list, 
#                   z_scan.df_spectrum.x, 
#                   num_bins = int(np.ceil((len(crit_wln_list)**0.5))), 
#                   bin_range = (500,900), 
#                   df_avg_threshold = 2,
#                   title = 'Co-TAPP-SMe')

#%%

# my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-26_TiO_NPoM_DF_633_Powerseries_Track.h5")

# particle = my_h5['ParticleScannerScan_1']['Particle_0']
# keys = list(particle.keys())
# keys = natsort.natsorted(keys)

# z_scans = []
# images = []

# for key in keys:
#     if 'z_scan' in key:
#         z_scans.append(particle[key])
#     if 'image' in key:
#         images.append(particle[key])

# for i, scan in enumerate(z_scans):
    
#     name = str(copy(scan.name))
    
#     scan = Z_Scan(scan, z_min = scan.attrs['neg'], z_max = scan.attrs['pos'],z_trim = [8, -8])
#     scan.truncate(400, 900)   
#     scan.condense_z_scan()
#     spectrum = scan.df_spectrum
#     scan.df_spectrum = DF_Spectrum(scan.x, spectrum)
#     scan.df_spectrum.find_critical_wln()
#     image = images[i]
#     scan.df_spectrum = df_screening(scan, scan.df_spectrum, image, plot = True, brightness_threshold = 3)
    
#     z_scans[i] = scan
#     images[i] = image
    
# particle.z_scans = z_scans
# particle.images = images
# particle.crit_wlns = [z_scans[0].df_spectrum.crit_wln, z_scans[1].df_spectrum.crit_wln]
# particle.is_npom = [particle.z_scans[0].df_spectrum.is_npom, particle.z_scans[0].df_spectrum.is_npom]  