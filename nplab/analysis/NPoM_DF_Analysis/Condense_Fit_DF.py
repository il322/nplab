# -*- coding: utf-8 -*-
"""
Created on Thu May 31 01:11:45 2018

@author: car72
"""
from __future__ import division
from __future__ import print_function
from past.utils import old_div
if __name__ == '__main__':
    print('Importing modules...')

#import h5py
import os
rootDir = os.getcwd()
import numpy as np
import time
import matplotlib.pyplot as plt
from nplab.analysis.NPoM_DF_Analysis import DF_Multipeakfit as mpf
from nplab.analysis.NPoM_DF_Analysis import Condense_DF_Spectra as cdf
os.chdir(rootDir) #Important

raiseExceptions = False #bool; Setting this to True will stop the analysis and return the traceback if an individual spectrum fails. Useful for debugging
'''If the analysis fails, set raiseExceptions = True - this will return the traceback for debugging'''

intensityRatios = False #bool; Plots CM peak positions against CM/TM intensity ratio. Useful for determining gap size and RI
statsOnly = False #bool; Set to True if the spectra have already been analysed and want to re-plot histograms (etc)
pl = False #bool; Set to True if your dataset contains PL measurments
npSize = 80 #int (50, 60, 70, 80); Specify np diameter (nm) to ensure analysis looks for peaks in the right place
npomTypes = ['All NPoMs', 'Ideal NPoMs']
consolidateScans = False #bool; set this to true if you have multiple particle tracks in the file and want to combine them
customScan = None #int; if your file contains multiple particle tracks from different samples, use this to specify the correct one
extractFirst = True #bool; set to false if your summary file already exists, true to create another
avgZScans = False #bool; if your data ends up with a weird, rising baseline, set this to True and try again
upperCutoff = 900 #int; upper wavelength limit (nm) for peak detection; fiddle with this if the code keeps finding "peaks" in the noisy longer-wavelength region
lowerCutoff = 580 #int; lower wavelength limit (nm) for peak detection; fiddle with this if you have crap in the shorter-wavelength region
startSpec = 0
finishSpec = 0

def run(raiseExceptions = raiseExceptions, intensityRatios = intensityRatios, 
        statsOnly = statsOnly, pl = pl, npSize = npSize, npomTypes = npomTypes, consolidateScans = consolidateScans,
        customScan = customScan, extractFirst = extractFirst, avgZScans = avgZScans, upperCutoff = upperCutoff, 
        lowerCutoff = lowerCutoff,startSpec = startSpec, finishSpec = finishSpec):
    
    print('\tModules imported')
    absoluteStartTime = time.time()

    if statsOnly == True:
        outputFileName = mpf.findH5File(os.getcwd(), nameFormat = 'MultiPeakFitOutput', mostRecent = True)#finds the most recent file with given name format
        mpf.doStats(outputFileName, stacks = False, pl = pl, npomTypes = npomTypes, intensityRatios = intensityRatios, 
                    upperCutoff = upperCutoff, lowerCutoff = lowerCutoff)

    else:

        #if consolidateScans == True:
        #    cdf.consoliData(os.getcwd())
        
        if extractFirst == True:
            summaryFile = cdf.extractAllSpectra(os.getcwd(), returnIndividual = True, start = startSpec,
                                                finish = finishSpec, raiseExceptions = raiseExceptions, customScan = customScan,
                                                consolidated = consolidateScans, avgZScans = avgZScans)#condenses Z-stack (inc. background subtraction and referencing) for each particle and makes summary file
    
            if pl == True:
                summaryFile, pl = cdf.transferPlSpectra(os.getcwd(), startWl = 505, start = startSpec,
                                                    finish = finishSpec)#background subtracts each PL spectra and transfers them to the existing summary file

        #summaryFile = 'summary.h5' #use this instead of the above functions if you already have a summary file

        if raiseExceptions == True:
            outputFileName = mpf.createOutputFile('MultiPeakFitOutput')
            mpf.fitAllSpectra(os.getcwd(), outputFileName, npSize = npSize, first = startSpec, last = finishSpec,
                              pl = pl, closeFigures = True, stats = True, npomTypes = npomTypes,
                              raiseExceptions = raiseExceptions, raiseSpecExceptions = raiseExceptions, 
                              intensityRatios = intensityRatios, upperCutoff = upperCutoff, lowerCutoff = lowerCutoff)


            print('\nData fitting complete')

        else:
            try:
                outputFileName = mpf.createOutputFile('MultiPeakFitOutput')
                mpf.fitAllSpectra(os.getcwd(), outputFileName, npSize = npSize, first = startSpec, last = finishSpec,
                                  pl = pl, closeFigures = True, stats = True, npomTypes = npomTypes, lowerCutoff = lowerCutoff,
                                  raiseExceptions = raiseExceptions, raiseSpecExceptions = raiseExceptions, 
                                  intensityRatios = intensityRatios, upperCutoff = upperCutoff)

                print('\nData fitting complete')

            except Exception as e:
                print('\nData fitting failed because %s' % (e))

    plt.close('all')
    absoluteEndTime = time.time()
    timeElapsed = absoluteEndTime - absoluteStartTime

    hours = int(old_div(timeElapsed, 3600))
    mins = int(np.round(old_div((timeElapsed % 3600),60)))
    secs = int(np.round(timeElapsed % 60))

    if hours > 0:
        print('\nM8 that took ages. %s hours %s min %s sec' % (hours, mins, secs))

    else:
        print('\nFinished in %s min %s sec' % (mins, secs))

if __name__ == '__main__':
    run()