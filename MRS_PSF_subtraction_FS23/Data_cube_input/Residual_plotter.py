#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:13:10 2023

@author: Gian
"""

import sys
sys.path.append("/Users/Gian/Documents/Github/ASL-Variable_Stars/Code/")
from plotter import plot

import numpy as np

# Residual = np.load("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/output/Residual_Frame.npy")
# wav_arr = np.load("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/output/Residual_wave_arr.npy")


import os
import sys
import configparser
import pdb
import numpy as np

sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
sys.path.append("/Users/Gian/Documents/GitHub/ASL-Variable_Stars/Code")

#from spectres import spectres
from ifuframeselection import SelectWavelengthRangeModule
from ifubadpixel import NanFilterModule
from ifucentering_test import IFUAlignCubesModule
from ifupsfpreparation import IFUStellarSpectrumModule
from ifupsfsubtraction import IFUPSFSubtractionModule
from ifuresizing import FoldingModule
from ifustacksubset import CrossCorrelationPreparationModule
from ifucrosscorrelation import CrossCorrelationModule
from ifupcasubtraction import IFUResidualsPCAModule
from ifuresizing import UnfoldingModule
from IFS_basic_subtraction import IFS_ClassicalRefSubstraction

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, FitCenterModule, RemoveLinesModule, StackCubesModule, BadPixelSigmaFilterModule, ParangReadingModule, AddLinesModule, FitsWritingModule, TextWritingModule
from pynpoint import MultiChannelReader, ShiftImagesModule, PaddingModule
from IFS_Plot import PlotCenterDependantWavelength, PlotSpectrum
from IFS_SimpleSubtraction import IFS_normalizeSpectrum

from plotter import plot, cplot
import matplotlib.pyplot as plt

# pdb.set_trace()


######### 
# After letting TestFile_Gian.py run all the image_tags should be saved in the database
# and can be reused in this file
#########


# =============================================================================
# Initialize Pipeline
# =============================================================================

# Define Directories
working_place_in = "/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input"
input_place_in = "/Users/Gian/Documents/JWST_Central-Database/cubes_obs3_red_red"
output_place_in = "/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output"

pipeline = Pypeline(working_place_in, input_place_in, output_place_in)

# module = IFS_normalizeSpectrum(name_in='norm_ref',
#                                image_in_tag='centered_ref',
#                                image_out_tag='normed_ref')
# pipeline.add_module(module)
# pipeline.run()

# =============================================================================
# science target
# =============================================================================

dat = pipeline.get_data("science")
wav = pipeline.get_attribute("science","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: sci")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: sci")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Raw star data", vmax=avg)

# =============================================================================
# reference star
# =============================================================================

dat = pipeline.get_data("ref")
wav = pipeline.get_attribute("ref","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: ref")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: ref")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Raw ref star data", vmax=avg)


# =============================================================================
# science star padded
# =============================================================================

dat = pipeline.get_data("science_pad")
wav = pipeline.get_attribute("science_pad","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: padded sci")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: padded sci")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Padded star", vmax=avg)

# =============================================================================
# reference star centered
# =============================================================================

dat = pipeline.get_data("ref_pad")
wav = pipeline.get_attribute("ref_pad","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: padded ref")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: padded ref")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Padded ref star", vmax=avg)

# =============================================================================
# science star centered
# =============================================================================

dat = pipeline.get_data("centered")
wav = pipeline.get_attribute("centered","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=1)
plt.title("Average flux vs wavelength: centered sci")
plt.show()
plt.scatter(wav,std,s=1)
plt.title("Std of flux vs wavelength: centered sci")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Centered science star", vmax=avg)

# =============================================================================
# ref star centered
# =============================================================================

dat = pipeline.get_data("centered_ref")
wav = pipeline.get_attribute("centered_ref","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=1)
plt.title("Average flux vs wavelength: centered ref")
plt.show()
plt.scatter(wav,std,s=1)
plt.title("Std of flux vs wavelength: centered ref")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Centered ref star", vmax=avg)

# =============================================================================
# science star normed
# =============================================================================

dat = pipeline.get_data("normed")
wav = pipeline.get_attribute("normed","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=1)
plt.title("Average flux vs wavelength: normed sci")
plt.show()
plt.scatter(wav,std,s=1)
plt.title("Std of flux vs wavelength: normed sci")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Normed science star", vmax=avg)

# =============================================================================
# ref star normed
# =============================================================================

dat = pipeline.get_data("normed_ref")
wav = pipeline.get_attribute("normed_ref","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=1)
plt.title("Average flux vs wavelength: normed ref")
plt.show()
plt.scatter(wav,std,s=1)
plt.title("Std of flux vs wavelength: normed ref")
plt.show()

frame = dat[5]
avg = np.average(frame)

cplot(frame,"Normed ref star", vmax=avg)

# =============================================================================
# Residual
# =============================================================================

dat = pipeline.get_data("Residual")
wav = pipeline.get_attribute("Residual","WAV_ARR",static=False)[0].astype(np.float32)
N = dat[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(dat[i])
    std[i] = np.std(dat[i])
plt.scatter(wav,avg,s=1)
plt.title("Average flux vs wavelength: Residual")
plt.show()
plt.scatter(wav,std,s=1)
plt.title("Std of flux vs wavelength: Residual")
plt.show()

for i in np.arange(dat.size):
    frame = dat[i]
    avg = np.average(frame)

    cplot(frame,"Residual of simple star subtraction", vmax=avg)