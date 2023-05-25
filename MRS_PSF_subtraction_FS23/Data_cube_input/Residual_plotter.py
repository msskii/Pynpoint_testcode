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
# sys.path.append("/Users/Gian/Documents/GitHub/ASL-Variable_Stars/Code")

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

# Set the wavelength for which all the plots are created (i.e. set the index between 0 and the size of the cube):
K = 11000
# Choose whether the maximum scale of the plots should be chosen as the average (avg_max=True) or the percentage p of the maximum (avg_max=False):
avg_max = False
p = 0.7

# =============================================================================
# science target
# =============================================================================

sci = pipeline.get_data("science")
wav = pipeline.get_attribute("science","WAV_ARR",static=False)[0].astype(np.float32)
N = sci[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(sci[i])
    std[i] = np.std(sci[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: sci")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: sci")
plt.show()

frame = sci[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Raw star data", vmin=0, vmax=avg)
else:
    cplot(frame,"Raw star data", vmin=0, vmax=p*frame.max())

# =============================================================================
# reference star
# =============================================================================

ref = pipeline.get_data("ref")
wav = pipeline.get_attribute("ref","WAV_ARR",static=False)[0].astype(np.float32)
N = ref[:,0,0].size
avg = np.zeros(N)
std = np.zeros(N)
for i in np.arange(N):
    avg[i] = np.average(ref[i])
    std[i] = np.std(ref[i])
plt.scatter(wav,avg,s=0.1)
plt.title("Average flux vs wavelength: ref")
plt.show()
plt.scatter(wav,std,s=0.1)
plt.title("Std of flux vs wavelength: ref")
plt.show()

frame = ref[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Raw ref star data",vmin=0,  vmax=avg)
else:
    cplot(frame,"Raw ref star data",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Padded star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Padded star",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Padded ref star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Padded ref star",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Centered science star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Centered science star",vmin=0,  vmax=p*frame.max())


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

frame = dat[K]
avg = np.average(frame)


if avg_max:
    cplot(frame,"Centered ref star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Centered ref star",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Normed science star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Normed science star",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Normed ref star",vmin=0,  vmax=avg)
else:
    cplot(frame,"Normed ref star",vmin=0,  vmax=p*frame.max())

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

frame = dat[K]
avg = np.average(frame)

if avg_max:
    cplot(frame,"Residual of simple star subtraction", vmax=avg)
else:
    cplot(frame,"Residual of simple star subtraction", vmax=p*frame.max())

# for i in np.arange(dat.size):
#     frame = dat[i]
#     avg = np.average(frame)

#     cplot(frame,"Residual of simple star subtraction", vmax=avg)