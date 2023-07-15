#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:00:37 2023

@author: Gian
"""

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader, RefLibraryMultiReader

from plotter import cplot
import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureMask, aperture_photometry



file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Full_cubes'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)


sci = pipeline.get_data("normed")
ref = pipeline.get_data("normed_ref")
opt_ref = pipeline.get_data("normed_ref_opt")

k = int(0)
cplot(sci[k],"Science", vmax=0.005)
cplot(ref[k],"Ref", vmax=0.005)
cplot(opt_ref[k],"Optim ref", vmax=0.005)
cplot(sci[k]-ref[k],"Sci - Ref", vmax=0.005)
cplot(sci[k]-opt_ref[k],"Sci - Optim ref", vmax=0.001)

wavelengths = pipeline.get_attribute("normed","WAV_ARR",static=False)[0]
pixelscale = pipeline.get_attribute_full_len("normed","PIXSCALE",static=False)

xcenter = (sci[0].shape[0] - 1) / 2
ycenter = (sci[0].shape[1] - 1) / 2
j = 0
aperture = None

fwhm = [1.15504083e-08, -1.70009986e-06, \
        8.73285027e-05, -2.16106801e-03, \
        2.83057945e-02, -1.59613156e-01, \
        6.60276371e-01]
fwhm = np.poly1d(fwhm)

r_out = 2.5 * fwhm(wavelengths[j]) / (pixelscale[j] * 3600)
r_in = 0.5 * fwhm(wavelengths[j]) / (pixelscale[j] * 3600)
aperture = CircularAnnulus((xcenter, ycenter), r_in,r_out)
visualizer = (aperture.to_mask()).to_image((sci.shape[1],sci.shape[2]))
print("Center coords: ", xcenter,ycenter)
cplot(2*sci[j]-1*visualizer,"ApertureMask",vmax=0.005)
