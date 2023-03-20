#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:16:03 2023

@author: Gian
"""
import sys
from astropy.io import fits
# sys.path.insert(1, '/Users/Gian/Documents/GitHub/student-code/MRS_PSF_subtraction_FS23/Data_cube_input/Data')
# hdul = fits.open('/Users/Gian/Documents/GitHub/student-code/MRS_PSF_subtraction_FS23/Data_cube_input/input/cal_OBS091_0235_cam2.fits')  # open a FITS file
hdul = fits.open('/Users/Gian/Documents/GitHub/student-code/MRS_PSF_subtraction_FS23/Data_cube_input/Data/Level3_ch1-long_s3d.fits')  # open a FITS file

hdr = hdul[0].header  # the primary HDU header
print(list(hdr.keys()))