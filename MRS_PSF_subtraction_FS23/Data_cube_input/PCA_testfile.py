#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Identification for JWST point spread function
"""

import sys
import os
import pdb
import numpy as np

#########################
# User specific path
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
#########################

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
from IFS_SimpleSubtraction import IFS_normalizeSpectrum

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule,  ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader

from plotter import plot

# pdb.set_trace()

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs3_red_red'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)

reader1 = MultiChannelReader(name_in="read",
                            input_dir=in_path,
                            image_tag="rawdog")
reader2 = MultiChannelReader(name_in="read2",
                             input_dir='/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs9_red_red',
                             image_tag="rawcat")
pipeline.add_module(reader1)
pipeline.add_module(reader2)

module = IFS_normalizeSpectrum(name_in='normd',
                               image_in_tag='rawdog',
                               image_out_tag='normed_dog')
pipeline.add_module(module)

module = IFS_normalizeSpectrum(name_in='norm_ref',
                               image_in_tag='rawcat',
                               image_out_tag='normed_cat')
pipeline.add_module(module)

paddington = PaddingModule(name_in="pad",
                           image_in_tags=["normed_dog","normed_cat"],
                           image_out_suff="pad")

pipeline.add_module(paddington)


pipeline.run()

def trip_Matmul(U,S,V):
    return(np.matmul(np.matmul(U,np.diag(S)),V))


data1 = pipeline.get_data("normed_dog_pad")[0]
data1_arr = np.reshape(data1,(1,data1.size))
data2 = pipeline.get_data("normed_cat_pad")[0]
data2_arr = np.reshape(data2,(1,data2.size))

U,S,V = np.linalg.svd(data1,full_matrices=False)

# plot(data1,"Frame",vmax=2000)
# plot(U,"U",vmax=1)
# plot(np.diag(S),"S",vmax=2000)
# plot(V,"V",vmax=1)
plot(trip_Matmul(U,S,V),"svd",vmax=0.005)
S_red1 = np.zeros_like(S)
cut = 1
S_red1[0:cut] = S[0:cut]
plot(trip_Matmul(U,S_red1,V),f'{cut} PC',vmax=0.005)

S_red2 = np.zeros_like(S)
cut = 1


U,S,V = np.linalg.svd(data2,full_matrices=False)

# plot(data1,"Frame",vmax=2000)
# plot(U,"U",vmax=1)
# plot(np.diag(S),"S",vmax=2000)
# plot(V,"V",vmax=1)
# plot(trip_Matmul(U,S,V),"svd",vmax=2000)
S_red1 = np.zeros_like(S)
cut = 1
S_red1[0:cut] = S[0:cut]
plot(trip_Matmul(U,S_red1,V),f'{cut} PC',vmax=0.005)
# # model = np.concatenate((data1_arr, data2_arr), axis=0)


# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# princ_comp = np.ascontiguousarray(pca.fit_transform(model))
