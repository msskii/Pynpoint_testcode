"""
testfile to check any modules
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
from center_guess import IFS_RefStarAlignment#, StarCenterFixedGauss
# from center_guess_rev import StarCenterFixedGauss
from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_binning

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader, RefLibraryMultiReader


from scratch_primaryhducombiner import hducollapser
from plotter import cplot
import matplotlib.pyplot as plt
import time

# pdb.set_trace()

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Full_cubes'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)
# pdb.set_trace()
# reader = MultiChannelReader(name_in="reader", 
#                             input_dir='/Users/Gian/Documents/JWST_Central-Database/Full_cubes/1536/cubes_obs23',
#                             image_tag="cube23")
# pipeline.add_module(reader)
# pipeline.run_module("reader")


# cube = pipeline.get_data("cube23")
# cplot(cube[0],title="before",vmax=1000,vmin=0)
# PCAN = 1
# import time
# start = time.time()
# u,s,v = np.linalg.svd(cube[0],full_matrices=False)
# corr = np.zeros_like(np.diag(s))
# for i in np.arange(PCAN): corr[i,i] = 1
# cub_c = u @ np.diag(s) @ corr @ v
# stop = time.time()
# print("T=",stop-start)
# x,y = np.where(cub_c==np.max(cub_c))
# x = x[0]
# y = y[0]
# cplot(cub_c,title="after",vmax=1000,vmin=0,show=False)
# plt.scatter(y,x,s=10,c="Red")
# OUTPUT = cub_c[8:-8,12:-12]/cub_c.max()


# shift = (cub_c.shape[1]/2 - y,cub_c.shape[0]/2-x

# module = ShiftImagesModule(name_in='shift',
#                                     image_in_tag='science_pad',
#                                     shift_xy=shift,
#                                     image_out_tag='centered')
# pipeline.add_module(module)
# pipeline.run_module("shift")

reader = MultiChannelReader(name_in="reader", 
                            input_dir='/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs22_useful',
                            image_tag="cube3")
pipeline.add_module(reader)
pipeline.run_module("reader")

reader2 = MultiChannelReader(name_in="reader2", 
                            input_dir='/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs23_useful',
                            image_tag="cube9")
pipeline.add_module(reader2)
pipeline.run_module("reader2")

padding = PaddingModule(name_in="puddle",
                        image_in_tags=["cube3","cube9"],
                        image_out_suff="p")
pipeline.add_module(padding)
pipeline.run_module("puddle")


# # =============================================================================
# # Center and Normalize science target
# # =============================================================================

start = time.time()
# dat = pipeline.get_data("cube3_p")

# shift = StarCenterFixedGauss(dat)

module = IFS_normalizeSpectrum(name_in='norm',
                               image_in_tag='cube3_p',
                               image_out_tag='normed')
pipeline.add_module(module)
pipeline.run_module("norm")

# # =============================================================================
# # Center and Normalize ref target
# # =============================================================================

# dat_ref = pipeline.get_data("cube9_p")

# shift_ref = StarCenterFixedGauss(dat_ref)

module = IFS_normalizeSpectrum(name_in='norm_ref',
                               image_in_tag='cube9_p',
                               image_out_tag='normed_ref')
pipeline.add_module(module)
pipeline.run_module("norm_ref")

bins = 10
module = IFS_binning(name_in="bin", image_in_tag="normed", image_out_tag="binned",bin_size=bins)
pipeline.add_module(module)
pipeline.run_module("bin")

module = IFS_binning(name_in="bin_ref", image_in_tag="normed_ref", image_out_tag="binned_ref",bin_size=bins)
pipeline.add_module(module)
pipeline.run_module("bin_ref")

test = IFS_RefStarAlignment(name_in="aligner", 
                            sci_in_tag="binned", 
                            ref_in_tags="binned_ref", 
                            fit_out_tag_suff="opt",
                            qual_method = "L2",
                            in_rad = 0.8,
                            out_rad = 2.5,
                            apertshap = "Circle")
pipeline.add_module(test)
pipeline.run_module("aligner")

stop = time.time()
print("––––––––––––––––––––––––––––\n")
print("TIME: ", stop-start)

module = IFS_ClassicalRefSubstraction(name_in="subtr", image_in_tags=["binned","binned_ref_opt"], image_out_tag="residual")
pipeline.add_module(module)
pipeline.run_module("subtr")

sci = pipeline.get_data("cube3")
ref = pipeline.get_data("cube9")
norm = pipeline.get_data("normed")
norf = pipeline.get_data("normed_ref")
bi = pipeline.get_data("binned")
birf = pipeline.get_data("binned_ref")
birfal = pipeline.get_data("binned_ref_opt")
res = pipeline.get_data("residual")

k=2000
A = 0.8
kb = int(np.floor(k/bins))
cplot(sci[k],"Science",vmax=A*sci[k].max())
cplot(ref[k],"ref",vmax=A*ref[k].max())
cplot(norm[k],"norm",vmax=A*norm[k].max())
cplot(norf[k],"norm ref",vmax=A*norf[k].max())
cplot(bi[kb],"binned",vmax=A*bi[kb].max())
cplot(birf[kb],"binned ref",vmax=A*birf[kb].max())
cplot(birfal[kb],"aligned ref",vmax=A*birfal[kb].max())
cplot(res[kb],"residual",vmax=A*res[kb].max())