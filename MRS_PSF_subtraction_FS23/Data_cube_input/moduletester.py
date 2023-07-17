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
from center_guess import StarCenterFixedGauss, IFS_RefStarAlignment
from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_binning

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader, RefLibraryMultiReader


from scratch_primaryhducombiner import hducollapser
from plotter import cplot

import time

# pdb.set_trace()

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Full_cubes'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)
# pdb.set_trace()
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


# =============================================================================
# Center and Normalize science target
# =============================================================================
start = time.time()

dat = pipeline.get_data("cube3_p")

shift = StarCenterFixedGauss(dat)

module = ShiftImagesModule(name_in='shift',
                                    image_in_tag='cube3_p',
                                    shift_xy=shift,
                                    image_out_tag='centered')
pipeline.add_module(module)
pipeline.run_module("shift")

module = IFS_normalizeSpectrum(name_in='norm',
                               image_in_tag='centered',
                               image_out_tag='normed')
pipeline.add_module(module)
pipeline.run_module("norm")


# =============================================================================
# Center and Normalize ref target
# =============================================================================

dat_ref = pipeline.get_data("cube9_p")

shift_ref = StarCenterFixedGauss(dat_ref)

module = ShiftImagesModule(name_in='shift_ref',
                                    image_in_tag='cube9_p',
                                    shift_xy=shift_ref,
                                    image_out_tag='centered_ref')
pipeline.add_module(module)
pipeline.run_module("shift_ref")

module = IFS_normalizeSpectrum(name_in='norm_ref',
                               image_in_tag='centered_ref',
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

module = IFS_RefStarAlignment(name_in="align", sci_in_tag="binned", ref_in_tags="binned_ref", fit_out_tag_suff="al")
pipeline.add_module(module)
pipeline.run_module("align")

stop = time.time()
print("––––––––––––––––––––––––––––\n")
print("TIME: ", stop-start)

module = IFS_ClassicalRefSubstraction(name_in="subtr", image_in_tags=["binned","binned_ref_al"], image_out_tag="residual")
pipeline.add_module(module)
pipeline.run_module("subtr")

sci = pipeline.get_data("cube3")
ref = pipeline.get_data("cube9")
norm = pipeline.get_data("normed")
norf = pipeline.get_data("normed_ref")
bi = pipeline.get_data("binned")
birf = pipeline.get_data("binned_ref")
birfal = pipeline.get_data("binned_ref_al")
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


