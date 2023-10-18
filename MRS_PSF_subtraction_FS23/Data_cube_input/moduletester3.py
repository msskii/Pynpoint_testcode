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
from jwstframeselection import SelectWavelengthCenterModuleJWST
from center_guess import IFS_RefStarAlignment#, StarCenterFixedGauss
# from center_guess_rev import StarCenterFixedGauss
from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_ClassicalRefSubstraction, IFS_binning, IFS_collapseBins

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


selection = SelectWavelengthCenterModuleJWST(name_in="selective",
                                             image_in_tag="cube3",
                                             image_out_tag="select3",
                                             nr_frames = 10,
                                             wave_center = 6.0)
pipeline.add_module(selection)
pipeline.run_module("selective")

data = pipeline.get_data("select3")
cplot(data[0],"Image", vmax=1000)

binning = IFS_collapseBins(name_in="binnion",
                           image_in_tag="select3",
                           image_out_tag="oneframe")

pipeline.add_module(binning)
pipeline.run_module("binnion")

data = pipeline.get_data("oneframe")
cplot(data,"oneframe", vmax=1000)

