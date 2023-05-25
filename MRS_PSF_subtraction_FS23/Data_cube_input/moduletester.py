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
from center_guess import StarCenterFixedGauss

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader



from scratch_primaryhducombiner import hducollapser
from plotter import cplot

# pdb.set_trace()

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/cubes_obs3_red_red'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)

reader1 = MultiChannelReader(name_in="read",
                            input_dir=in_path,
                            image_tag="rawdog")
reader2 = MultiChannelReader(name_in="read2",
                             input_dir='/Users/Gian/Documents/JWST_Central-Database/cubes_obs9_red_red',
                             image_tag="rawcat")
pipeline.add_module(reader1)
pipeline.add_module(reader2)


pipeline.run_module("read")
pipeline.run_module("read2")

paddingtonbear = PaddingModule(name_in="bear",
                               image_in_tags=["rawdog","rawcat"],
                               image_out_suff="pad")
pipeline.add_module(paddingtonbear)
pipeline.run_module("bear")

dog = pipeline.get_data("rawdog_pad")
cat = pipeline.get_data("rawcat_pad")

shits = ShiftImagesModule(name_in='shit',
                          image_in_tag='rawdog',
                          shift_xy=(5,-7),
                          image_out_tag='centereddawg')

pipeline.add_module(shits)
pipeline.run_module('shit')
dogshit = pipeline.get_data('centereddawg')
cplot(dogshit[0],"Doggy style shit",vmax=15000)
# pdb.set_trace()
shifter = StarCenterFixedGauss(dogshit,plot_star=False,plot_gauss_aligned=False,A=1,bounds=(15,10))

shift = ShiftImagesModule(name_in='shifterz',
                          image_in_tag='centereddawg',
                          shift_xy=shifter,
                          image_out_tag='centereddogs')
pipeline.add_module(shift)
pipeline.run_module('shifterz')
doggy = pipeline.get_data("centereddogs")
print("##############HERE##########")
cplot(doggy[27],"Doggy",vmax=15000)


shifter = StarCenterFixedGauss(dog,plot_star=False,plot_gauss_aligned=False,A=1)
shifter2 = StarCenterFixedGauss(cat)

shift = ShiftImagesModule(name_in='shift',
                          image_in_tag='rawdog_pad',
                          shift_xy=shifter,
                          image_out_tag='centereddog')
shift2 = ShiftImagesModule(name_in='shift2',
                          image_in_tag='rawcat_pad',
                          shift_xy=shifter2,
                          image_out_tag='centeredcat')

pipeline.add_module(shift)
pipeline.add_module(shift2)
pipeline.run_module('shift')
pipeline.run_module("shift2")

centered = pipeline.get_data("centereddog")[27]
cplot(centered,"blob",vmax=15000)
centered = pipeline.get_data("centeredcat")[27]
cplot(centered,"blob",vmax=15000)


# pdb.set_trace()

subtr = IFS_ClassicalRefSubstraction(name_in = "sub", 
                                     image_in_tags=["centereddog_pad","centeredcat_pad"], 
                                     image_out_tag="residual")
pipeline.add_module(subtr)


# pdb.set_trace()
pipeline.run()
X = pipeline.get_data("residual")
print(X)

# Collapser = PrimaryHDUCombiner(name_in = "collagen", 
#                                input_dir = None,
#                                output_dir = None,
#                                change_name = True,
#                                primarywarn = False,
#                                overwrite = True)

# pipeline.add_module(Collapser)
# pipeline.run_module("collagen")

