"""
testfile to check any modules
"""
import sys
import os
import pdb

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

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader


from scratch_primaryhducombiner import hducollapser

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

center = FitCenterModule(name_in='center_all',
                                  image_in_tag='rawdog',
                                  fit_out_tag='shiftdog',
                                  mask_radii=(0,0.5),
                                  method='full',
                                  guess=(5., -2., 1., 1., 16000., 0., 0.))

center2 = FitCenterModule(name_in='center_ref',
                                  image_in_tag='rawcat',
                                  fit_out_tag='shiftcat',
                                  mask_radii=(0,0.5),
                                  method='full',
                                  guess=(5., -2., 1., 1., 16000., 0., 0.))

pipeline.add_module(center)
pipeline.add_module(center2)

shift = ShiftImagesModule(name_in='shift',
                                    image_in_tag='rawdog',
                                    shift_xy='shiftdog',
                                    image_out_tag='centereddog')
pipeline.add_module(shift)

shift2 = ShiftImagesModule(name_in='shift2',
                                    image_in_tag='rawcat',
                                    shift_xy='shiftcat',
                                    image_out_tag='centeredcat')
pipeline.add_module(shift2)


# pdb.set_trace()
paddingtonbear = PaddingModule(name_in="bear",
                               image_in_tags=["centereddog","centeredcat"],
                               image_out_suff="pad")
pipeline.add_module(paddingtonbear)

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

