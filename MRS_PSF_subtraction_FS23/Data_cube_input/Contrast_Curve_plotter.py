"""
Script to generate the contrast curve from a residual saved in the pipeline.
"""

import sys
sys.path.append("/Users/Gian/Documents/Github/ASL-Variable_Stars/Code/")
from plotter import plot

import numpy as np

# Residual = np.load("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/output/Residual_Frame.npy")
# wav_arr = np.load("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/output/Residual_wave_arr.npy")


import os
import configparser
import pdb
import numpy as np

sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
sys.path.append("/Users/Gian/Documents/GitHub/applefy")
# sys.path.append("/Users/Gian/Documents/GitHub/ASL-Variable_Stars/Code")

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, FitCenterModule, RemoveLinesModule, StackCubesModule, BadPixelSigmaFilterModule, ParangReadingModule, AddLinesModule, FitsWritingModule, TextWritingModule
from pynpoint import MultiChannelReader, ShiftImagesModule, PaddingModule
from center_guess import StarCenterFixedGauss, IFS_RefStarAlignment
from IFS_Plot import PlotCenterDependantWavelength, PlotSpectrum
from IFS_SimpleSubtraction import IFS_normalizeSpectrum

from plotter import plot, cplot
import matplotlib.pyplot as plt

from applefy.detections.contrast_JWST import Contrast

from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest

from applefy.utils.file_handling import load_adi_data
from applefy.utils import flux_ratio2mag, mag2flux_ratio

from IPython import display

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
input_place_in = "/Users/Gian/Documents/JWST_Central-Database/Full_cubes/1536/cubes_obs22"
output_place_in = "/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output"

working_place_in = "/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input"
star_dir = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs22_min"
pca_dir = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs23_min"



pipeline = Pypeline(working_place_in, input_place_in, output_place_in)

# Set wavelength:
k = 0
# =============================================================================
# Import raw Science and Ref star
# =============================================================================
Import_Science = MultiChannelReader(name_in = "readersci",
                            input_dir = star_dir,
                            image_tag = "sci",
                            check = True,
                            ifs_data=False,
                            overwrite=True)

pipeline.add_module(Import_Science)
pipeline.run_module("readersci")

Import_Ref = MultiChannelReader(name_in = "readerref",
                            input_dir = pca_dir,
                            image_tag = "ref",
                            check = True,
                            ifs_data=False,
                            overwrite=True)

pipeline.add_module(Import_Ref)
pipeline.run_module("readerref")

padding = PaddingModule(name_in="padding_module",
                        image_in_tags=["sci","ref"],
                        image_out_suff="pad",
                        squaring=True)
pipeline.add_module(padding)
pipeline.run_module("padding_module")

dat_ref = pipeline.get_data("ref_pad")

shift_ref = StarCenterFixedGauss(dat_ref)

module = ShiftImagesModule(name_in='shift_ref',
                                    image_in_tag='ref_pad',
                                    shift_xy=shift_ref,
                                    image_out_tag='centered_ref')
pipeline.add_module(module)
pipeline.run_module("shift_ref")

module = IFS_normalizeSpectrum(name_in='norm_ref',
                               image_in_tag='centered_ref',
                               image_out_tag='normed_ref')
pipeline.add_module(module)
pipeline.run_module("norm_ref")


science = pipeline.get_data("sci_pad")

pixscale = pipeline.get_attribute("sci_pad","PIXSCALE",static=False)[0]
dit_sci = pipeline.get_attribute("sci_pad","DIT",static=False)[0]

ref = pipeline.get_data("normed_ref") # we use the reference star as psf template for the moment
dit_ref = pipeline.get_attribute("normed_ref","DIT",static=False)[0]


p = np.array([1.15504083e-08, -1.70009986e-06, 8.73285027e-05, -2.16106801e-03,
              2.83057945e-02, -1.59613156e-01, 6.60276371e-01])
fwhm_fn = np.poly1d(p)
wav = pipeline.get_attribute("sci_pad","WAV_ARR",static=False)[0][k]
print(wav)
fwhm = fwhm_fn(wav)
print("FWHM: ", fwhm)

psf_template = ref[k]
# psf_template = ref[10:-10,14:-14]
cplot(psf_template,"PSF model",vmin=0,vmax=0.01)


cplot(science[k],"Science star",vmin=0,vmax=10000)


# =============================================================================
# Contrast class instance
# =============================================================================
angles = np.zeros(science.shape[0])

contrast_instance = Contrast(
    pipeline=pipeline,
    science_sequence=science,
    science_dir = star_dir,
    psf_template=psf_template,
    psf_dir = pca_dir,
    parang_rad=angles,
    psf_fwhm_radius=fwhm / 2,
    dit_psf_template=dit_ref,
    dit_science=dit_sci,
    scaling_factor=1., # A factor to account e.g. for ND filters
    checkpoint_dir="/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output")

# =============================================================================
# Step 1: Design fake planet experiments
# =============================================================================

# fake planet brightness
flux_ratio_mag = 14
flux_ratio = mag2flux_ratio(14)

print("Brightness of fake planets in mag: " + str(flux_ratio_mag))
print("Planet-to-star flux ratio: " + str(flux_ratio))

num_fake_planets = 3

contrast_instance.design_fake_planet_experiments(
    flux_ratios=flux_ratio,
    num_planets=num_fake_planets,
    overwrite=True)

# =============================================================================
# Step 2: Run fake planet experiments
# =============================================================================

from applefy.wrappers.JWSTpynpoint_wrap import JWSTSimpleSubtractionPynPoint

algorithm_function = JWSTSimpleSubtractionPynPoint(
    scratch_dir=working_place_in)

contrast_instance.run_fake_planet_experiments(
    algorithm_function=algorithm_function,
    num_parallel=1)

# =============================================================================
# Step 3: Compute the contrast curve
# =============================================================================

# Use apertures pixel values
photometry_mode_planet = AperturePhotometryMode(
    "ASS", # or "ASS"
    psf_fwhm_radius=fwhm/2,
    search_area=0.5)

photometry_mode_noise = AperturePhotometryMode(
    "AS",
    psf_fwhm_radius=fwhm/2)

contrast_instance.prepare_contrast_results(
    photometry_mode_planet=photometry_mode_planet,
    photometry_mode_noise=photometry_mode_noise)

statistical_test = TTest()

# statistical_test = LaplaceBootstrapTest.construct_from_json_file("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/input/Testforapplefy/"+"laplace_lookup_tables.csv")

contrast_curves, contrast_errors = contrast_instance.compute_analytic_contrast_curves(
    statistical_test=statistical_test,
    confidence_level_fpf=gaussian_sigma_2_fpf(5),
    num_rot_iter=20,
    pixel_scale=pixscale)




