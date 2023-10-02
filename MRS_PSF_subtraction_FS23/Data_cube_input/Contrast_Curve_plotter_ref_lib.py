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
from IFS_Centering import IFS_Centering
from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_collapseBins
from jwstframeselection import SelectWavelengthCenterModuleJWST

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
star_dir = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs3_min"
pca_dir = "/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs9_min"



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

module = IFS_Centering(name_in = "centermod_ref",
                       image_in_tag = "ref_pad",
                       fit_out_tag = "shift_ref")

pipeline.add_module(module)
pipeline.run_module("centermod_ref")

module = ShiftImagesModule(name_in='shift_ref',
                           image_in_tag='ref_pad',
                           shift_xy="shift_ref",
                           image_out_tag='centered_ref')
pipeline.add_module(module)
pipeline.run_module("shift_ref")

selection = SelectWavelengthCenterModuleJWST(name_in="selective",
                                             image_in_tag="sci_pad",
                                             image_out_tag="select_sci",
                                             nr_frames = 10,
                                             wave_center = 5.3)
pipeline.add_module(selection)
pipeline.run_module("selective")

binning = IFS_collapseBins(name_in="binnion",
                           image_in_tag="select_sci",
                           image_out_tag="bin_sci")

pipeline.add_module(binning)
pipeline.run_module("binnion")

selection = SelectWavelengthCenterModuleJWST(name_in="selective_ref",
                                             image_in_tag="centered_ref",
                                             image_out_tag="select_ref",
                                             nr_frames = 10,
                                             wave_center = 5.3)
pipeline.add_module(selection)
pipeline.run_module("selective_ref")

binning = IFS_collapseBins(name_in="binnion_ref",
                           image_in_tag="select_ref",
                           image_out_tag="bin_ref")

pipeline.add_module(binning)
pipeline.run_module("binnion_ref")


science = pipeline.get_data("bin_sci")

pixscale = pipeline.get_attribute("bin_sci","PIXSCALE",static=False)[0]
dit_sci = pipeline.get_attribute("bin_sci","DIT",static=False)[0]

ref = pipeline.get_data("bin_ref") # we use the reference star as psf template for the moment
dit_ref = pipeline.get_attribute("bin_ref","DIT",static=False)[0]


p = np.array([1.15504083e-08, -1.70009986e-06, 8.73285027e-05, -2.16106801e-03,
              2.83057945e-02, -1.59613156e-01, 6.60276371e-01])
fwhm_fn = np.poly1d(p)
wav_i = np.argmin(pipeline.get_attribute("bin_sci","WAV_ARR",static=False)[0]-5.3)
wav = pipeline.get_attribute("bin_sci","WAV_ARR",static=False)[0][int(wav_i)]
print(wav)
fwhm = fwhm_fn(wav)/ (pixscale * 3600)
print("FWHM: ", fwhm)


psf_template = ref[0]
# psf_template = ref[10:-10,14:-14]
cplot(psf_template,"PSF model",vmin=0,vmax=0.01)


cplot(science[0],"Science star",vmin=0,vmax=10000)


# =============================================================================
# Contrast class instance
# =============================================================================
angles = np.linspace(0,2*np.pi,6)

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

num_fake_planets = 6

contrast_instance.design_fake_planet_experiments(
    flux_ratios=flux_ratio,
    num_planets=num_fake_planets,
    separations = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50]),
    overwrite=True)

# =============================================================================
# Step 2: Run fake planet experiments
# =============================================================================

from applefy.wrappers.JWSTpynpoint_wrap_unopt import JWSTSimpleSubtractionPynPoint_unopt

algorithm_function = JWSTSimpleSubtractionPynPoint_unopt(
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
    pixel_scale=pixscale*3600)



# import pdb
# pdb.set_trace()
# =============================================================================
# Plotting contrast curve
# =============================================================================
# compute the overall best contrast curve
overall_best = np.min(contrast_curves.values, axis=1)

# get the error bars of the the overall best contrast curve
best_idx = np.argmin(contrast_curves.values, axis=1)
best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

# Find one color for each number of PCA components used.
import seaborn as sns
colors = sns.color_palette(
    "rocket_r",
    n_colors=len(contrast_curves.columns))
colors

separations_arcsec = contrast_curves.reset_index(level=0).index
separations_FWHM = contrast_curves.reset_index(level=1).index

# 1.) Create Plot Layout
fig = plt.figure(constrained_layout=False, figsize=(12, 8))
gs0 = fig.add_gridspec(1, 1)
axis_contrast_curvse = fig.add_subplot(gs0[0, 0])


# ---------------------- Create the Plot --------------------
i = 0 # color picker

for tmp_model in contrast_curves.columns:

    num_components = "Residuals"
    tmp_flux_ratios = contrast_curves.reset_index(
        level=0)[tmp_model].values
    tmp_errors = contrast_errors.reset_index(
        level=0)[tmp_model].values

    axis_contrast_curvse.plot(
        separations_arcsec,
        tmp_flux_ratios,
        color = colors[i],
        label=num_components)

    axis_contrast_curvse.fill_between(
        separations_arcsec,
        tmp_flux_ratios + tmp_errors,
        tmp_flux_ratios - tmp_errors,
        color = colors[i],
        alpha=0.5)
    i+=1

axis_contrast_curvse.set_yscale("log")
# ------------ Plot the overall best -------------------------
axis_contrast_curvse.plot(
    separations_arcsec,
    overall_best,
    color = "blue",
    lw=3,
    ls="--",
    label="Best")

# ------------- Double axis and limits -----------------------
from scipy import interpolate
lim_mag_y = (4, -6)
lim_arcsec_x = (0.1, 2.8)
sep_lambda_arcse = interpolate.interp1d(
    separations_arcsec,
    separations_FWHM,
    fill_value='extrapolate')

axis_contrast_curvse_mag = axis_contrast_curvse.twinx()
axis_contrast_curvse_mag.plot(
    separations_arcsec,
    flux_ratio2mag(tmp_flux_ratios),
    alpha=0.)
axis_contrast_curvse_mag.invert_yaxis()

axis_contrast_curvse_lambda = axis_contrast_curvse.twiny()
axis_contrast_curvse_lambda.plot(
    separations_FWHM,
    tmp_flux_ratios,
    alpha=0.)

axis_contrast_curvse.grid(which='both')
axis_contrast_curvse_mag.set_ylim(*lim_mag_y)
axis_contrast_curvse.set_ylim(
    mag2flux_ratio(lim_mag_y[0]),
    mag2flux_ratio(lim_mag_y[1]))

axis_contrast_curvse.set_xlim(
    *lim_arcsec_x)
axis_contrast_curvse_mag.set_xlim(
    *lim_arcsec_x)
axis_contrast_curvse_lambda.set_xlim(
    *sep_lambda_arcse(lim_arcsec_x))

# ----------- Labels and fontsizes --------------------------

axis_contrast_curvse.set_xlabel(
    r"Separation [arcsec]", size=16)
axis_contrast_curvse_lambda.set_xlabel(
    r"Separation [FWHM]", size=16)

axis_contrast_curvse.set_ylabel(
    r"Planet-to-star flux ratio", size=16)
axis_contrast_curvse_mag.set_ylabel(
    r"$\Delta$ Magnitude", size=16)

axis_contrast_curvse.tick_params(
    axis='both', which='major', labelsize=14)
axis_contrast_curvse_lambda.tick_params(
    axis='both', which='major', labelsize=14)
axis_contrast_curvse_mag.tick_params(
    axis='both', which='major', labelsize=14)

axis_contrast_curvse_mag.set_title(
    r"$5 \sigma_{\mathcal{N}}$ Contrast Curves",
    fontsize=18, fontweight="bold", y=1.1)

# --------------------------- Legend -----------------------
handles, labels = axis_contrast_curvse.\
    get_legend_handles_labels()

leg1 = fig.legend(handles, labels,
                  bbox_to_anchor=(0.12, -0.08),
                  fontsize=14,
                  title="# PCA components",
                  loc='lower left', ncol=8)

_=plt.setp(leg1.get_title(),fontsize=14)
plt.show()