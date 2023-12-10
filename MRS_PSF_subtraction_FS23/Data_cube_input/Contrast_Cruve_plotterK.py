import numpy as np
import sys

sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs/background_files")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/applefy")

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
from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, FitCenterModule, RemoveLinesModule, \
    StackCubesModule, BadPixelSigmaFilterModule, ParangReadingModule, AddLinesModule, FitsWritingModule, \
    TextWritingModule
from pynpoint import MultiChannelReader, ShiftImagesModule, PaddingModule

import seaborn as sns
from scipy import interpolate

directory = "C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs/data/"

path = 'C:/Users/BIgsp/Documents/GitHub/Pynpoint_testcode/data/'
pca_dir = 'C:/Users/BIgsp/Documents/GitHub/Pynpoint_testcode/data/reference_stars'
star_dir = 'C:/Users/BIgsp/Documents/GitHub/Pynpoint_testcode/data/sci'

import_dirs = ['cubes_obs1',
               'cubes_obs2',
               'cubes_obs3b',
               'cubes_obs4',
               'cubes_obs6',
               'cubes_obs9',
               'cubes_obs17',
               'cubes_obs22',
               'cubes_obs23',
               'cubes_obs24',
               'cubes_obsP']

select = 10

binned_modules = ['binned_' + sub for sub in import_dirs]
padded_modules = [sub + "_padded" for sub in binned_modules]
centered_modules = [sub + '_centered' for sub in import_dirs]
normed_modules = [sub + '_normed' for sub in import_dirs]
shift_arr = [sub + 'shift' for sub in import_dirs]

pipeline = Pypeline(working_place_in=str(directory) + 'working_place/',
                    input_place_in=str(directory) + 'input_place/',
                    output_place_in=str(directory) + 'output_place/')

pixscale = pipeline.get_attribute(normed_modules[select], "PIXSCALE", static=False)[0]
dit_sci = pipeline.get_attribute(import_dirs[select], "DIT", static=False)[0]

# use the mean of all stars as a ref pdf
ref = pipeline.get_data("pca_model1")
dit_ref = pipeline.get_attribute("pca_model1", "DIT", static=False)[0]

# get all frames for pca
pca_ref = []
for tag in (normed_modules[:select] + normed_modules[select+1:]):
    pca_ref.append(pipeline.get_data(tag))

pca_ref = np.array(pca_ref)

p = np.array([1.15504083e-08, -1.70009986e-06, 8.73285027e-05, -2.16106801e-03,
              2.83057945e-02, -1.59613156e-01, 6.60276371e-01])
fwhm_fn = np.poly1d(p)

wav = pipeline.get_attribute(normed_modules[select], "WAV_ARR", static=False)[0][0]

fwhm = fwhm_fn(wav) / (pixscale * 3600)
print("FWHM: ", fwhm)

psf_template = (ref[0])
science = pipeline.get_data(normed_modules[select])

# cplot(psf_template, "PSF model",vmin=0,vmax=0.01)


# cplot(science[0], "Science star", vmin=0, vmax=0.01)

# =============================================================================
# Contrast class instance
# =============================================================================
angles = np.linspace(0, 2 * np.pi, 6)

contrast_instance = Contrast(
    pipeline=pipeline,
    science_sequence=science,
    science_dir=star_dir,
    psf_template=psf_template,
    psf_dir=pca_dir,
    parang_rad=angles,
    psf_fwhm_radius=fwhm / 2,
    dit_psf_template=dit_ref,
    dit_science=dit_sci,
    scaling_factor=1.,  # A factor to account e.g. for ND filters
    checkpoint_dir=path + "output")

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
    separations=np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5,  9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    overwrite=True)

# =============================================================================
# Step 2: Run fake planet experiments
# =============================================================================

from applefy.wrappers.JWSTpynpoint_wrap_unopt import JWSTPCASubtractionPynPoint_unopt

components = [0, 1, 2, 3, 4, 6, 8, 9]

algorithm_function = JWSTPCASubtractionPynPoint_unopt(
    scratch_dir=str(directory) + 'working_place',
    num_pcas=components,
    image_ref=pca_ref)

contrast_instance.run_fake_planet_experiments(
    algorithm_function=algorithm_function,
    num_parallel=1)

# =============================================================================
# Step 3: Compute the contrast curve
# =============================================================================


# Use apertures pixel values
photometry_mode_planet = AperturePhotometryMode(
    "ASS",  # or "ASS"
    psf_fwhm_radius=fwhm / 2,
    search_area=0.5)

photometry_mode_noise = AperturePhotometryMode(
    "AS",
    psf_fwhm_radius=fwhm / 2)

contrast_instance.prepare_contrast_results(
    photometry_mode_planet=photometry_mode_planet,
    photometry_mode_noise=photometry_mode_noise)

statistical_test = TTest()


contrast_curves, contrast_errors, pipeline2 = contrast_instance.compute_analytic_contrast_curves(
    statistical_test=statistical_test,
    confidence_level_fpf=gaussian_sigma_2_fpf(5),
    num_rot_iter=20,
    pixel_scale=pixscale * 3600)


# compute the overall best contrast curve
overall_best = np.min(contrast_curves.values, axis=1)

# get the error bars of the the overall best contrast curve
best_idx = np.argmin(contrast_curves.values, axis=1)
best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

# Find one color for each number of PCA components used.
colors = sns.color_palette(
    "rocket_r",
    n_colors=len(contrast_curves.columns))



separations_arcsec = contrast_curves.reset_index(level=0).index
separations_FWHM = contrast_curves.reset_index(level=1).index

# 1.) Create Plot Layout
fig = plt.figure(constrained_layout=False, figsize=(12, 8))
gs0 = fig.add_gridspec(1, 1)
axis_contrast_curvse = fig.add_subplot(gs0[0, 0])

# ---------------------- Create the Plot --------------------
i = 0 # color picker

for tmp_model in contrast_curves.columns:

    num_components = int(tmp_model[5:9])
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

lim_mag_y = (10, 0)

axis_contrast_curvse_mag.set_ylim(*lim_mag_y)
axis_contrast_curvse.set_ylim(
    mag2flux_ratio(lim_mag_y[0]),
    mag2flux_ratio(lim_mag_y[1]))

axis_contrast_curvse.grid(which='both')



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
    r"$5 \sigma_{\mathcal{N}}$ Contrast Curves for a potential  exo-planet at $5.3 \mu m$ Wavelength",
    fontsize=18, fontweight="bold", y=1.1)

# --------------------------- Legend -----------------------
handles, labels = axis_contrast_curvse.\
    get_legend_handles_labels()

leg1 = fig.legend(handles, labels,
                  bbox_to_anchor=(.50, 0),
                  fontsize=14,
                  title="# PCA components",
                  loc='lower center', ncol=9)

_=plt.setp(leg1.get_title(), fontsize=14)
plt.show()

print("stop")

plt.figure(figsize=(12, 8))

plt.plot(separations_arcsec,
         np.array(components)[np.argmin(
             contrast_curves.values,
             axis=1)],)

plt.title(r"Best number of PCA components",
          fontsize=18, fontweight="bold", y=1.1)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel("Separation [arcsec]", fontsize=16)
plt.ylabel("Number of PCA components", fontsize=16)

plt.grid()
ax2 = plt.twiny()
ax2.plot(separations_FWHM,
         np.array(components)[
             np.argmin(contrast_curves.values, axis=1)],)

print(list(np.array(components)[np.argmin(contrast_curves.values, axis=1)]))


ax2.set_xlabel("Separation [FWHM]", fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)

plt.show()

print("stop")