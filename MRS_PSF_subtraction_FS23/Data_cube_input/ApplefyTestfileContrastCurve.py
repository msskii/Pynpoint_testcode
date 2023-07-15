"""
Test file for applefy compatibility with JWST data
"""

from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate

from applefy.detections.contrast import Contrast

from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf, \
    fpf_2_gaussian_sigma, LaplaceBootstrapTest

from applefy.utils.file_handling import load_adi_data
from applefy.utils import flux_ratio2mag, mag2flux_ratio

from IPython import display
display.Image("../04_apples_with_apples/paper_experiments/01_01_observation.png")

test = TTest()
test.t_2_fpf(2.28, num_noise_values=10)

root_dir = Path("/Users/Gian/Documents/Github/Pynpoint_testcode/Data/input/Testforapplefy")

dataset_config = {'file_path': Path('30_data/betapic_naco_lp_LR.hdf5'),
                  'stack_key': "science_no_planet",
                  'psf_template_key': "psf_template",
                  'parang_key': 'header_science_no_planet/PARANG',
                  'dit_psf_template': 0.02019,
                  'fwhm_size' : 4.2, # Diameter in pixel
                  'dit_science': 0.2}

# we need the psf template for contrast calculation
science_data, angles, raw_psf_template_data = load_adi_data(
    root_dir/dataset_config["file_path"],
    data_tag=dataset_config["stack_key"],
    psf_template_tag=dataset_config["psf_template_key"],
    para_tag=dataset_config["parang_key"])

dit_psf_template = dataset_config["dit_psf_template"]
dit_science = dataset_config["dit_science"]
fwhm = dataset_config["fwhm_size"]

psf_template = raw_psf_template_data[82:-82, 82:-82]

plt.imshow(psf_template)
_=plt.title("Input PSF image")
plt.show()

plt.imshow(science_data[0, :, :])
_=plt.title("Example of a sciene frame \n with AGPM coronagraph")
plt.show()
# =============================================================================
# How to compute Contrast Curve
# =============================================================================

contrast_instance = Contrast(
    science_sequence=science_data,
    psf_template=psf_template,
    parang_rad=angles,
    psf_fwhm_radius=fwhm / 2,
    dit_psf_template=dit_psf_template,
    dit_science=dit_science,
    scaling_factor=1., # A factor to account e.g. for ND filters
    checkpoint_dir=root_dir / Path("70_results/contrast_curves"))

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
    overwrite=True)

# =============================================================================
# Step 2: Run fake planet experiments
# =============================================================================

from applefy.wrappers.pynpoint import MultiComponentPCAPynPoint

components = [5, 10, 20, 30, 50, 75, 100]

algorithm_function = MultiComponentPCAPynPoint(
    num_pcas=components,
    scratch_dir=contrast_instance.scratch_dir,
    num_cpus_pynpoint=1)

contrast_instance.run_fake_planet_experiments(
    algorithm_function=algorithm_function,
    num_parallel=1)

# =============================================================================
# Step 3: Compute the contrast curve
# =============================================================================

# # Use spaced pixel values
# photometry_mode_planet = AperturePhotometryMode(
#     "FS", # or "P"
#     psf_fwhm_radius=fwhm/2,
#     search_area=0.5)
# photometry_mode_noise = AperturePhotometryMode(
#     "P",
#     psf_fwhm_radius=fwhm/2)

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
    pixel_scale=0.02718)

# =============================================================================
# Plotting contrast curve
# =============================================================================
# compute the overall best contrast curve
overall_best = np.min(contrast_curves.values, axis=1)

# get the error bars of the the overall best contrast curve
best_idx = np.argmin(contrast_curves.values, axis=1)
best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

# Find one color for each number of PCA components used.
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
lim_mag_y = (13.2, 7)
lim_arcsec_x = (0.1, 2.5)
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


# =============================================================================
# Best number of PCA components
# =============================================================================

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
ax2.set_xlabel("Separation [FWHM]", fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)
