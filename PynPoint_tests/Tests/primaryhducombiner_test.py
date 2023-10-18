"""
File to test primaryhducombiner with Gaussian test data with multiple HDU's
"""

#########################
# User and device specific path for pynpoint location
from astropy.io import fits
from pynpoint import Pypeline, PrimaryHDUCombiner
import sys
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
#########################


# =============================================================================
# User specified file paths
# =============================================================================
data_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_data'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Full_cubes'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_output_dump'

# Use this file for running simple testing
file_path = data_path + '/data/Gaussian_multiHDU/Star1'

# =============================================================================
# PrimaryHDUCombiner test preparation
# =============================================================================

pipeline = Pypeline(work_path, in_path, out_path)

pre_hdu = fits.open(file_path+"/Gaussian_multiHDUdata1.fits")

# =============================================================================
# Print the initial data form
# =============================================================================

print("\nBefore collapsing: Two HDUs with header information")
print("PrimaryHDU:")
print("Header")
print(pre_hdu[0].header)
print("Data shape")
print(pre_hdu[0].data)
print("ImageHDU:")
print("Header")
print(pre_hdu[1].header)
print("Data shape")
print(pre_hdu[1].data.shape)


test_module = PrimaryHDUCombiner(name_in="combiner",
                                 input_dir=file_path,
                                 output_dir=out_path,
                                 file_list=None,
                                 change_name=True,
                                 overwrite=True,
                                 primarywarn=False,
                                 only2hdus=False)
pipeline.add_module(test_module)
pipeline.run_module("combiner")

post_hdu = fits.open(out_path+"/col_Gaussian_multiHDUdata1.fits")

# =============================================================================
# Show test results
# =============================================================================
print("\n\nAfter collapsing: One HDU with header information")
print("PrimaryHDU:")
print("Header")
print(post_hdu[0].header)
print("Data shape")
print(post_hdu[0].data.shape)
