"""
File to test DataCubeReplacer
"""

#########################
# User and device specific path for pynpoint location
import matplotlib.pyplot as plt
import numpy as np
from pynpoint import Pypeline, MultiChannelReader, DataCubeReplacer
import sys
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
#########################


# =============================================================================
# User specified file paths
# =============================================================================
data_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_data'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Full_cubes'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = work_path

# Use this file for running simple testing
file_path = data_path + '/data/Gaussian_JWST/StarB_24-26'

# =============================================================================
# Test preparation
# =============================================================================

pipeline = Pypeline(work_path, in_path, out_path)

reader = MultiChannelReader(name_in="reader",
                            input_dir=file_path,
                            image_tag="cube",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("reader")

# =============================================================================
# Create test array
# =============================================================================

replace_arr = np.zeros((23, 55, 55))
replace_arr[:, :, 20:] = 1

# =============================================================================
# Replace data cube test
# =============================================================================

replacer = DataCubeReplacer(name_in="replacer",
                            image_in_tag="cube",
                            image_out_tag="replaced_cube",
                            new_cube=replace_arr)
pipeline.add_module(replacer)
pipeline.run_module("replacer")

# =============================================================================
# Show test results
# =============================================================================


before_data = pipeline.get_data("cube")
after_data = pipeline.get_data("replaced_cube")
print("\nData replacement test:")
print("\nBefore data cube replacement")
plt.imshow(before_data[0])
plt.show()

print("\nAfter data replacement")
plt.imshow(after_data[0])
plt.show()

# =============================================================================
# Test warnings output -> potential implementation of certain attributes
#                         that can be changed manually could be done in the future
# =============================================================================

# Different wavelengths
diff_wav_arr = np.zeros((10, 55, 55))
diff_wav_arr[:, 30:, :] = 1
replacer = DataCubeReplacer(name_in="replacer_wav",
                            image_in_tag="cube",
                            image_out_tag="replaced_cube_wav",
                            new_cube=diff_wav_arr)
pipeline.add_module(replacer)
pipeline.run_module("replacer_wav")

# Different spatial dimensions
diff_spatial_arr = np.zeros((23, 50, 60))
diff_spatial_arr[:, :29, :] = 1
replacer = DataCubeReplacer(name_in="replacer_spat",
                            image_in_tag="cube",
                            image_out_tag="replaced_cube_spat",
                            new_cube=diff_spatial_arr)
pipeline.add_module(replacer)
pipeline.run_module("replacer_spat")

# Two dimensional
twodim_arr = np.zeros((55, 55))
twodim_arr[:, :26] = 1
replacer = DataCubeReplacer(name_in="replacer_twod",
                            image_in_tag="cube",
                            image_out_tag="replaced_cube_twod",
                            new_cube=twodim_arr)
pipeline.add_module(replacer)
pipeline.run_module("replacer_twod")

# One dimensional -> will raise a ValueError, hence this code will be commented out
# onedim_arr = np.zeros((55))
# onedim_arr[:26] = 1
# replacer = DataCubeReplacer(name_in="replacer_oned",
#                             image_in_tag="cube",
#                             image_out_tag="replaced_cube_oned",
#                             new_cube=onedim_arr)
# pipeline.add_module(replacer)
# pipeline.run_module("replacer_oned")
