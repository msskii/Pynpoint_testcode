"""
File to test MultiChannelReader with simple Gaussian test data
"""

#########################
# User and device specific path for pynpoint location
import matplotlib.pyplot as plt
from pynpoint import Pypeline, MultiChannelReader
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
file_path = data_path + '/data/Gaussian_JWST/StarA_24-30'

# =============================================================================
# MultiChannelReader test
# =============================================================================

pipeline = Pypeline(work_path, in_path, out_path)

reader = MultiChannelReader(name_in="reader",
                            input_dir=file_path,
                            image_tag="cube",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("reader")

# =============================================================================
# Show test results
# =============================================================================


data = pipeline.get_data("cube")
plt.imshow(data[0])

print("\nPipeline tests\n")
print("Wave array:")
print(pipeline.get_attribute("cube", "WAV_ARR", static=False))
print(pipeline.get_attribute_full_len("cube", "WAV_ARR", static=False))
print("Pixelscale:")
print(pipeline.get_attribute("cube", "PIXSCALE", static=False))
print(pipeline.get_attribute_full_len("cube", "PIXSCALE", static=False))