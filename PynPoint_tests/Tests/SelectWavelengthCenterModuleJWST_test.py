"""
File to test SelectWavelengthCenterModuleJWST with simple Gaussian test data
"""

#########################
# User and device specific path for pynpoint and pynpoint_ifs location (not necessary if installed as package)
import matplotlib.pyplot as plt
from jwstframeselection import SelectWavelengthCenterModuleJWST
from pynpoint import Pypeline, MultiChannelReader
import sys
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
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
# SelectWavelengthCenterModuleJWST test preparation
# =============================================================================

pipeline = Pypeline(work_path, in_path, out_path)

reader = MultiChannelReader(name_in="reader",
                            input_dir=file_path,
                            image_tag="cube",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("reader")

# =============================================================================
# SelectWavelengthCenterModuleJWST test
# =============================================================================

wavselect = SelectWavelengthCenterModuleJWST(name_in="selector",
                                             image_in_tag="cube",
                                             image_out_tag="selected_cube",
                                             nr_frames=5,
                                             wave_center=5.63)

pipeline.add_module(wavselect)
pipeline.run_module("selector")

# Edge case 1: wave_center at left corner
wavselect = SelectWavelengthCenterModuleJWST(name_in="selector1",
                                             image_in_tag="cube",
                                             image_out_tag="selected_cube1",
                                             nr_frames=6,
                                             wave_center=5.6)
pipeline.add_module(wavselect)
pipeline.run_module("selector1")

# Edge case 2: wave_center at right corner
wavselect = SelectWavelengthCenterModuleJWST(name_in="selector2",
                                             image_in_tag="cube",
                                             image_out_tag="selected_cube2",
                                             nr_frames=10,
                                             wave_center=7.045)
pipeline.add_module(wavselect)
pipeline.run_module("selector2")

# Edge case 3: too many frames to be selected
wavselect = SelectWavelengthCenterModuleJWST(name_in="selector3",
                                             image_in_tag="cube",
                                             image_out_tag="selected_cube3",
                                             nr_frames=50,
                                             wave_center=5.7)
pipeline.add_module(wavselect)
pipeline.run_module("selector3")

# Edge case 4: out of bounds wave center
wavselect = SelectWavelengthCenterModuleJWST(name_in="selector4",
                                             image_in_tag="cube",
                                             image_out_tag="selected_cube4",
                                             nr_frames=12,
                                             wave_center=1)
pipeline.add_module(wavselect)
pipeline.run_module("selector4")

# =============================================================================
# Show test results
# =============================================================================
print("\n Test Results \n")


data = pipeline.get_data("selected_cube")
plt.imshow(data[0])
plt.show()

print("Original wavelengths: ",
      pipeline.get_attribute("cube", "WAV_ARR", static=False))
print("Nr_frames: 5, wave_center: 5.65 -> ",
      pipeline.get_attribute("selected_cube", "WAV_ARR", static=False))
print("Edge case 1: corner left -> ",
      pipeline.get_attribute("selected_cube1", "WAV_ARR", static=False))
print("Edge case 2: corner right -> ",
      pipeline.get_attribute("selected_cube2", "WAV_ARR", static=False))
print("Edge case 3: too many frames -> ",
      pipeline.get_attribute("selected_cube3", "WAV_ARR", static=False))
print("Edge case 4: out of bounds center -> ",
      pipeline.get_attribute("selected_cube4", "WAV_ARR", static=False))

print("\nAttribute reduction test: \n")
print("NAXISA:")
print("Before reduction: ",
      pipeline.get_attribute("cube", "NAXISA", static=False))
print("Reduction to 5 frames: ",
      pipeline.get_attribute("selected_cube", "NAXISA", static=False))
print("Reduction to 12 frames: ",
      pipeline.get_attribute("selected_cube4", "NAXISA",static=False))