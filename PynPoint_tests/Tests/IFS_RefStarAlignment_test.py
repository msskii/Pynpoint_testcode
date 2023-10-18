"""
File to test IFS_RefStarAlignment with simple Gaussian test data
"""

#########################
# User and device specific path for pynpoint location
import matplotlib.pyplot as plt
from center_guess import IFS_RefStarAlignment
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

# Use these file for running simple testing
test_data_path = data_path + '/data/Gaussian_JWST'

# =============================================================================
# IFS_RefStarAlignment test preparation
# =============================================================================

pipeline = Pypeline(work_path, in_path, out_path)

reader = MultiChannelReader(name_in="readerA",
                            input_dir=test_data_path+'/StarA_24-30',
                            image_tag="cubeA",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("readerA")

reader = MultiChannelReader(name_in="readerB",
                            input_dir=test_data_path+'/StarB_24-26',
                            image_tag="cubeB",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("readerB")

reader = MultiChannelReader(name_in="readerC",
                            input_dir=test_data_path+'/StarC_26-29',
                            image_tag="cubeC",
                            only2hdus=True)
pipeline.add_module(reader)
pipeline.run_module("readerC")

# =============================================================================
# IFS_RefStarAlignment test (L1,spline,Circle)
# =============================================================================

aligner = IFS_RefStarAlignment(name_in="alignerL1",
                               sci_in_tag="cubeB",
                               ref_in_tags=["cubeC", "cubeA"],
                               fit_out_tag_suff="al1",
                               qual_method="L1",
                               seeApert=True,
                               interpol="spline",
                               apertshap="Circle")
pipeline.add_module(aligner)
pipeline.run_module("alignerL1")

sci_data = pipeline.get_data("cubeB")[0]
plt.imshow(sci_data)
plt.colorbar()
plt.show()

ref_data_A1 = pipeline.get_data("cubeA_al1")[0]
plt.imshow(sci_data-ref_data_A1)
plt.colorbar()
plt.show()

ref_data_C1 = pipeline.get_data("cubeC_al1")[0]
plt.imshow(sci_data-ref_data_C1)
plt.colorbar()
plt.show()

# =============================================================================
# IFS_RefStarAlignment test (L2,bilinear,Circle)
# =============================================================================

aligner = IFS_RefStarAlignment(name_in="alignerL2",
                               sci_in_tag="cubeB",
                               ref_in_tags=["cubeC", "cubeA"],
                               fit_out_tag_suff="al2",
                               qual_method="L2",
                               seeApert=False,
                               interpol="bilinear",
                               apertshap="Circle")
pipeline.add_module(aligner)
pipeline.run_module("alignerL2")

sci_data = pipeline.get_data("cubeB")[0]
plt.imshow(sci_data)
plt.colorbar()
plt.show()

ref_data_A2 = pipeline.get_data("cubeA_al2")[0]
plt.imshow(sci_data-ref_data_A2)
plt.colorbar()
plt.show()

ref_data_C2 = pipeline.get_data("cubeC_al2")[0]
plt.imshow(sci_data-ref_data_C2)
plt.colorbar()
plt.show()

# =============================================================================
# IFS_RefStarAlignment test (MultMax,fft,Circle)
# =============================================================================

aligner = IFS_RefStarAlignment(name_in="alignermult",
                               sci_in_tag="cubeB",
                               ref_in_tags=["cubeC", "cubeA"],
                               fit_out_tag_suff="al3",
                               qual_method="MultMax",
                               seeApert=False,
                               interpol="spline",
                               apertshap="Circle")
pipeline.add_module(aligner)
pipeline.run_module("alignermult")

sci_data = pipeline.get_data("cubeB")[0]
plt.imshow(sci_data)
plt.colorbar()
plt.show()

ref_data_A3 = pipeline.get_data("cubeA_al3")[0]
plt.imshow(sci_data-ref_data_A3)
plt.colorbar()
plt.show()

ref_data_C3 = pipeline.get_data("cubeC_al3")[0]
plt.imshow(sci_data-ref_data_C3)
plt.colorbar()
plt.show()

# =============================================================================
# IFS_RefStarAlignment test (L2,spline,Ring)
# =============================================================================

aligner = IFS_RefStarAlignment(name_in="alignerring",
                               sci_in_tag="cubeB",
                               ref_in_tags=["cubeC", "cubeA"],
                               fit_out_tag_suff="alring",
                               qual_method="L2",
                               seeApert=True,
                               interpol="spline",
                               in_rad=0.5,
                               out_rad=2.5,
                               apertshap="Ring")
pipeline.add_module(aligner)
pipeline.run_module("alignerring")

sci_data = pipeline.get_data("cubeB")[0]
plt.imshow(sci_data)
plt.colorbar()
plt.show()

ref_data_Ar = pipeline.get_data("cubeA_alring")[0]
plt.imshow(sci_data-ref_data_Ar)
plt.colorbar()
plt.show()

ref_data_Cr = pipeline.get_data("cubeC_alring")[0]
plt.imshow(sci_data-ref_data_Cr)
plt.colorbar()
plt.show()

# =============================================================================
# IFS_RefStarAlignment test (L2,spline,full frame)
# =============================================================================

aligner = IFS_RefStarAlignment(name_in="alignerfull",
                               sci_in_tag="cubeB",
                               ref_in_tags=["cubeC", "cubeA"],
                               fit_out_tag_suff="alfull",
                               qual_method="L2",
                               seeApert=True,
                               interpol="spline",
                               in_rad=0.5,
                               out_rad=2.5,
                               apertshap=None)
pipeline.add_module(aligner)
pipeline.run_module("alignerfull")

sci_data = pipeline.get_data("cubeB")[0]
plt.imshow(sci_data)
plt.colorbar()
plt.show()

ref_data_Af = pipeline.get_data("cubeA_alfull")[0]
plt.imshow(sci_data-ref_data_Af)
plt.colorbar()
plt.show()

ref_data_Cf = pipeline.get_data("cubeC_alfull")[0]
plt.imshow(sci_data-ref_data_Cf)
plt.colorbar()
plt.show()
