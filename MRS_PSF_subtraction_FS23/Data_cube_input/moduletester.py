"""
testfile to check any modules
"""
import sys

#########################
# User specific path
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
#########################

from scratch_primaryhducombiner import hducollapser

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output'

headr = hducollapser(file_path,out_path)
