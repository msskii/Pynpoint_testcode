"""
testfile to check any modules
"""
import sys
import os
import pdb
import numpy as np

#########################
# User specific path
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
#########################

sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')

#from spectres import spectres
from ifuframeselection import SelectWavelengthRangeModule
from ifubadpixel import NanFilterModule
from ifucentering_test import IFUAlignCubesModule
from ifupsfpreparation import IFUStellarSpectrumModule
from ifupsfsubtraction import IFUPSFSubtractionModule
from ifuresizing import FoldingModule
from ifustacksubset import CrossCorrelationPreparationModule
from ifucrosscorrelation import CrossCorrelationModule
from ifupcasubtraction import IFUResidualsPCAModule
from ifuresizing import UnfoldingModule
from IFS_basic_subtraction import IFS_ClassicalRefSubstraction
from center_guess import StarCenterFixedGauss, IFS_RefStarAlignment
from IFS_SimpleSubtraction import IFS_normalizeSpectrum

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, \
                    FitCenterModule, ShiftImagesModule, RemoveLinesModule, StackCubesModule, \
                    BadPixelSigmaFilterModule, ParangReadingModule, \
                    AddLinesModule, FitsWritingModule, TextWritingModule, \
                    PrimaryHDUCombiner, PaddingModule, MultiChannelReader, RefLibraryMultiReader, ArrayReadingModule


from scratch_primaryhducombiner import hducollapser
from plotter import cplot

# pdb.set_trace()

file_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/input/Level3_ch1-long_s3d.fits'
in_path = '/Users/Gian/Documents/JWST_Central-Database/Reduced_cubes/cubes_obs3_red_red'
work_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/MRS_PSF_subtraction_FS23/Data_cube_input'
out_path = '/Users/Gian/Documents/GitHub/Pynpoint_testcode/Data/output/cubes_obs3'

# headr = hducollapser(file_path,out_path)

pipeline = Pypeline(work_path, in_path, out_path)
# pdb.set_trace()
Import_Science = MultiChannelReader(name_in = "readersci",
                            input_dir = None,
                            image_tag = "science",
                            check = True,
                            ifs_data=False,
                            overwrite=True)

pipeline.add_module(Import_Science)
pipeline.run_module("readersci")

sci = pipeline.get_data("science")
nframes = int(pipeline.get_attribute("science","NFRAMES",static=False)[0])

wavelengths = pipeline.get_attribute("science","WAV_ARR",static=False)[0] # self.m_image_in_port.get_attribute('WAV_ARR')[0].astype(np.float16)
pixelscale = pipeline.get_attribute_full_len("science","PIXSCALE",static=False) # self.m_image_in_port.get_attribute('PIXSCALE')[0].astype(np.float16)
bands =  pipeline.get_attribute("science","BAND_ARR",static=False)[0] # self.m_image_in_port.get_attribute('BAND_ARR')[0].astype(str)

# print("this should print",nframes,wavelengths,pixelscale,bands)

hdr = {}
hdr["NAXIS"] = 3
hdr["WAV_ARR"] = wavelengths
hdr["PIXSCALE"] = pixelscale
hdr["BAND_ARR"] = bands

fits_header = []
fits_dict = {}
hdr_template = headers[0]
for key in list(hdr_template.keys()):
    if '\n' in key:
        continue
    tobeornottobe = True
    # if all keys have the same value we leave 'tobeornottobe' on True
    for i in np.arange(N):
        hdr_prev = headers[i-1]
        hdr = headers[i]
        if hdr[key] != hdr_prev[key]:
            tobeornottobe = False
    if key == 'NAXIS3':
        fits_header.append(f'{key} = {N_wave}')
        fits_dict[key] = N_wave
        continue
    
    if tobeornottobe:
        fits_header.append(f'{key} = {headers[0][key]}')
        fits_dict[key] = headers[0][key]
    else:
        typ = type(headers[0][key])
        stringy = False
        if typ==str:
            stringy = True
            card = np.zeros(N,dtype=object)
        else:
            card = np.zeros(N,dtype=typ)
        for i in np.arange(N):
            card[i] = headers[i][key]
        if stringy:
            card = card.astype('S') # convert to numpy bytes array which is compatible with HDF5 database
        fits_header.append(f'{key} = {card}')
        fits_dict[key] = card

# New attributes:
fits_header.append(f'WAV_ARR = {wavelength_arr}')
fits_dict['WAV_ARR'] = wavelength_arr
fits_header.append(f'CHAN_ARR = {channel_arr}')
fits_dict['CHAN_ARR'] = channel_arr
fits_header.append(f'BAND_ARR = {band_arr}')
fits_dict['BAND_ARR'] = band_arr

Export_Science = ArrayReadingModule(name_in = "readarray", 
                                    array_star = sci,
                                    header=hdr,
                                    image_tag = "test")

pipeline.add_module(Export_Science)
pipeline.run_module("readarray")

sci_new = pipeline.get_data("test")
wavarr = pipeline.get_attribute("test", "WAV_ARR", static=False)

print((sci[0]-sci_new[0]).sum(), wavarr)

