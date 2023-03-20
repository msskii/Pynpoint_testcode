"""
Test file inside Pynpoint fork
"""

# import os
# import urllib
# import matplotlib.pyplot as plt

# from pynpoint import Pypeline, FitsReadingModule, Hdf5ReadingModule, PSFpreparationModule, PcaPsfSubtractionModule

# pipeline = Pypeline(working_place_in='../../JWST_Central-Database',
#                     input_place_in='../../JWST_Central-Database/cubes_obs3',
#                     output_place_in='../../JWST_Central-Database')

# reader = FitsReadingModule(name_in='read',
#                            input_dir='../../JWST_Central-Database/cubes_obs3',
#                            image_tag='cucumber')
# pipeline.add_module(reader)

# pipeline.run()

# print(pipeline.get_data('data'))

##########################
# Pynpoint Tutorial 2

import os
import sys
import configparser
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


##### User specific path ######
sys.path.insert(1, '/Users/Gian/Documents/GitHub/Pynpoint')
###############################

from pynpoint import Pypeline, FitsReadingModule, ParangReadingModule, \
                     StarExtractionModule, BadPixelSigmaFilterModule, \
                     StarAlignmentModule, FitCenterModule, ShiftImagesModule, \
                     PSFpreparationModule, PcaPsfSubtractionModule, \
                     FalsePositiveModule, SimplexMinimizationModule, \
                     FakePlanetModule, ContrastCurveModule, Hdf5ReadingModule, \
                     FitsWritingModule, TextWritingModule

sys.path.insert(2, '/Users/Gian/Documents/GitHub/student-code/MRS_PSF_subtraction_FS23/Data_cube_input')


pipeline = Pypeline(working_place_in='./',
                    input_place_in='./Data',
                    output_place_in='./')

# config = configparser.ConfigParser()
# config.add_section('header')
# config.add_section('settings')
# config['settings']['PIXSCALE'] = '0.0036'
# config['settings']['MEMORY'] = 'None'
# config['settings']['CPU'] = '1'

# with open('PynPoint_config.ini', 'w') as configfile:
#     config.write(configfile)

module = FitsReadingModule(name_in='read',
                            input_dir=None,
                            image_tag='testimg',
                            overwrite=True,
                            check=True,
                            filenames=None,
                            ifs_data=False)

pipeline.add_module(module)
# pipeline.run_module('read')

import plotter as plot

plot.plot(pipeline.get_data('testimg')[1],title="testdata",vmax=2000) # vmax is purposely lowered to see the PSF

module = PSFpreparationModule(name_in='prep',
                              image_in_tag='testimg',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=0.01,
                              edge_size=0.1)


pipeline.add_module(module)
# pipeline.run_module('prep')

module = PcaPsfSubtractionModule(pca_numbers=[20, ],
                                  name_in='pca',
                                  images_in_tag='prep',
                                  reference_in_tag='prep',
                                  res_median_tag='residuals')

pipeline.add_module(module)
# pipeline.get_attribute('prep','PARANG')
pipeline.run()

##########################
# # Pynpoint Tutorial 2

# urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/hd142527_zimpol_h-alpha.tgz',
#                             'hd142527_zimpol_h-alpha.tgz')

# tar = tarfile.open('hd142527_zimpol_h-alpha.tgz')
# tar.extractall(path='input')

# pipeline = Pypeline(working_place_in='./',
#                     input_place_in='input/',
#                     output_place_in='./testfile_output')

# config = configparser.ConfigParser()
# config.add_section('header')
# config.add_section('settings')
# config['settings']['PIXSCALE'] = '0.0036'
# config['settings']['MEMORY'] = 'None'
# config['settings']['CPU'] = '1'

# with open('PynPoint_config.ini', 'w') as configfile:
#     config.write(configfile)

# module = FitsReadingModule(name_in='read',
#                             input_dir=None,
#                             image_tag='a1',
#                             overwrite=True,
#                             check=False,
#                             filenames=None,
#                             ifs_data=False)

# pipeline.add_module(module)
# pipeline.run_module('read')



##########################
# # Pynpoint Tutorial 1

# urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/betapic_naco_mp.hdf5',
#                             './Data/betapic_naco_mp.hdf5')

# pipeline = Pypeline(working_place_in='./',
#                     input_place_in='./Data',
#                     output_place_in='./testfile_output')

# module = Hdf5ReadingModule(name_in='read',
#                             input_filename='betapic_naco_mp.hdf5',
#                             input_dir=None,
#                             tag_dictionary={'stack': 'stack'})

# pipeline.add_module(module)

# module = PSFpreparationModule(name_in='prep',
#                               image_in_tag='stack',
#                               image_out_tag='prep',
#                               mask_out_tag=None,
#                               norm=False,
#                               resize=None,
#                               cent_size=0.15,
#                               edge_size=1.1)

# pipeline.add_module(module)

# module = PcaPsfSubtractionModule(pca_numbers=[20, ],
#                                   name_in='pca',
#                                   images_in_tag='prep',
#                                   reference_in_tag='prep',
#                                   res_median_tag='residuals')

# pipeline.add_module(module)


# pipeline.run()

# residuals = pipeline.get_data('residuals')

# pixscale = pipeline.get_attribute('residuals', 'PIXSCALE')
# print(f'Pixel scale = {pixscale*1e3} mas')

# size = pixscale * residuals.shape[-1]/2.

# plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
# plt.xlabel('RA offset (arcsec)', fontsize=14)
# plt.ylabel('Dec offset (arcsec)', fontsize=14)
# cb = plt.colorbar()
# cb.set_label('Flux (ADU)', size=14.)    

