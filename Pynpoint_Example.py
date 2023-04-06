import os
import sys
import urllib
import matplotlib.pyplot as plt
##### User specific path ######
sys.path.insert(1, '/Users/Gian/Documents/GitHub/Pynpoint')
###############################
from pynpoint import Pypeline, Hdf5ReadingModule, PSFpreparationModule, PcaPsfSubtractionModule

urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/betapic_naco_mp.hdf5','./input/betapic_naco_mp.hdf5')

pipeline = Pypeline(working_place_in='./processing',
                    input_place_in='./input',
                    output_place_in='./output')

module = Hdf5ReadingModule(name_in='read',
                           input_filename='betapic_naco_mp.hdf5',
                           input_dir=None,
                           tag_dictionary={'stack': 'stack'})

pipeline.add_module(module)

module = PSFpreparationModule(name_in='prep',
                              image_in_tag='stack',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=0.15,
                              edge_size=1.1)

pipeline.add_module(module)

module = PcaPsfSubtractionModule(pca_numbers=[20, ],
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals')

pipeline.add_module(module)


pipeline.run()

residuals = pipeline.get_data('residuals')
pixscale = pipeline.get_attribute('residuals', 'PIXSCALE')
print(f'Pixel scale = {pixscale*1e3} mas')


size = pixscale * residuals.shape[-1]/2.

plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
plt.xlabel('RA offset (arcsec)', fontsize=14)
plt.ylabel('Dec offset (arcsec)', fontsize=14)
cb = plt.colorbar()
cb.set_label('Flux (ADU)', size=14.)

plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
plt.xlabel('RA offset (arcsec)', fontsize=14)
plt.ylabel('Dec offset (arcsec)', fontsize=14)
cb = plt.colorbar()
cb.set_label('Flux (ADU)', size=14.)

plt.show()
