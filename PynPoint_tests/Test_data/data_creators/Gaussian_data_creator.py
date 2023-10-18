"""
File to create Test data for PynPoint in the style of JWST MIRI data.
"""

import numpy as np
from scipy import signal
from astropy.io import fits

# Set JWST to True if channel and band are supposed to be added as well as wavelength start and increment
JWST = True
nr_HDU = 2

# Set center to where the Gaussian is centered
center = (24,26)

N1 = np.linspace(35,45,num=6)

file1_list = []
for N in N1:
    n = int(N)
    k1d = signal.gaussian(n, std=n//8).reshape(n, 1)
    kernel = np.outer(k1d, k1d)
    frame = np.zeros((49, 54))
    frame[center[0],center[1]] = 1    # random
    row, col = np.where(frame == 1)
    frame[row[0]-(n//2):row[0]+(n//2)+1, col[0]-(n//2):col[0]+(n//2)+1] = kernel
    file1_list.append(frame)
file1 = np.stack(file1_list,axis=0)


N2 = np.linspace(29,49,num=11)

file2_list = []
for N in N2:
    n = int(N)
    k2d = signal.gaussian(n, std=n//8).reshape(n, 1)
    kernel = np.outer(k2d, k2d)
    frame = np.zeros((50, 53))
    frame[center[0],center[1]] = 1    # random
    row, col = np.where(frame == 1)
    frame[row[0]-(n//2):row[0]+(n//2)+1, col[0]-(n//2):col[0]+(n//2)+1] = kernel
    file2_list.append(frame)
file2 = np.stack(file2_list,axis=0)


N3 = np.linspace(39,49,num=6)

file3_list = []
for N in N3:
    n = int(N)
    k3d = signal.gaussian(n, std=n//8).reshape(n, 1)
    kernel = np.outer(k3d, k3d)
    frame = np.zeros((55, 55))
    frame[center[0],center[1]] = 1    # random
    row, col = np.where(frame == 1)
    frame[row[0]-(n//2):row[0]+(n//2)+1, col[0]-(n//2):col[0]+(n//2)+1] = kernel
    file3_list.append(frame)
file3 = np.stack(file3_list,axis=0)

hdr1 = fits.Header()
hdr1['Name'] = "Gaussian test star 1"
hdr1['CenterX'] = 24
hdr1['CenterY'] = 30
hdr2 = fits.Header()
hdr2['Name'] = "Gaussian test star 2"
hdr2['CenterX'] = 24
hdr2['CenterY'] = 30
hdr3 = fits.Header()
hdr3['Name'] = "Gaussian test star 3"
hdr3['CenterX'] = 24
hdr3['CenterY'] = 30

if nr_HDU==2:
    h1a = fits.PrimaryHDU(data=None,header=hdr1)
    h2a = fits.PrimaryHDU(data=None,header=hdr2)
    h3a = fits.PrimaryHDU(data=None,header=hdr3)
    hdr1 = fits.Header()
    hdr2 = fits.Header()
    hdr3 = fits.Header()
    if JWST:
        hdr1['CHANNEL'] = 1
        hdr1['BAND'] = "LONG"
        hdr1['CRVAL3'] = 5.6
        hdr1['CDELT3'] = 0.009
        hdr1['CDELT1'] = 3.6e-5
        hdr1['INSTRUME'] = "MIRI"
        hdr2['CHANNEL'] = 2
        hdr2['BAND'] = "MEDIUM"
        hdr2['CRVAL3'] = 5.7
        hdr2['CDELT3'] = 0.009
        hdr2['CDELT1'] = 3.7e-5
        hdr2['INSTRUME'] = "MIRI"
        hdr3['CHANNEL'] = 3
        hdr3['BAND'] = "SHORT"
        hdr3['CRVAL3'] = 7
        hdr3['CDELT3'] = 0.009
        hdr3['CDELT1'] = 3.8e-5
        hdr3['INSTRUME'] = "MIRI"
    h1b = fits.ImageHDU(data=file1,header=hdr1)
    h2b = fits.ImageHDU(data=file2,header=hdr2)
    h3b = fits.ImageHDU(data=file3,header=hdr3)
    hdul1 = fits.HDUList(hdus=[h1a,h1b])
    hdul2 = fits.HDUList(hdus=[h2a,h2b])
    hdul3 = fits.HDUList(hdus=[h3a,h3b])

# =============================================================================
# User specified file paths
# =============================================================================
hdul1.writeto('/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_data/data/Gaussian_multiHDU/Star1/Gaussian_multiHDUdata1.fits',overwrite=True)
hdul2.writeto('/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_data/data/Gaussian_multiHDU/Star1/Gaussian_multiHDUdata2.fits',overwrite=True)
hdul3.writeto('/Users/Gian/Documents/GitHub/Pynpoint_testcode/PynPoint_tests/Test_data/data/Gaussian_multiHDU/Star1/Gaussian_multiHDUdata3.fits',overwrite=True)