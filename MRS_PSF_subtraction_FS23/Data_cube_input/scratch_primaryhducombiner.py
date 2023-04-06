#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:19:18 2023

@author: Gian
"""
import numpy as np
from astropy.io import fits


def hducollapser(fits_file):
    """
    Function doesnt allow for multiple datasets in single fits file!
    No errors for no data found, etc.

    Parameters
    ----------
    fits_file : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    header : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    hdu_list = fits.open(fits_file)
    main = int(0)
    
    for i in np.arange(len(hdu_list)):
        if hdu_list[i].data is not None:
            main = i
            images = hdu_list[i].data.byteswap().newbyteorder()
            break

    images = np.nan_to_num(images)

    header = hdu_list[int(main)].header
    
    

    fits_header = []
    for key in header:
        fits_header.append(f'{key} = {header[key]}')

    hdu_list.close()

    return header, images.shape
