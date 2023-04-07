#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:19:18 2023

@author: Gian
"""
import numpy as np
from astropy.io import fits


def hducollapser(fits_file,out_path):
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
    hdr = fits.Header()
    hdr.extend(hdu_list[0].header)
    hdr.extend(hdu_list[1].header)
    data=hdu_list[1].data
    hdu_list.close()
    fits.writeto(out_path + '/new_fits.fits', data=data,header=hdr)
    return hdr
