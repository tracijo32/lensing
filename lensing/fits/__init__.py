from astropy.io import fits
import numpy as np
import astropy.units as u

def pixscale(fitsfile):
    '''
    returns the size of one pixel of a fits file from the header
    assumes that the pixel scale is the same for both axes
    '''
    ps = np.abs(fits.getval(fitsfile,'CDELT1'))*3600
    return ps
    
def fitsfix_6pt5(fitsfile):
    '''fixes the wcs header of a lenstool file generated using version 6.5'''
    fits.setval(fitsfile,'CRPIX1',value=fits.getval(fitsfile,'CRPIX1')+0.5)
    fits.setval(fitsfile,'CRPIX2',value=fits.getval(fitsfile,'CRPIX2')+0.5)

