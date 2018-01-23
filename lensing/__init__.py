'''
lensing.py
author: Traci Johnson
date: 3/17/2016

Library of useful lensing functions
'''
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate

def dlsds(zl,zs,Om0=0.3,H0=70):
    # computes the lensing ratio for a lens redshift and source redshift
    # and for a given flat Lambda CDM cosmology
    cosmo = FlatLambdaCDM(Om0=Om0,H0=H0)

    dls = cosmo.angular_diameter_distance_z1z2(zl,zs).value
    ds = cosmo.angular_diameter_distance(zs).value

    return dls/ds

def delens(ix,iy,deflectx,deflecty,dlsds=1,dx=0,dy=0):
    # delenses a set of x and y coordinates in the image plane
    # dx,dy are offsets to origin if deflection matrices are cropped
    # ps=pixelscale of image

    x = ix-dx
    y = iy-dy

    n = len(ix)
    # load deflection matrices and convert from arcsec -> pix
    dplx = deflectx*dlsds
    dply = deflecty*dlsds

    srcx = np.zeros(n)
    srcy = np.zeros(n)

    # create interpolation functions for the deflection matrices
    xpixvals = np.arange(dplx.shape[0])
    ypixvals = np.arange(dply.shape[1])
    dplx_interp = interpolate.interp2d(xpixvals,ypixvals,dplx)
    dply_interp = interpolate.interp2d(xpixvals,ypixvals,dply)

    for i in range(n):
        deflectx = dplx_interp(x[i]-1,y[i]-1)[0]
        deflecty = dply_interp(x[i]-1,y[i]-1)[0]
        srcx[i] = x[i] - deflectx
        srcy[i] = y[i] - deflecty
   
    srcx = srcx+dx
    srcy = srcy+dy
 
    return srcx,srcy

def relens(x,y,deflectx,deflecty,xa,ya,dlsds=1,dx=0,dy=0):
    ## ray trace specific locations in the source plane back to the image plane

    x = x-dx
    y = y-dy

    dplx = deflectx*dlsds
    dply = deflecty*dlsds

    # create interpolation functions for the deflection matrices
    xpixvals = np.arange(dplx.shape[0])
    ypixvals = np.arange(dply.shape[1])
    dplx_interp = interpolate.interp2d(xpixvals,ypixvals,dplx)
    dply_interp = interpolate.interp2d(xpixvals,ypixvals,dply)

    ## find the source plane position of every pixel in the image plane 
    dims = dplx.shape
    source_x = dplx*0
    source_y = dply*0
    if dims[0] == dims[1]:
        for i in range(dims[0]):
            source_x[:,i] = i + 1 - dplx[:,i]
            source_y[i,:] = i + 1 - dply[i,:]
            #source_x[:,i] = i - dplx[:,i]
            #source_y[i,:] = i - dply[i,:]
    else:
        for j in range(dims[0]): source_x[:,j] = j + 1 - dplx[:,j]
        for k in range(dims[1]): source_y[k,:] = k + 1 - dply[k,:]
        #for j in range(dims[0]): source_x[:,j] = j - dplx[:,j]
        #for k in range(dims[1]): source_y[k,:] = k - dply[k,:]

    X,Y = np.meshgrid(np.arange(dplx.shape[1]),np.arange(dplx.shape[0]))
    conditions = np.array([X >= xa[0]-dx,
                           X <  xa[1]-dx,
                           Y >= ya[0]-dy,
                           Y <  ya[1]-dy])
    pixels = np.all(conditions,axis=0)

    #for i in range(n):
    #    deflectx = dplx_interp(x[i]-1,y[i]-1)[0]
    #    deflecty = dply_interp(x[i]-1,y[i]-1)[0]
    #    srcx[i] = x[i] - deflectx
    #    srcy[i] = y[i] - deflecty

    ix = np.zeros(x.size)
    iy = np.zeros(y.size)

    for i in range(len(x)):
        dist = (source_x-x[i])**2+(source_y-y[i])**2
        closest = np.where(dist[pixels].flat == np.amin(dist[pixels]))

        # find the approximate position of the source in the image plane
        ix_close = x[i] + (dplx[pixels]).flat[closest]
        iy_close = y[i] + (dply[pixels]).flat[closest]

        # trace around the approximate position until you find a spot really close to source plane
        gridsize = 1001
        ixval = np.linspace(ix_close-0.5,ix_close+0.5,gridsize)
        iyval = np.linspace(iy_close-0.5,iy_close+0.5,gridsize)
        ixgrid,iygrid = np.meshgrid(ixval,iyval)

        #deflectx = dplx_interp(ixval-1,iyval-1)
        #deflecty = dply_interp(ixval-1,iyval-1)
        deflectx = dplx_interp(ixval,iyval)
        deflecty = dply_interp(ixval,iyval)

        sxgrid = ixgrid - deflectx
        sygrid = iygrid - deflecty

        dist_fine = (sxgrid - x[i])**2+(sygrid - y[i])**2
        new_closest = np.where(dist_fine.flat == np.amin(dist_fine))[0][0]

        ix[i] = ixgrid.flat[new_closest]
        iy[i] = iygrid.flat[new_closest]

    ix = ix+dx
    iy = iy+dy

    return ix,iy

def get_mag(kappa, gamma, zl, zs, Om0=0.3, H0=70):
    r = dlsds(zl,zs,Om0=Om0,H0=H0)
    A = (1-kappa*r)**2-(gamma*r)**2
    return 1.0/np.abs(A)


