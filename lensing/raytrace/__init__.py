import numpy as np
from scipy import interpolate
from ..helper_functions import param2array, array2param

def delens(ix,iy,deflectx,deflecty,ratio=1,dx=0,dy=0):
    '''
    delenses a set of x and y coordinates in the image plane

    Parameters:
    ix,iy: scalar or array-like, image plane x,y coordinates (units: pixels)
    deflectx,deflecty: 2d numpy array, deflection matrices (units: pixels)
    ratio: scalar scaling factor for the deflection matrices
        ex. if defl. maps are @ z = 2 and your source is z = 4:
            ratio = dls/ds(z=4) / dls/ds(z=2)
    dx,dy: scalars, offset in x-y pixels between grid that ix,iy
        are on and where your deflection matrices start
        if both are on the same grid, then dx=dy=0

    returns: srcx,srcy, numpy array of source plane x,y positions
    '''
    ix,xptype = param2array(ix)
    iy,yptype = param2array(iy)
    x = ix-dx
    y = iy-dy

    n = len(ix)
    # load deflection matrices and convert from arcsec -> pix
    dplx = deflectx*ratio
    dply = deflecty*ratio

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

def lens(xs,ys,deflectx,deflecty,ratio=1,dx=0,dy=0,maxdist=0.2,return_dist=False):
    '''
    lenses a set of x and y coordinates in the source plane to the image plane

    Parameters:
    xs,ys: array-like, source plane x,y coordinates (units: pixels)
    deflectx,deflecty: 2d numpy array, deflection matrices (units: pixels)
    ratio: scalar, scaling factor for the deflection matrices
        ex. if defl. maps are @ z = 2 and your source is @ z = 4:
            ratio = dls/ds(z=4) / dls/ds(z=2)
    dx,dy: scalar, offset in x-y pixels between grid that ix,iy
        are on and where your deflection matrices start
        if both are on the same grid, then dx=dy=0
    maxdist: scalar, the maximum distance in the source plane for which pixels are considered
        associated with the same source plane position
        -setting this value too high may cause nearby images to merge into "giant arcs"
        -setting this value too low may cause demagnified images to not appear
    return_dist (boolean): if True, the function returns a third variable representing the
        distance from the desired source plane position for each image pixel

    Returns:
        ximp,yimp(,dist): numpy arrays of image plane position x,y (source plane distance)
    '''

    xs,xptype = param2array(xs)
    ys,yptype = param2array(ys)
    xs = xs-dx
    ys = ys-dy
    
    dims = deflectx.shape
    source_x = np.zeros_like(deflectx)
    source_y = np.zeros_like(deflecty)
    if dims[0] == dims[1]:
        for i in range(dims[0]):
            source_x[:,i] = i + 1 - deflectx[:,i]*ratio
            source_y[i,:] = i + 1 - deflecty[i,:]*ratio
    else:
        for j in range(dims[0]): source_x[:,j] = j + 1 - deflectx[:,j]*ratio
        for k in range(dims[1]): source_y[k,:] = k + 1 - deflecty[k,:]*ratio
    
    d = np.sqrt((source_x-xs)**2+(source_y-ys)**2)
    indices = np.where(d<maxdist)
    dist = d[indices]
    
    ximp = []
    yimp = []
    for i,j in zip(indices[1],indices[0]): ximp.append(i+1),yimp.append(j+1)
    ximp = np.array(ximp)+dx
    yimp = np.array(yimp)+dy
    
    if return_dist:
        return ximp, yimp, dist
    else:
        return ximp, yimp

def lens_fine(x,y,deflectx,deflecty,xa,ya,ratio=1,dx=0,dy=0):
    x,xptype = param2array(x)
    y,yptype = param2array(y)
    x -= dx
    y -= dy
    

    dplx = deflectx*ratio
    dply = deflecty*ratio

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
    else:
        for j in range(dims[0]): source_x[:,j] = j + 1 - dplx[:,j]
        for k in range(dims[1]): source_y[k,:] = k + 1 - dply[k,:]

    X,Y = np.meshgrid(np.arange(dplx.shape[1]),np.arange(dplx.shape[0]))
    conditions = np.array([X >= xa[0]-dx,
                           X <  xa[1]-dx,
                           Y >= ya[0]-dy,
                           Y <  ya[1]-dy])
    pixels = np.all(conditions,axis=0)

    ix = np.zeros(x.size)
    iy = np.zeros(y.size)

    for i in range(x.size):
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

        deflectx = dplx_interp(ixval,iyval)
        deflecty = dply_interp(ixval,iyval)

        sxgrid = ixgrid - deflectx
        sygrid = iygrid - deflecty

        dist_fine = (sxgrid - x[i])**2+(sygrid - y[i])**2
        new_closest = np.where(dist_fine.flat == np.amin(dist_fine))[0][0]

        ix[i] = ixgrid.flat[new_closest]
        iy[i] = iygrid.flat[new_closest]

    ix = array2param(ix,xptype)+dx
    iy = array2param(iy,yptype)+dy

    return ix,iy