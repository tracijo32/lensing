import numpy as np
import astropy.units as u
import astropy.constants as const
import warnings
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

def jacobian_kg(kappa,gamma,ratio=1,absolute=True):
    '''
    computes the Jacobian matrix from kappa and gamma matrices
    '''
    A = (1-kappa*ratio)**2-(gamma*ratio)**2
    if absolute: A = np.abs(A)
    return A
    
def jacobian_dxdy(deflectx,deflecty,ratio=1,dunit=1,absolute=True):
    '''
    computes the Jacobian matrix from deflection matrices
    '''
    A = (1-kappa*ratio)**2-(gamma*ratio)**2
    if absolute: A = np.abs(A)
    return A

def radial_eigenvalue_kg(kappa,gamma,ratio=1):
    '''
    computes the radial eigenvalue of the Jacobian matrix from kappa and gamma
    '''
    return (1-kappa*ratio+gamma*ratio)

def tangential_eigenvalue_kg(kappa,gamma,ratio=1):
    '''
    computes the tangential eigenvalue of the Jacobian matrix from kappa and gamma
    '''
    return (1-kappa*ratio-gamma*ratio)

def radial_eigenvalue_dxdy(kappa,gamma,ratio=1,dunit=1):
    '''
    computes the radial eigenvalue of the Jacobian matrix from the deflection matrices
    '''
    kappa,gamma = get_aps(deflectx,deflecty,ratio=ratio,dunit=dunit)
    return (1-kappa*ratio+gamma*ratio)

def tangential_eigenvalue_dxdy(deflectx,deflecty,ratio=1,dunit=1):
    '''
    computes the tangential eigenvalue of the Jacobian matrix from the deflection matrices
    '''
    kappa,gamma = get_maps(deflectx,deflecty,ratio=ratio,dunit=dunit)
    return (1-kappa*ratio-gamma*ratio)

def magnification_kg(kappa,gamma,ratio=1,absolute=True):
    '''
    computes the magnification from kappa and gamma matrices
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A = (1-kappa*ratio)**2-(gamma*ratio)**2
        if absolute: A = np.abs(A)
        return 1.0/A

def magnification_dxdy(deflectx,deflecty,ratio=1,dunit=1,absolute=True):
    '''
    computes the magnification from deflection matrices (in units of pixels)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
        dDYdy,dDYdx = np.gradient(deflecty*ratio/dunit)
        A = (1-dDXdx)*(1-dDYdy)-dDXdy*dDYdx
        if absolute: A = np.abs(A)
        return 1.0/A

def get_kappa(deflectx,deflecty,ratio=1,dunit=1):
    '''
    computes kappa from deflection matrices
    '''
    dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
    dDYdy,dDYdx = np.gradient(deflecty*ratio/dunit)
    return 0.5*(dDXdx+dDYdy)

def get_gamma1(deflectx,deflecty,ratio=1,dunit=1):
    '''
    computes first component of gamma from deflection matrices
    '''
    dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
    dDYdy,dDYdx = np.gradient(deflecty*ratio/dunit)
    return 0.5*(dDXdx-dDYdy)

def get_gamma2(deflectx,deflecty,ratio=1,dunit=1):
    '''
    computes second component of gamma from deflection matrices
    '''
    dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
    return dDXdy

def get_gamma(deflectx,deflecty,ratio=1,dunit=1):
    '''
    computes gamma from deflection matrices
    '''
    dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
    dDYdy,dDYdx = np.gradient(deflecty*ratio/dunit)
    gamma1 = 0.5*(dDXdx-dDYdy)
    gamma2 = dDXdy
    return np.sqrt(gamma1**2+gamma2**2) 

def get_maps(deflectx,deflecty,ratio=1,dunit=1,return_all=False):
    '''
    computes kappa and gamma (and gamma1,gamma2) from deflection matrices
    '''
    dDXdy,dDXdx = np.gradient(deflectx*ratio/dunit)
    dDYdy,dDYdx = np.gradient(deflecty*ratio/dunit)
    kappa = 0.5*(dDXdx+dDYdy)
    gamma1 = 0.5*(dDXdx-dDYdy)
    gamma2 = dDXdy
    gamma = np.sqrt(gamma1**2+gamma2**2)
    if return_all:
        return kappa,gamma,gamma1,gamma2
    else:
        return kappa,gamma

def fermat_potential(ix,iy,deflectx,deflecty,psi,zl,zs,ratio=1,cosmo=FlatLambdaCDM(Om0=0.3,H0=70),dx=0,dy=0,ps=0.03*u.arcsec):
    '''
    computes the time delays in seconds in the image plane with respect to an image plane position
    '''

    sx0,sy0 = delens(ix,iy,deflectx,deflecty,ratio=ratio,dx=dx,dy=dy)
    thetaX,thetaY = np.meshgrid(np.arange(psi.shape[1])+1,np.arange(psi.shape[0])+1)
    sep = (np.sqrt((thetaX-sx0)**2+(thetaY-sy0)**2))*ps

    dls = cosmo.angular_diameter_distance_z1z2(zl,zs)
    ds = cosmo.angular_diameter_distance(zs)
    dl = cosmo.angular_diameter_distance(zl)

    fermat = ((1+zl)/const.c * (dl*ds)/dls * (0.5*sep**2 - psi*(u.arcsec)**2*ratio)).to('rad**2 s')

    return fermat