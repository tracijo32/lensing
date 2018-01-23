import numpy as np

def param2array(param):
    if isinstance(param,np.ndarray):
        return param,'array'
    elif isinstance(param,list):
        return np.asarray(param)
    elif isinstance(param,float):
        return np.array([param]),'scalar'
    elif isinstance(param,np.float64):
        return np.array([param]),'scalar'
    elif isinstance(param,int):
        return np.array(param).astype(float),'scalar'
    elif isinstance(param,tuple):
        return np.asarray(param), 'tuple'
    else:
        return

def array2param(array,ptype):
    if ptype == 'list':
        return np.tolist(array)
    if ptype == 'scalar':
        return array[0]
