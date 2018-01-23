import numpy as np

class ArcList:
    def __init__(self,arcfile):
        self.arclist = []
        id,ra,dec,z = np.loadtxt(arcfile,usecols=(0,1,2,6),unpack=True,dtype='S8,f8,f8,f8')        

        id_s = np.chararray(len(id))
        im = np.chararray(len(id))
        for i in range(len(id)):
            id_split = id.rsplit('.')
            id_s[i] = id_split[0]
            im[i] = id_split[1]

        self.sources = np.unique(id_s)
        
        for i in range(len(self.sources)):
            ind = id_s == self.sources[i]
            self.arclist.append(Arc(id[ind],ra[ind],dec[ind],z[ind]))

        self.nimages = len(id)
        self.nsources = len(self.sources)


class Arc:
    def __init__(self,id,ra,dec,z):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.z = z[0]
        self.multiplicity = len(id)
