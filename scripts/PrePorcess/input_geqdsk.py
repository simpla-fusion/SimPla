'''
Created on 2013-6-9

@author: salmon
'''

class GEQDSK(object):
    '''
    G-EQDSK File
    '''
    def read_from(self,f):
        fid=open(f)
        lines= fid.readlines()
        self.idnum=int(str[-3])
        self.nw=int(str[-2])
        self.nh=int(str[-1])
        strs=fid.readline.split()
        return;
    def write_to(self,f):
        return;

    def __init__(selfparams):
        '''
        Constructor
        '''
        return;