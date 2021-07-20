import numpy as np
import scipy as sp
import scipy.sparse
'''assuming only one Mat is saved in sparse format, then this will read it'''

def read_SPIMat(filename):
    '''Read in a SPIMat saved binary file and output a scipy sparse csr_matrix'''
    with open(filename,'rb') as fid:
        header = np.fromfile(fid,dtype=np.dtype('>i4'),count=1)[0]
        M,N,nz = np.fromfile(fid,dtype=np.dtype('>i4'),count=3)
        I = np.empty(M+1,dtype=np.dtype('>i4'))
        I[0] = 0
        rownz = np.fromfile(fid,dtype=np.dtype('>i4'),count=M)
        np.cumsum(rownz,out=I[1:])
        assert I[-1] == nz
        J = np.fromfile(fid,dtype=np.dtype('>i4'),count=nz)
        assert len(J) == nz
        V = np.fromfile(fid,dtype=np.dtype('>c16'),count=nz)
        assert len(V) == nz
    #print(sp.sparse.csr_matrix((V,J,I),shape=(M,N)).todense()) # if you want the dense format
    return sp.sparse.csr_matrix((V,J,I),shape=(M,N))
if __name__=="__main__":
    fname = 'A.dat'
    data = read_SPIMat(fname)
    print('data = ',data)
    print('or in dense format')
    print('data = ',data.todense())
