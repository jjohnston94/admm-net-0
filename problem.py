import numpy as np


class Problem:
  def __init__(self, dims, scen, **kwargs):
    M, N = dims

    self.N_part = N

    self.partition = True
    
    self.scen = scen

    
    if scen == 'sinusoids':

      self.Np = M
      self.Nd = 1
      self.NtxNrx = 0

      m = np.arange(M)[:,None]
      n = np.arange(N)

      self.A = np.exp(1j*2*np.pi*m*n/N)/np.sqrt(M)
    
    # if scen == 'gaussian':
    #   self.Np = M
    #   self.Nd = 1
    #   self.NtxNrx = 0

    #   self.A = np.random.normal(size=(M,N)) + 1j*np.random.normal(size=(M,N))
    #   # normalize cols
    #   self.A = np.matmul(self.A, np.diag(1/np.sqrt(np.sum(np.abs(self.A)**2,axis=0))))
      
      
    # concatenate A with identity
    self.A = np.concatenate((self.A, np.eye(np.shape(self.A)[0])),axis=1)

    return

  def size(self, dim=None):
    if dim is None:
      return self.A.shape
    else:
      return self.A.shape[dim]
