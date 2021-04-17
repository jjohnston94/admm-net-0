import time
import numpy as np
import scipy.io
import os
from pdb import set_trace

def CN(d1,d2,variance):

  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def gen_x_unscaled(Nsamp, p, s1, s2, grid, fp=None, adjacent=False):
  start_time = time.time()
  A = p.A  
  m, n = A.shape

  X = np.zeros((n, Nsamp), dtype=complex)

  # Np = p.Np
  # Nd = p.Nd
  # NtxNrx = p.NtxNrx

  s1_unif = (type(s1) == list)
  if s1_unif:
    print("s1 random")
    s1_min = min(s1)
    s1_max = max(s1)

  if adjacent:
    print("Adjacent samples")

  for i in range(Nsamp):

      # Experiment: Each sample has a different sparsity
      if s1_unif:
        s1 = np.random.permutation(s1_max - s1_min + 1)[0] + s1_min

      X[np.random.permutation(n)[:s1],i] = CN(s1,1,1)[:,0]


  print('Avg ||x||_0 =', '{}'.format(np.sum(X!=0)/Nsamp))
  # E_n = np.mean(np.sum(np.abs(nn)**2,axis=0))
  # snr_data = 10*np.log10(np.divide(E_A1x1,E_n))
  # sir_data = 10*np.log10(np.divide(E_A1x1/m,E_A2x2/(4*s2))) #MIMO
  # # sir_data = 10*np.log10(np.divide(E_A1x1/m,E_A2x2/s2)) #SISO

  # print("SNR of generated data:", snr_data)
  # print("SIR of generated data:", sir_data)
  # print("sig:", sig)
  # print("sig2:", sig2)

  if fp is not None:
    # try:
    # os.mkdir(fp)
    # except:
    #   FileExistsError
    np.save(fp+'/X',X.T)
    print('Saved data to:' + fp)
  # print('Generating data took ' + str(round(time.time() - start_time,1)) + ' seconds')
  
  return X.T

def scale_data_large(X, p, s1, s2, SNR, SIR, grid):

  X = X.T
  Nsamp = X.shape[1]

  # Np = p.Np
  # Nd = p.Nd
  # NtxNrx = p.NtxNrx

  A = p.A
  m, n = A.shape

  Y = np.zeros((m,Nsamp),dtype=complex)
  # X = np.zeros((n,Nsamp),dtype=complex)
  
  #======================SNR/SIR Scaling ===========================
  # AWGN
  SNR_tally = 0
  SIR_tally = 0
  for i in range(Nsamp):
    norm_AX = np.linalg.norm(np.matmul(A,X[:,i])) 

    noise = CN(m,1,1)
    if type(SNR) == list:
      SNR_temp = (max(SNR) - min(SNR))*np.random.rand(1)[0] + min(SNR)
      noise = 10**(-SNR_temp/20) * norm_AX * noise/np.linalg.norm(noise)
    else: 
      noise = 10**(-SNR/20) * norm_AX * noise/np.linalg.norm(noise)
    SNR_tally = SNR_tally + 20*np.log10(norm_AX/np.linalg.norm(noise))

    # X[:,i] = np.concatenate((X[:,i],X2[:,i]),0)
    Y[:,i] = np.matmul(A,X[:,i]) + noise[:,0]
       
  SNR_emp = SNR_tally/Nsamp
  SIR_emp = SIR_tally/Nsamp

  
  print('Average SIR =',round(SIR_emp,2))
  print('Average SNR =', round(SNR_emp,2))
  
  # set_trace()
  X = np.concatenate((X.real,X.imag),axis=0)
  Y = np.concatenate((Y.real,Y.imag),axis=0)
  return X.T,Y.T.astype(np.float64)

def gen_data(Nsamp, p, s1, s2, SNR, SIR, grid, *args, savefilepath=None, adjacent=None):
  start_time = time.time()
  X1 = gen_x_unscaled(Nsamp, p, s1, s2, grid, fp=None, adjacent=adjacent)

  res = scale_data_large(X1, p, s1, s2, SNR, SIR, grid)
  print('Generated data in ' + str(round(time.time() - start_time,1)) + ' seconds')
      
  return res
