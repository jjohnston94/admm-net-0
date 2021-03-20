import time
import numpy as np
import scipy.io
import os

def CN(d1,d2,variance):

  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def gen_x_unscaled(Nsamp, p, s1, s2, grid, fp=None, adjacent=False):
  if Nsamp == 0:
    return 0,0
  start_time = time.time()
  A = p.A  
  A1 = A[:,:p.N_part]
  A2 = A[:,p.N_part:]
  m, n = A.shape

  if grid:
    X1 = np.zeros((p.N_part, Nsamp), dtype=complex)
  elif not grid:
    X1 = np.zeros((s1, Nsamp), dtype=complex)

  X2 = np.zeros((m, Nsamp), dtype=complex)

  Np = p.Np
  Nd = p.Nd
  NtxNrx = p.NtxNrx

  s1_unif = (type(s1) == list)
  if s1_unif:
    print("s1 random")
    s1_min = min(s1)
    s1_max = max(s1)

  s2_unif = (type(s2) == list)
  if s2_unif:
    print("s2 random")
    s2_min = min(s2)
    s2_max = max(s2)

  if adjacent:
    print("Adjacent samples")

  for i in range(Nsamp):

      # Experiment: Each sample has a different sparsity
      if s1_unif:
        s1 = np.random.permutation(s1_max - s1_min + 1)[0] + s1_min

      if s2_unif:
        s2 = np.random.permutation(s2_max - s2_min + 1)[0] + s2_min
      

      # ==================== X1 =============================
      if grid:
        if adjacent is not None:
          adj_dim = p.dims[adjacent]
          # ind0 = np.random.permutation(adj_dim-s1+1)[0]
          ind0 = 2
          temp = len(p.dims)*[s1*[0]]
          temp[adjacent] = [ind0+c for c in range(s1)]
          if adjacent == 0:
            temp[-1] = s1*[2]
          ind = np.ravel_multi_index(temp, p.dims) # Ngl,Ngk1,Ngk2,Ngm
          X1[ind,i] = CN(s1,1,1)[:,0]
          # import pdb; pdb.set_trace()

          # ind = np.random.permutation(p.N_part-1)[0]
          # X1[ind:ind+s1,i] = CN(s1,1,1)[:,0]
        else:
          X1[np.random.permutation(p.N_part)[:s1],i] = CN(s1,1,1)[:,0]
      elif not grid:
        X1[:,i] = CN(s1,1,1)[:,0]
      # ==================== X2 =============================
      temp = np.zeros(Np*Nd, dtype=complex)
      temp[np.random.permutation(Np*Nd)[:s2]] = CN(s2,1,1)[:,0]

      # old
      # temp = np.zeros((Np,Nd),dtype=complex)
      # for t in range(Nd):
      #   temp[np.random.permutation(Np)[:s2],t] = CN(s2,1,1)[:,0]
      
      # MIMO
      if p.scen == 'mimo_d':
        X2[:,i] = np.kron(np.ones(NtxNrx),temp)
      
      elif p.scen == 'siso_d':
        X2[:,i] = temp

      elif p.scen == 'sinusoids' or p.scen == 'gaussian':
        X2[np.random.permutation(m)[:s2],i] = CN(s2,1,1)[:,0]

  print('Avg ||x1||_0 =', '{}'.format(np.sum(X1!=0)/Nsamp))
  print('Avg ||x2||_0 =', '{}'.format(np.sum(X2!=0)/Nsamp))
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
    np.save(fp+'/X1',X1.T)
    np.save(fp+'/X2',X2.T)
    print('Saved data to:' + fp)
  # print('Generating data took ' + str(round(time.time() - start_time,1)) + ' seconds')
  
  return X1.T, X2.T

def scale_data_large(X1, X2, p, s1, s2, SNR, SIR, grid):

  X1 = X1.T
  X2 = X2.T
  Nsamp = X1.shape[1]

  Np = p.Np
  Nd = p.Nd
  NtxNrx = p.NtxNrx

  A = p.A
  A1 = A[:,:p.N_part] 
  A2 = A[:,p.N_part:]
  m, n = A.shape

  Y = np.zeros((m,Nsamp),dtype=complex)
  X = np.zeros((n,Nsamp),dtype=complex)
  
  #======================SNR/SIR Scaling ===========================
  # AWGN
  SNR_tally = 0
  SIR_tally = 0
  for i in range(Nsamp):
    norm_A1X1 = np.linalg.norm(np.matmul(A1,X1[:,i])) 

    noise = CN(m,1,1)
    if type(SNR) == list:
      SNR_temp = (max(SNR) - min(SNR))*np.random.rand(1)[0] + min(SNR)
      noise = 10**(-SNR_temp/20) * norm_A1X1 * noise/np.linalg.norm(noise)
    else: 
      noise = 10**(-SNR/20) * norm_A1X1 * noise/np.linalg.norm(noise)
    SNR_tally = SNR_tally + 20*np.log10(norm_A1X1/np.linalg.norm(noise))

    if type(SIR) == list:
      SIR_temp = (max(SIR) - min(SIR))*np.random.rand(1)[0] + min(SIR)
      X2[:,i] = 10**(-SIR_temp/20) * norm_A1X1 * X2[:,i]/np.linalg.norm(X2[:,i])
    else:
      X2[:,i] = 10**(-SIR/20) * norm_A1X1 * X2[:,i]/np.linalg.norm(X2[:,i])
    SIR_tally = SIR_tally + 20*np.log10(norm_A1X1/np.linalg.norm(X2[:,i]))

    if grid:
      X[:,i] = np.concatenate((X1[:,i],X2[:,i]),0)
      Y[:,i] = np.matmul(A,X[:,i]) + noise[:,0]
    elif not grid:
      Y[:,i] = np.matmul(A1,X1[:,i]) + np.matmul(A2,X2[:,i]) + noise 
       
  SNR_emp = SNR_tally/Nsamp
  SIR_emp = SIR_tally/Nsamp

  
  print('Average SIR =',round(SIR_emp,2))
  print('Average SNR =', round(SNR_emp,2))
  

  if grid:
    X = np.concatenate((X.real,X.imag),axis=0)
    Y = np.concatenate((Y.real,Y.imag),axis=0)
    return X.T,Y.T
  elif not grid:
    Y = np.concatenate((Y.real,Y.imag),axis=0)
    return X1.T, Y.T, np.array(coordvals)

def gen_data(Nsamp, p, s1, s2, SNR, SIR, grid, *args, savefilepath=None, adjacent=None):
  start_time = time.time()
  X1,X2 = gen_x_unscaled(Nsamp, p, s1, s2, grid, fp=None, adjacent=adjacent)

  res = scale_data_large(X1, X2, p, s1, s2, SNR, SIR, grid)
  print('Generated data in ' + str(round(time.time() - start_time,1)) + ' seconds')
    
  return res
