import time
import numpy as np
import scipy.io
import problem
import admm2 as admm

def eval_dir(num_stage, snr, sir, s1, s2, tied_str, dirname, input_scen, *args, show_plots=False, grid=True, datadict=None):
  print("Evaluate nets in directory", dirname)
  names = os.listdir(dirname)
  if '.DS_Store' in names:
    names.remove('.DS_Store')
  # names = [int(n) for n in names]
  # names = np.sort(names)
  pairs = []
  flag = False

  Ntest = int(1e3)


  for n in names:
    name = dirname + '/' + n
    # print(name)
    net,p = load_net(name, input_scen, num_stage, tied_str)
    if flag == False:
      flag == True
      if datadict is None:
        data_test = gen_data(Ntest, p, s1, s2, snr, sir, grid)
      else:
        data_test = datadict['X'], datadict['Y']
    print(name)
    err = eval_net(net, data_test, show_plots, p, grid, *args)
    pairs.append((name,err))

  best_err = 0
  best_name = None
  for k,v in pairs:
    print(k)
    print(v)
    if v < best_err:
      best_name = k
      best_err = v

  print('Best Name:', best_name)
  print('Best Error:', best_err, 'dB')
  return best_name, best_err

# def nmse(x,xhat, db=True):
#   if db:
#     return 10*np.log10(np.mean(np.linalg.norm((x-xhat),axis=0)**2/np.linalg.norm(x,axis=0)**2))
#   else:
#     return np.mean(np.linalg.norm((x-xhat),axis=1)**2/np.linalg.norm(x,axis=1)**2)

def eval_net(net, data_test, show_plots, p=None, grid=True, *args):
  if grid:
    x_test,y_test = data_test
    N = y_test.shape[1]
    M = x_test.shape[1]
    Ng = M - N

    # xhat = net.predict_on_batch(y_test)
    xhat = net(np.float32(y_test))

    xhat = xhat.numpy()
    xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
    x_test = x_test[:,0:M//2] + 1j*x_test[:,M//2:]

    x1_hat_c = xhat[:,0:Ng//2]
    x1_test_c = x_test[:,0:Ng//2]
    
    err = 10*np.log10(np.mean(np.linalg.norm((x1_test_c-x1_hat_c),axis=1)**2/np.linalg.norm(x1_test_c,axis=1)**2))
    # err2 = 10*np.log10(np.mean(np.linalg.norm((np.abs(x1_test_c)-np.abs(x1_hat_c)),axis=1)**2/np.linalg.norm(np.abs(x1_test_c),axis=1)**2))
    
    # err = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))

    if 'quiet' in args:
      pass
    else:
      print('Error:', err, 'dB')


    if show_plots == True:
      count = 0
      ind = []
      for i in range(10000):
          temp = 10*np.log10(np.mean(np.linalg.norm((x1_test_c[i]-x1_hat_c[i]))**2/np.linalg.norm(x1_test_c[i])**2))
        # if temp > 1.05*err and temp < 0.95*err:
          # x1hat_plot = np.abs(np.reshape(x1_hat_c[i],(4,10)))
          # x1test_plot = np.abs(np.reshape(x1_test_c[i],(4,10)))
          # plt.figure()
          # plt.matshow(x1hat_plot)
          # plt.figure()
          # plt.matshow(x1test_plot)
          # plt.show()

          plt.figure()
          # plt.plot(x1_hat_r[i], 'bx', label='xhat')
          # plt.plot(x1_test_r[i], 'r+', label='x_test')
          

          if p is not None:
            A = p.A
            A = A[:,:Ng//2]
            y = y_test[i]
            y = y[:N//2]+1j*y[N//2:]
            y_mf = np.matmul(A.T.conj(),y)
            y_mf = np.abs(y_mf)
            # y_mf = np.concatenate((y_mf.real,y_mf.imag))
            plt.plot(y_mf, 'k--', linewidth=1, label='x_mf')

          # plt.legend()


          # plt.figure()
          plt.plot(np.abs(x1_hat_c[i]), 'bx', label='xhat')
          plt.plot(np.abs(x1_test_c[i]), 'r+', label='x_test')
          plt.title(str(temp))
          # plt.plot(np.abs(x_test[i]), 'r+', label='x_test')
          

          # print(np.round(np.abs(x1_hat_c[i]),3))
          # print(np.round(np.abs(x1_test_c[i]),3))
          ind.append(i)
          count = count + 1
          if count == 10:
            plt.show()
            X = x1_hat_c[ind]
            # L = ['x'+str(k) for k in list(range(10))]
            # d = dict(zip(L,x1_hat_c[ind]))
            # import pdb; pdb.set_trace()
            scipy.io.savemat('./savetest.mat', {'X_py':X})
            # plt.legend()
            break

      
      # for i in range(10):
      #   plt.figure()
      #   plt.plot(x1_hat_r[i], 'bo', label='xhat')
      #   plt.plot(x1_test_r[i], 'ro', label='x_test')
      #   plt.legend()
      #   plt.show()

    return round(err,2)

  elif not grid:
    Ng = p.N_part

    x1_test_c, y_test, x1_true_nz_grid_vals = data_test
    Nsamp = x1_test_c.shape[0]
    s1 = x1_test_c.shape[1]


    xhat = net(np.float32(y_test))
    xhat = xhat.numpy()

    M = xhat.shape[1]
    xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
    # for i in range(20):
    #   plt.plot(np.abs(xhat[i]))
    #   plt.show()

    return eval_offgrid(xhat, x1_true_nz_grid_vals, p)

    # old
    if 0:
      x1_hat_c = xhat[:,0:Ng].T

      x1_hat_max = np.max(np.abs(x1_hat_c),axis=0)
      x1_thresholded = x1_hat_c * (np.abs(x1_hat_c) > 0.3*x1_hat_max)
      
      x1_hat_nz_locs = np.array(np.where(np.abs(x1_thresholded)>0)).T
      
      # temp = x1_hat_c
      # top_inds = []
      # for s in range(s1):
      #   m = np.max(temp,axis=0)
      #   wh = np.array(np.where(temp==m)).T
      #   for i in range(wh.shape[0]):
      #     top_inds.append(wh[i])
      #   temp[wh]=0
      # x1_hat_top_locs = np.array(top_inds)
      # x1_hat_nz_grid_vals = p.grid_vals[x1_hat_nz_locs[:,0]]
      x1_hat_nz_grid_vals = p.grid_vals[x1_hat_nz_locs[:,0]]

      num_nz_x1_hat = x1_hat_nz_locs.shape[0]
      x1_hat_AE_tau = np.zeros(num_nz_x1_hat)
      x1_hat_AE_v = np.zeros(num_nz_x1_hat)
      x1_hat_AE_theta1 = np.zeros(num_nz_x1_hat)
      x1_hat_AE_theta2 = np.zeros(num_nz_x1_hat)

      
      
      # x1_true_nz_grid_vals = p.grid_vals[x1_true_nz_locs[:,1]]
      # for i in range(num_nz_x1_hat):
      #   x1_hat_nz_grid_vals[i] = p.grid_vals[x1_hat_nz_locs[i,1]]
      #   x1_true_nz_grid_vals[i] = p.grid_vals[x1_true_nz_locs[i,1]]

      # ind = np.array(np.where(x1_hat_nz_loc[:,1] == 0))[0]
      # g = p.grid_vals[ind]
      # pdb.set_trace()
      
      for i in range(num_nz_x1_hat):
        nsamp = x1_hat_nz_locs[i,1]
        x1_hat_AE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,0] - x1_true_nz_grid_vals[nsamp,0,:]))
        x1_hat_AE_theta1[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,1] - x1_true_nz_grid_vals[nsamp,1,:]))
        x1_hat_AE_theta2[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,2] - x1_true_nz_grid_vals[nsamp,2,:]))
        x1_hat_AE_v[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,3] - x1_true_nz_grid_vals[nsamp,3,:]))

        # x1_hat_MAE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,0] - x1_true_nz_grid_vals[:,0]))
        # x1_hat_MAE_theta1[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,1] - x1_true_nz_grid_vals[:,1]))
        # x1_hat_MAE_theta2[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,2] - x1_true_nz_grid_vals[:,2]))
        # x1_hat_MAE_v[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,3] - x1_true_nz_grid_vals[:,3]))

      
      tau_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,0,:]))*1e6
      th1_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,1,:]))
      th2_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,2,:]))
      v_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,3,:]))

      x1_hat_MAE_tau = np.mean(x1_hat_AE_tau)*1e6
      
      
      x1_hat_MAE_theta1 = np.mean(x1_hat_AE_theta1)
      x1_hat_MAE_theta2 = np.mean(x1_hat_AE_theta2)
      x1_hat_MAE_v = np.mean(x1_hat_AE_v)

      # print(tau_avg)
      # print(x1_hat_MAE_tau)
      print(x1_hat_MAE_tau/tau_avg)

      # print(th1_avg)
      # print(x1_hat_MAE_theta1)
      print(x1_hat_MAE_theta1/th1_avg)
      
      # print(th2_avg)
      # print(x1_hat_MAE_theta2)
      print(x1_hat_MAE_theta2/th2_avg)
      
      # print(v_avg)
      # print(x1_hat_MAE_v)
      print(x1_hat_MAE_v/v_avg)



      

      # plt.figure()
      # for i in range(Nsamp):
      #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
      #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,0]*1e6, c='b')
      #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,0]*1e6, c='g')

      # plt.figure()
      # for i in range(Nsamp):
      #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
      #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,1], c='b')
      #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,1], c='g')

      # plt.figure()
      # for i in range(Nsamp):
      #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
      #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,2], c='b')
      #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,2], c='g')

      # plt.figure()
      # for i in range(Nsamp):
      #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
      #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,3], c='b')
      #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,3], c='g')

      plt.show()

      return x1_hat_MAE_tau, x1_hat_MAE_theta1, x1_hat_MAE_theta2, x1_hat_MAE_v
    # np.sum(np.abs(x1_hat_nz_locs - x1_true_nz_locs))
    # print('x1_max=' + str(x1_max))
    # for l in range(100):
    #   plt.plot(np.abs(x1_hat_c[l]))
    #   plt.figure()
    #   plt.plot(np.abs(x1_thresholded[l]))
    #   plt.show()

def eval_single_net(num_stage, snr, sir, s1,s2, tied_str, input_name, input_scen, show_plots=False, grid=True, datadict=None, adjacent=None):
  
  if type(input_name) == str:
    print(input_name)
    net,p = load_net(input_name, input_scen, num_stage, tied_str)
    
    if datadict is None:
      Ntest = int(1e3)
      print('Generating', Ntest, 'test samples...')
      data_test = gen_data(Ntest, p, s1, s2, snr, sir, grid, 'large', adjacent=adjacent)
    else:
      data_test = datadict['X'], datadict['Y']
    errs = eval_net(net, data_test, show_plots=show_plots, p=p, grid=grid)
    print(errs)
    return errs

  elif type(input_name) == list:
    I = []
    for name in input_name: 
      net,p = load_net(name, input_scen, num_stage, tied_str)
      I.append((net,name))
    if datadict is None:
      Ntest = int(1e3)
      print('Generating', Ntest, 'test samples...')
      data_test = gen_data(Ntest, p, s1, s2, snr, sir, grid, 'large', adjacent=adjacent)
    else:
      data_test = datadict['X'], datadict['Y']
    errs = [eval_net(net, data_test, show_plots=show_plots, p=p, grid=grid) for net,name in I]
    for name,err in zip(input_name,errs):
      print(name)
      print(err)

    return zip(input_name,errs)

def error_vs_num_stages(num_stage, snr, sir, s1,s2, tied_str, input_name, input_scen, show_plots=False, grid=True, datadict=None):
  errs = []
  I = []

  net,p = load_net(input_name, input_scen, num_stage, tied_str)

  Ntest = int(1e3)
  # print('Generating', Ntest, 'test samples...')
  if datadict is None:
    data_test = gen_data(Ntest, p, s1, s2, snr, sir, grid)
  else:
    data_test = datadict['X'], datadict['Y']
  
  errs = []
  
  for i in range(num_stage):
    errs.append(eval_net(net, data_test, show_plots=False, p=p))
    print(len(net.Layers))
    del net.Layers[0]

  print(errs)

  return errs

def eval_offgrid(xhat, x1_true_nz_grid_vals, p):
  """xhat is complex-valued"""
  Ng = p.N_part
  # pdb.set_trace()
  Nsamp = xhat.shape[0]
  s1 = x1_true_nz_grid_vals.shape[1]
  M = xhat.shape[1]
  # xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
  x1_hat_c = xhat[:,0:Ng].T

  x1_hat_max = np.max(np.abs(x1_hat_c),axis=0)
  x1_thresholded = x1_hat_c * (np.abs(x1_hat_c) > 0.3*x1_hat_max)
  
  x1_hat_nz_locs = np.array(np.where(np.abs(x1_thresholded)>0)).T
  
  # temp = x1_hat_c
  # top_inds = []
  # for s in range(s1):
  #   m = np.max(temp,axis=0)
  #   wh = np.array(np.where(temp==m)).T
  #   for i in range(wh.shape[0]):
  #     top_inds.append(wh[i])
  #   temp[wh]=0
  # x1_hat_top_locs = np.array(top_inds)
  # x1_hat_nz_grid_vals = p.grid_vals[x1_hat_nz_locs[:,0]]
  x1_hat_nz_grid_vals = p.grid_vals[x1_hat_nz_locs[:,0]]

  num_nz_x1_hat = x1_hat_nz_locs.shape[0]
  x1_hat_AE_tau = np.zeros(num_nz_x1_hat)
  x1_hat_AE_v = np.zeros(num_nz_x1_hat)
  x1_hat_AE_theta1 = np.zeros(num_nz_x1_hat)
  x1_hat_AE_theta2 = np.zeros(num_nz_x1_hat)

  
  
  # x1_true_nz_grid_vals = p.grid_vals[x1_true_nz_locs[:,1]]
  # for i in range(num_nz_x1_hat):
  #   x1_hat_nz_grid_vals[i] = p.grid_vals[x1_hat_nz_locs[i,1]]
  #   x1_true_nz_grid_vals[i] = p.grid_vals[x1_true_nz_locs[i,1]]

  # ind = np.array(np.where(x1_hat_nz_loc[:,1] == 0))[0]
  # g = p.grid_vals[ind]
  
  if p.scen == 'mimo_d':
    for i in range(num_nz_x1_hat):
      nsamp = x1_hat_nz_locs[i,1]
      x1_hat_AE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,0] - x1_true_nz_grid_vals[nsamp,0,:]))
      x1_hat_AE_theta1[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,1] - x1_true_nz_grid_vals[nsamp,1,:]))
      x1_hat_AE_theta2[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,2] - x1_true_nz_grid_vals[nsamp,2,:]))
      x1_hat_AE_v[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,3] - x1_true_nz_grid_vals[nsamp,3,:]))

      # x1_hat_MAE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,0] - x1_true_nz_grid_vals[:,0]))
      # x1_hat_MAE_theta1[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,1] - x1_true_nz_grid_vals[:,1]))
      # x1_hat_MAE_theta2[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,2] - x1_true_nz_grid_vals[:,2]))
      # x1_hat_MAE_v[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,3] - x1_true_nz_grid_vals[:,3]))

    
    tau_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,0,:]))*1e6
    th1_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,1,:]))
    th2_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,2,:]))
    v_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,3,:]))

    x1_hat_MAE_tau = np.mean(x1_hat_AE_tau)*1e6
    
    
    x1_hat_MAE_theta1 = np.mean(x1_hat_AE_theta1)
    x1_hat_MAE_theta2 = np.mean(x1_hat_AE_theta2)
    x1_hat_MAE_v = np.mean(x1_hat_AE_v)

    # print(tau_avg)
    # print(x1_hat_MAE_tau)
    print(x1_hat_MAE_tau/tau_avg)

    # print(th1_avg)
    # print(x1_hat_MAE_theta1)
    print(x1_hat_MAE_theta1/th1_avg)
    
    # print(th2_avg)
    # print(x1_hat_MAE_theta2)
    print(x1_hat_MAE_theta2/th2_avg)
    
    # print(v_avg)
    # print(x1_hat_MAE_v)
    print(x1_hat_MAE_v/v_avg)

    # plt.figure()
    # for i in range(Nsamp):
    #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
    #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,0]*1e6, c='b')
    #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,0]*1e6, c='g')

    # plt.figure()
    # for i in range(Nsamp):
    #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
    #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,1], c='b')
    #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,1], c='g')

    # plt.figure()
    # for i in range(Nsamp):
    #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
    #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,2], c='b')
    #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,2], c='g')

    # plt.figure()
    # for i in range(Nsamp):
    #   xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
    #   plt.scatter(s1*[i],x1_true_nz_grid_vals[i,3], c='b')
    #   plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs,3], c='g')

    plt.show()

    return x1_hat_MAE_tau, x1_hat_MAE_theta1, x1_hat_MAE_theta2, x1_hat_MAE_v
  elif p.scen == 'siso':
    for i in range(num_nz_x1_hat):
      nsamp = x1_hat_nz_locs[i,1]
      # pdb.set_trace()
      x1_hat_AE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i] - x1_true_nz_grid_vals[nsamp,:]))
      # x1_hat_MAE_tau[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,0] - x1_true_nz_grid_vals[:,0]))
      # x1_hat_MAE_theta1[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,1] - x1_true_nz_grid_vals[:,1]))
      # x1_hat_MAE_theta2[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,2] - x1_true_nz_grid_vals[:,2]))
      # x1_hat_MAE_v[i] = np.min(np.abs(x1_hat_nz_grid_vals[i,3] - x1_true_nz_grid_vals[:,3]))

    
    tau_avg = np.mean(np.abs(x1_true_nz_grid_vals[:,:]))

    x1_hat_MAE_tau = np.mean(x1_hat_AE_tau)
    

    # print(tau_avg)
    # print(x1_hat_MAE_tau)
    print(x1_hat_MAE_tau/tau_avg)

    plt.figure()
    for i in range(50):
      xhat_i_locs = np.array(np.where(x1_hat_nz_locs[:,1]==i))[0]
      plt.scatter(s1*[i],x1_true_nz_grid_vals[i], c='b')
      plt.scatter(xhat_i_locs.size*[i], x1_hat_nz_grid_vals[xhat_i_locs], c='g')

    # plt.show()

    return x1_hat_MAE_tau

def eval_nets(nets, data_test, L, params_initialization, names, quiet):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params_initialization
  
  errs = np.zeros((L1,L2,L3,L4))

  x_test,y_test = data_test
  N = y_test.shape[1]
  M = x_test.shape[1]
  Ng = M - N

  
  x_test = x_test[:,0:M//2] + 1j*x_test[:,M//2:]
  x1_test_c = x_test[:,0:Ng//2]
  x1_test_r = np.concatenate((x1_test_c.real,x1_test_c.imag),axis=1)

  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          xhat = nets[l1,l2,l3,l4].predict_on_batch(y_test)
          xhat = xhat.numpy()
          xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
          x1_hat_c = xhat[:,0:Ng//2]
          x1_hat_r = np.concatenate((x1_hat_c.real,x1_hat_c.imag),axis=1)
          errs[l1,l2,l3,l4] = 10*np.log10(np.mean(np.sum((x1_test_r-x1_hat_r)**2,axis=1)/np.sum(x1_test_r**2,axis=1)))
          # errs[l1,l2,l3,l4] = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))
          # names[l1,l2,l3,l4] = tn.save_net(a,props)
  
  print_best(errs, 
             names, 
             L, 
             params_initialization,
             quiet
             )
  return errs
      
def eval_net_no_partition(net, data_test, name, show_plots):
  x_test,y_test = data_test
  N = y_test.shape[1]
  M = x_test.shape[1]
  Ng = M - N

  xhat = net.predict_on_batch(y_test)
  xhat = xhat.numpy()

  err = 10*np.log10(np.mean(np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1)))

  print('Name:', name)
  print('Error:', err, 'dB')
  if show_plots == True:
    for i in range(10):
      plt.figure()
      plt.plot(xhat[i], 'bo', label='xhat')
      plt.plot(x_test[i], 'ro', label='x_test')
      plt.legend()
      plt.show()

  return err

def load_net(folder, scen, num_layers, tied, *args, params=None):
  # args = ln.get_args()
  # scen = args[0].scen
  # folder = './nets/' + args[0].folder

  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2-n1), scen, partition=True, N_part=n2-n1)

  if params is None:
    a = admm.ADMMNet(p, num_layers, tied, *args)
  else:
    a = admm.ADMMNet(p, num_layers, tied, *args, params_init=params)

  obj = a.load_weights(folder + '/weights')
  obj.expect_partial()
  return a, p
