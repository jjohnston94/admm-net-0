import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import data
import problem
import admm_net
import admm_net_tied
import numpy as np
import matplotlib.pyplot as plt
import time
from pdb import set_trace
import train
import BlackBoxNet as bbnet

import tensorflow as tf

def load(net, weights_path):
  obj = net.load_weights(weights_path)
  obj.expect_partial()
  return net

if __name__ == '__main__':
  # scen= 'sinusoids'
  scen = 'gaussian'
  dims = (20, 40)
  m,n = dims
  p = problem.Problem(dims, scen)
  p.A = np.load('./A.npy')
  # set_trace()
  # number of nonzero entries in each subvector
  s1 = 2
  s2 = None

  # signal-to-noise ratio
  SNR = 10
  # signal-to-interference ratio
  SIR = 0

  # If tied = True, then parameters are shared across layers
  # If tied = False, then each layer has its own parameters
  tied = False

  # number of stages
  # num_stages = 3
  for num_stages in [1,2,3,4,5,6]:
    dims_list = [(n,m),(m,n),(n,m)]

    Ntrain = 10**5
    Ntest = 10**3
    data_test = data.gen_data(Ntest, p, s1, s2, SNR, SIR, True, 'large')
    data_train = data.gen_data(Ntrain, p, s1, s2, SNR, SIR, True, 'large')

    # Path to folder where to save network. If None, then network not saved
    savepath = '/Users/jeremyjohnston/Documents/admm-net-0/nets/' + 'untied_' + str(num_stages) + 'layer'
    # time.strftime("%m_%d_%Y_%H_%M_%S",time.localtime()) 

    learning_rate = 10*[1e-3] + 10*[1e-4]
    # learning_rate = 50*[1e-3] + 50*[1e-4]
    batch_size = 100
    
    # Create new, initialized network
    net = admm_net.ADMMNet(p, num_stages, tied)
    # net = admm_net_tied.ADMMNet(p, num_stages)
    # net = bbnet.BBNet(p, dims_list)

    # Load previously saved network
    # net(data_test[1]) # Must call network before loading weights
    # net = load(net, 'savepath/weights')

    # Train net
    net = train.train_net(net, p, data_train, data_test, learning_rate, batch_size, savepath=savepath)