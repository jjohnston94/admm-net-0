import data
import problem
import admm_net
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import train

def load(net, weights_path):
  obj = net.load_weights(weights_path)
  obj.expect_partial()
  return net

if __name__ == '__main__':
  scen= 'sinusoids'
  dims = (64, 150)
  p = problem.Problem(dims, scen, N_part=dims[1])

  # number of nonzero entries in each subvector
  s1 = 2
  s2 = 4

  # signal-to-noise ratio
  SNR = 10
  # signal-to-interference ratio
  SIR = 0

  # If tied = True, then parameters are shared across layers
  # If tied = False, then each layer has its own parameters
  tied = False

  # number of stages
  num_stages = 5

  Ntrain = 10**5
  Ntest = 10**3
  data_test = data.gen_data(Ntest, p, s1, s2, SNR, SIR, True, 'large')
  data_train = data.gen_data(Ntrain, p, s1, s2, SNR, SIR, True, 'large')

  # Path to folder where to save network. If None, then network not saved
  savepath = None

  learning_rate = 5*[1e-3] + 5*[1e-4]
  batch_size = 100
  
  # Create new, initialized network
  net = admm_net.ADMMNet(p, num_stages, tied)

  # Load previously saved network
  # net(data_test[1]) # Must call network before loading weights
  # net = load(net, 'savepath/weights')

  # Train net
  net = train.train_net(net, p, data_train, data_test, learning_rate, batch_size, savepath=savepath)