#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:40:31 2020

@author: jeremyjohnston
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import numpy.linalg as la
from pdb import set_trace

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

class ADMMNet(tf.keras.Model):
  
  def __init__(self, p, num_stages, tied, *args, **kwargs):
    super().__init__()

    self.loss_fcn = tf.keras.losses.MeanSquaredError()

    self.params_init = {'lambda':0.1,'lambda2':0.1 , 'alpha':1. , 'rho':1.}
    
    if 'params_init' in kwargs.keys():
      for k,v in kwargs['params_init'].items():
        self.params_init[k] = v
    
    self.tied = tied
    self.n1 = p.A.shape[1]
    self.Layers=[]
    for i in range(num_stages-1):
      self.Layers.append(Stage(self.params_init, p, self.tied, *args))
    self.Layers.append(StageFinal(self.params_init, p, self.tied, *args))
    
    if p.partition == True:
      print('Scenario:','{0}x{1}'.format(p.size(0),p.size(1)),p.scen,'partition')
    else:
      print('Scenario:','{0}x{1}'.format(p.size(0),p.size(1)),p.scen)

    if 'quiet' not in args:
      if self.tied:
        print('TIED ADMM-Net with {0} stages and initial parameters:'.format(len(self.Layers)))
      elif not self.tied:
        print('UNTIED ADMM-Net with {0} stages'.format(len(self.Layers)))

      # print("Initial parameters:")
      # for k,v in self.params_init.items():
      #   print(k,'=',v)

  @tf.function
  def train_step(self, x, y_true):
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.loss_fcn(y_pred, y_true)

    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

  def call(self, inputs):
    # layer 1
    z_0 = np.zeros((2*self.n1,1),dtype=np.float32)
    u_0 = np.zeros((2*self.n1,1),dtype=np.float32)
    z,u = self.Layers[0](inputs, z_0, u_0)

    # layers 2, 3, ... 
    for l in self.Layers[1:]:
      z,u = l(inputs,z,u)

    return tf.transpose(z)

class Stage(layers.Layer):

  def __init__(self, params_init, p, tied, *args):
    super().__init__()
    m = p.size(0)
    n = p.size(1)
    self.m = m
    self.n = n
    self.p = p
    self.rho0 = params_init['rho']
    self.alpha0 = params_init['alpha']
    self.lambda0 = params_init['lambda']
    self.lambda0_2 = params_init['lambda2']

    print('Tied stage' if tied else 'Untied stage')

    if 'random_init' in args:
      M1_init = np.random.normal(size=(2*n,2*m))
      M2_init = np.random.normal(size=(2*n,2*n))
      # normalize cols
      M1_init = np.matmul(M1_init,np.diag(1/np.sqrt(np.sum(M1_init**2,axis=0))))
      M2_init = np.matmul(M2_init,np.diag(1/np.sqrt(np.sum(M2_init**2,axis=0))))
    else:
      AULA = self.AULA(p)
      M1 = np.matmul(np.eye(n)/self.rho0 - (1/self.rho0**2)*AULA, p.A.T.conj())
      M2 = np.eye(n) - (1/self.rho0)*AULA

      top = np.concatenate((M1.real, -M1.imag),axis=1)
      bot = np.concatenate((M1.imag, M1.real),axis=1)
      M1_init = np.concatenate((top,bot),axis=0)

      top = np.concatenate((M2.real, -M2.imag),axis=1)
      bot = np.concatenate((M2.imag, M2.real),axis=1)
      M2_init = np.concatenate((top,bot),axis=0)

    self.M1 = tf.Variable(initial_value=M1_init.astype(np.float32),
                         trainable=not tied, name='M1')
    self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
                         trainable=not tied, name='M2')

    self.alph = tf.Variable(initial_value=self.alpha0,
                         trainable=True, name='alpha')

    self.beta = tf.Variable(initial_value=1.,
                         trainable=True, name='beta')

    self.rho = tf.Variable(initial_value=self.rho0,
                         trainable=False, name='rho')

    self.lamb = tf.Variable(initial_value=params_init['lambda'],
                            trainable=True, name='lambda')

    if self.p.partition == True:
      self.lamb2 = tf.Variable(initial_value=params_init['lambda2'],
                              trainable=True, name='lambda2')
                      
  
  def AULA(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky(np.eye(M) + (1/self.rho0)*np.matmul(p.A,p.A.T.conj()))
    U = L.T.conj()

    return np.matmul(p.A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),p.A)))

  def call(self, y, z, u):
    x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, (z-u))

    
    x = self.alph*x + (1-self.alph)*z

    
    v = self.re2comp(x+u)

    if self.p.partition == True:
      zc = self.z_update_partition(v)
    else:
      zc = self.z_update_no_partition(v)

    z = self.comp2re(zc)

    u = u + self.beta*(x - z)

    return z,u
  
  def z_update_partition(self, v):
    z1 = self.soft_thresh_complex(v[:self.p.N_part], self.lamb/self.rho)
    z2 = self.soft_thresh_complex(v[self.p.N_part:], self.lamb2/self.rho)
    return tf.concat((z1,z2),axis=0)

  def z_update_no_partition(self, v):
    return self.soft_thresh_complex(v, self.lamb/self.rho)

  def comp2re(self, x):
    return tf.concat((x[:,0],x[:,1]), axis=0)

  def re2comp(self, x):
    # input: x is a shape (2N, 1) real-valued concatenation of a length-N complex vector
    # output: a shape (N, 2) array corresponding to the recomposed complex 
    
    ndiv2 = 2*self.n//2

    x_re = x[:ndiv2]
    x_im = x[ndiv2:]
    
    return tf.concat((x_re[:,None],x_im[:,None]), axis=1)

  def soft_thresh_complex(self, x, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers
    
    x_re = x[:,0]
    x_im = x[:,1]

    norm = tf.norm(x,axis=1)
    x_re_normalized = tf.math.divide_no_nan(x_re,norm)
    x_im_normalized = tf.math.divide_no_nan(x_im,norm)

    z_re = tf.math.multiply(x_re_normalized,tf.maximum(norm - kappa,0))
    z_im = tf.math.multiply(x_im_normalized,tf.maximum(norm - kappa,0))

    return tf.concat((z_re[:,None],z_im[:,None]),axis=1)

class StageFinal(layers.Layer):

  def __init__(self, params_init, p, tied, *args):
    super().__init__()
    m = p.size(0);
    n = p.size(1);
    self.m = m
    self.n = n
    self.p = p
    self.rho0 = params_init['rho']
    self.alpha0 = params_init['alpha']
    self.lambda0 = params_init['lambda']
    self.lambda0_2 = params_init['lambda2']
    
    print('Tied stage' if tied else 'Untied stage')
    
    if 'random_init' in args:
      M1_init = np.random.normal(size=(2*n,2*m))
      M2_init = np.random.normal(size=(2*n,2*n))
      # normalize cols
      M1_init = np.matmul(M1_init,np.diag(1/np.sqrt(np.sum(M1_init**2,axis=0))))
      M2_init = np.matmul(M2_init,np.diag(1/np.sqrt(np.sum(M2_init**2,axis=0))))
    else:
      AULA = self.AULA(p)
      M1 = np.matmul(np.eye(n)/self.rho0 - (1/self.rho0**2)*AULA, p.A.T.conj())
      M2 = np.eye(n) - (1/self.rho0)*AULA

      top = np.concatenate((M1.real, -M1.imag),axis=1)
      bot = np.concatenate((M1.imag, M1.real),axis=1)
      M1_init = np.concatenate((top,bot),axis=0)

      top = np.concatenate((M2.real, -M2.imag),axis=1)
      bot = np.concatenate((M2.imag, M2.real),axis=1)
      M2_init = np.concatenate((top,bot),axis=0)

    self.M1 = tf.Variable(initial_value=M1_init.astype(np.float32),
                         trainable=not tied, name='M1')
    self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
                         trainable=not tied, name='M2')

    self.alph = tf.Variable(initial_value=self.alpha0,
                         trainable=True, name='alpha')
    
    self.beta = tf.Variable(initial_value=1.,
                         trainable=False, name='beta')

    self.rho = tf.Variable(initial_value=self.rho0,
                         trainable=False, name='rho')

    self.lamb = tf.Variable(initial_value=params_init['lambda'],
                            trainable=True, name='lambda')

    if self.p.partition == True:
      self.lamb2 = tf.Variable(initial_value=params_init['lambda2'],
                              trainable=True, name='lambda2')
                            
  
  def AULA(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky(np.eye(M) + (1/self.rho0)*np.matmul(p.A,p.A.T.conj()))
    U = L.T.conj()
    return np.matmul(p.A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),p.A)))

  def call(self, y, z, u):
    x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, (z-u))

    
    x = self.alph*x + (1-self.alph)*z

    
    v = self.re2comp(x+u)

    if self.p.partition == True:
      zc = self.z_update_partition(v)
    else:
      zc = self.z_update_no_partition(v)

    z = self.comp2re(zc)

    u = u + (x - z)

    return z,u
  
  def z_update_partition(self, v):
    z1 = self.soft_thresh_complex(v[:self.p.N_part], self.lamb/self.rho)
    z2 = self.soft_thresh_complex(v[self.p.N_part:], self.lamb2/self.rho)
    return tf.concat((z1,z2),axis=0)

  def z_update_no_partition(self, v):
    return self.soft_thresh_complex(v, self.lamb/self.rho)

  def comp2re(self, x):
    return tf.concat((x[:,0],x[:,1]), axis=0)

  def re2comp(self, x):
    # input: x is a shape (2N, 1) real-valued concatenation of a length-N complex vector
    # output: a shape (N, 2) array corresponding to the recomposed complex 
    
    ndiv2 = 2*self.n//2

    x_re = x[:ndiv2]
    x_im = x[ndiv2:]
    
    return tf.concat((x_re[:,None],x_im[:,None]), axis=1)

  def soft_thresh_complex(self, x, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers
    
    x_re = x[:,0]
    x_im = x[:,1]

    norm = tf.norm(x,axis=1)
    x_re_normalized = tf.math.divide_no_nan(x_re,norm)
    x_im_normalized = tf.math.divide_no_nan(x_im,norm)

    z_re = tf.math.multiply(x_re_normalized,tf.maximum(norm - kappa,0))
    z_im = tf.math.multiply(x_im_normalized,tf.maximum(norm - kappa,0))

    return tf.concat((z_re[:,None],z_im[:,None]),axis=1)