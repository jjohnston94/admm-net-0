import gc
import problem
import data
import numpy as np
import time
import os
from pdb import set_trace
import tensorflow as tf


def nmse(X,Xhat):

  return 10*np.log10(np.mean(np.sum((Xhat-X)**2)/np.sum(X**2)))


def train_net(net, p, datatrain, datatest, lr, bs, savepath=None):
  
  Xtrain, Ytrain = datatrain
  Xtest, Ytest = datatest

  Ntr = len(Xtrain)

  print('Initial test error:', nmse(Xtest, net(Ytest)))
  print('Initial train error =', nmse(Xtrain, net(Ytrain)))

  net.compile(tf.keras.optimizers.Adam(learning_rate=lr[0]))
  net.summary()

  for i in range(len(lr)):
    gc.collect()
    net.optimizer.lr.assign(lr[i])
    print('Epoch', str(i+1), 'learning rate =', '{:.0e}'.format(lr[i]))

    progbar = tf.keras.utils.Progbar(Ntr//bs)
    nb = 0
    for b in list(range(0,Ntr,bs)):
      
      v = net.train_step(Ytrain[b:b+bs], Xtrain[b:b+bs])
      nb += 1
      if nb%100==0:
        error = nmse(Xtest, net(Ytest))
        progbar.update(nb, values=[('Train loss',v),('Test NMSE (dB)',error)])
      else:
        progbar.update(nb, values=[('Train loss',v)])
    error = nmse(Xtest, net(Ytest))

    if savepath is not None:
      path = savepath + '/' \
           + 'Epoch=' + str(i+1) \
           + '_Rate=' + '{:.0e}'.format(lr[i]) \
           + '_Batch=' + str(bs) \
           + '_Error=' + '{:.3f}'.format(error)
      save(net, p, path)
    
  return net

def save(net, p, filepath):
  # os.mkdir(filepath)
  net.save_weights(filepath+'/weights')
  np.save(filepath+'/A', p.A)



