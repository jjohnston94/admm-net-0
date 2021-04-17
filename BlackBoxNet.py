import tensorflow as tf
import numpy as np
from pdb import set_trace
from tensorflow.keras import layers

class BBNet(tf.keras.Model):
  
  def __init__(self, p, dims_list):
    super().__init__()
    self.n = p.A.shape[1]
    self.loss_fcn = tf.keras.losses.MeanSquaredError()
    self.Layers=[]
    # self.Layers.append(StageIn(p))
    for dims in dims_list:
      self.Layers.append(Stage(p,dims))
  
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
    # set_trace() 
    x = tf.transpose(inputs)
    for l in self.Layers:
      x = l(x)

    return tf.transpose(x)

class StageIn(layers.Layer):
  def __init__(self, p):
    super().__init__()
    m = p.size(0)
    n = p.size(1)
    self.m = m
    self.n = n
    self.p = p
    
    M1 = p.A.T.conj()

    top = np.concatenate((M1.real, -M1.imag),axis=1)
    bot = np.concatenate((M1.imag, M1.real),axis=1)
    M1_init = np.concatenate((top,bot),axis=0)

    self.W = tf.Variable(initial_value=M1_init.astype(np.float32),
                         trainable=False, name='M1')
    # self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
    #                      trainable=not tied, name='M2')

  def call(self, x):
    return tf.matmul(self.W, x)

class Stage(layers.Layer):

  def __init__(self, p, dims):
    super().__init__()
    m,n = dims
    
    W_init = np.random.normal(size=(2*m,2*n))
    # normalize cols
    W_init = np.matmul(W_init,np.diag(1/np.sqrt(np.sum(W_init**2,axis=0))))

    # ww = np.random.normal(size=(2*m,2*n)) + 1j*np.random.normal(size=(2*m,2*n))
    # top = np.concatenate((ww.real, -ww.imag),axis=1)
    # bot = np.concatenate((ww.imag, .real),axis=1)
    # W_init = np.concatenate((top,bot),axis=0)

    self.W = tf.Variable(initial_value=W_init.astype(np.float32),
                         trainable=True, name='W')
    # self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
    #                      trainable=not tied, name='M2')
    self.b = tf.Variable(initial_value=np.float32(np.random.randn(2*m,1)/10),
                         trainable=True, name='bias')

  def call(self, x):
    return tf.maximum(0.,tf.matmul(self.W, x) + self.b)