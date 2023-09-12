
import tensorflow as tf 
import numpy as np 

def _scaled_noise(size, dtype):
    x = tf.random.normal(shape=size, dtype=dtype)
    return tf.sign(x) * tf.sqrt(tf.abs(x))

class NoisyLayer0():
    
    def __init__(self, input_dims, output_dims, activation = tf.identity,
                 sigma = 0.5,
                 use_factorised = True, 
                 i_d = 0):
        
        ## initialization Parameter 
        self.use_factorised = use_factorised
        self.p = input_dims 
        self.q = output_dims
        self.dtype = tf.float32
        self.sigma = sigma
        self.f = activation
        ### Initialization 
        if self.use_factorised:
            
            mu_w_0 = tf.random.uniform(shape = (self.p,self.q), 
                                       minval = -tf.math.sqrt(1/self.p),
                                       maxval = tf.math.sqrt(1/self.p))
            
            
            sigma_w_0 = tf.constant( self.sigma * tf.math.sqrt(1/self.p),
                                    shape = (self.p, self.q))
            
            mu_b_0 = tf.random.uniform(shape = (self.q,), 
                                       minval = -tf.math.sqrt(1/self.p),
                                       maxval = tf.math.sqrt(1/self.p))
            
            sigma_b_0 = tf.constant(self.sigma* tf.math.sqrt(1/self.p),
                                    shape = (self.q))
            
            
        else: #(Independent noise)
          
            mu_w_0 = tf.random.uniform(shape = (self.p,self.q), 
                                       minval = -tf.math.sqrt(3/self.p),
                                       maxval = tf.math.sqrt(3/self.p))
            
            sigma_w_0 = tf.constant(0.017, shape = (self.p, self.q))
            
            mu_b_0 = tf.random.uniform(shape = (self.q,), 
                                       minval = -tf.math.sqrt(3/self.p),
                                       maxval = tf.math.sqrt(3/self.p))
            sigma_b_0 = tf.constant(0.017, shape = (self.q))
            
            # tf.Variable(initial_value = W0, name = 'W_dense_%i' % i_d)
            
        ### Define the variables 
        self.mu_w = tf.Variable(initial_value = mu_w_0, trainable=True,  name = 'mu_w' + str(i_d))
        
        self.sigma_w = tf.Variable(initial_value = sigma_w_0, trainable=True,  name = 'sigma_w' +str(i_d))
        
        self.mu_b = tf.Variable(initial_value=mu_b_0, trainable=True,  name = 'mu_b'+str(i_d))
        
        self.sigma_b = tf.Variable(initial_value=sigma_b_0, trainable=True, name = 'sigma_b'+str(i_d))
        
        ## create tranable variable 
        self.eps_kernel = tf.Variable(initial_value= tf.zeros(
            shape = (self.p,self.q)),trainable=False, name = 'eps_w'+str(i_d) )
        self.eps_bias = tf.Variable(initial_value= tf.zeros(
            shape =  (self.q)),trainable=False, name = 'eps_b' +str(i_d))
        
        self.trainabel_variables = [self.sigma_w,self.mu_w ,  self.sigma_b, self.mu_b]
        
    def reset_noise(self):
        """Create the factorised Gaussian noise."""

        if self.use_factorised:
            # Generate random noise
            in_eps = _scaled_noise([self.p, 1], dtype=self.dtype)
            out_eps = _scaled_noise([1, self.q], dtype=self.dtype)
            # Scale the random noise
            self.eps_kernel.assign(tf.matmul(in_eps, out_eps))
            self.eps_bias.assign(out_eps[0])
        else:
            # generate independent variables
            self.eps_kernel.assign(
                tf.random.normal(shape=[self.p, self.q], dtype=self.dtype)
            )
            self.eps_bias.assign(
                tf.random.normal(
                    shape=[
                        self.q,
                    ],
                    dtype=self.dtype,
                )
            )
            
    def remove_noise(self):
        """Remove the factorised Gaussian noise."""

        self.eps_kernel.assign(tf.zeros([self.p, self.q], dtype=self.dtype))
        self.eps_bias.assign(tf.zeros([self.q], dtype=self.dtype))
    
    def __call__(self, inputs, training):
        # TODO(WindQAQ): Replace this with `dense()` once public.
        w = self.mu_w + (self.sigma_w *self.eps_kernel) # in pxq
        # print(w)
        b = self.mu_b + (self.sigma_b * self.eps_bias) # in q
        
        z = tf.matmul(inputs, w) + b # inputs in Nxp
        
        return self.f(z) 
        
