#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:58:06 2018

@author: aidanrocke
"""

import tensorflow as tf

class policy_gradients:
    
    def __init__(self,lr,seed,batch_size,p_est):
        self.batch_size = batch_size ## number of rollouts
        self.T = 0.1 ## the temperature
        
        self.state = tf.placeholder(tf.float32, [None, 9]) ## the board representation
        self.action = tf.placeholder(tf.float32, [None, 9]) ## the agent's action
        self.state_action = tf.placeholder(tf.float32, [None, 18]) ## (state,action)
        self.reward = tf.placeholder(tf.float32, [None, 1]) ## the time-discounted reward signal
        self.seed = seed ## the random seed
        
        ## define output of policy network:
        self.policy = self.controller()
        
        ## define the probability distribution:
        self.dist = self.multinomial()
        self.log_prob = self.log_prob()*(p_est == 1) + self.max_log_prob()*(p_est == 2) + \
                        self.gumbel_softmax(self.T)*(p_est ==3)
        self.sample_action = self.sample_action()
        
        ## define what is necessary for the loss:
        self.reinforce_loss = self.reinforce_loss()
        self.value_estimate = self.value_estimator()
        self.baseline = self.baseline() 
        
        self.average_loss = -1.0*tf.reduce_mean(self.reinforce_loss)
        
        self.average_loss = -1.0*tf.reduce_mean(tf.subtract(self.reinforce_loss,self.baseline)) + \
                            0.5*tf.reduce_mean(tf.square(self.value_estimate-self.reward))
        
        ## collect trainable variables:
        self.TV = tf.trainable_variables()
        
        #self.TV = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES,
        #                             scope= "policy_net")
        
        ## define training operations:
        self.optimizer = tf.train.AdagradOptimizer(lr)
        
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.TV]                                        
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
        
        self.gvs_ = self.optimizer.compute_gradients(self.average_loss, self.TV)
        #self.gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs_] ## clip gradients
        self.gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val) for grad,val in self.gvs_]
        
        self.accum_ops = [self.accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs)]
        
        self.train_step = self.optimizer.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(self.gvs)])
    
        self.init_g = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()
    def init_weights(self,shape,var_name):
        """
            Xavier initialisation of neural networks
        """
        initializer = tf.contrib.layers.xavier_initializer(seed=self.seed)
        
        return tf.Variable(initializer(shape),name = var_name)
    
    def controller(self):
        """
            The policy gradient model is a neural network that 
            parametrises a categorical softmax distribution.  
            
            input: state(i.e. board position)
            output: a 9-dimensional vector that parametrises a softmax distribution. 
        
        """
        
        with tf.variable_scope("policy_net"):
            
            tf.set_random_seed(self.seed)
    
            w_h = self.init_weights([9,30],"w_h")
            w_o = self.init_weights([30,9],"w_o")
            
            ### bias terms:
            bias_1 = self.init_weights([30],"bias_1")
            bias_2 = self.init_weights([9],"bias_2")
                
            h = tf.nn.elu(tf.add(tf.matmul(self.state, w_h),bias_1))
            
        return tf.add(tf.matmul(h, w_o),bias_2)
    
    def value_estimator(self):
        """
        This value function is used to approximate the expected value of each state. 
        
        input: state-action vector
        output: value estimate
        """
        
        with tf.variable_scope("value_estimate"):
            
            tf.set_random_seed(self.seed)
    
            w_h = self.init_weights([18,50],"w_h")
            w_o = self.init_weights([50,1],"w_o")
            
            ### bias terms:
            bias_1 = self.init_weights([50],"bias_1")
            bias_2 = self.init_weights([1],"bias_2")
                
            h = tf.nn.elu(tf.add(tf.matmul(self.state_action, w_h),bias_1))
            
        return tf.nn.elu(tf.add(tf.matmul(h, w_o),bias_2))
    
    def multinomial(self):
                
        ## identify the free positions:    
        free_positions = tf.to_float(tf.equal(self.state,tf.zeros((1,9))))
    
        fm_mapping = lambda x: tf.diag(tf.reshape(x,(9,)))
    
        free_matrices = tf.map_fn(fm_mapping,free_positions)


        ## calculate probability vector:
        pvec_mapping = lambda x: tf.transpose(tf.matmul(x,tf.transpose(self.policy)))
        
        prob_vec = tf.map_fn(pvec_mapping,free_matrices)
        prob = prob_vec/(tf.reduce_sum(prob_vec)+tf.constant(1e-5))
    
        return tf.contrib.distributions.Multinomial(total_count=1., probs=prob)
    
    def sample_action(self):
        """
            Samples an action from the stochastic controller which happens
            to be a softmax distribution on available positions. 
        """
        ## define multinomial distribution:
        
        return self.dist.sample()
    
    def max_log_prob(self):
        
        ## use softmax to calculate probabilities:
        probs = tf.nn.softmax(self.policy)
        
        ## approximate the maximum probability:
        max_p = tf.pow(tf.reduce_sum(tf.pow(probs,15)),1/15)
        
        return tf.log(max_p+tf.constant(1e-8))
    
    def log_prob(self):
        
        ## use softmax to calculate probabilities:
        probs = tf.nn.softmax(self.policy)
        
        return tf.log(probs+tf.constant(1e-10))
    
    def sample_gumbel(self,shape, eps=1e-20): 
      """Sample from Gumbel(0, 1)"""
      U = tf.random_uniform(shape,minval=0,maxval=1)
      return -tf.log(-tf.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self,logits, temperature): 
      """ Draw a sample from the Gumbel-Softmax distribution"""
      y = logits + self.sample_gumbel(tf.shape(logits))
      return tf.nn.softmax( y / temperature)
    
    def gumbel_softmax(self,temperature):
      """Sample from the Gumbel-Softmax distribution and optionally discretize.
      Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
      Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
      logits = tf.log(tf.nn.softmax(self.policy))
      
      y = self.gumbel_softmax_sample(logits, temperature)
    
      return tf.log(y+tf.constant(1e-10))
            
    def reinforce_loss(self):
        """
            The REINFORCE loss without subtracting a baseline. 
        """
                                                
        return self.log_prob*self.reward
    
    def baseline(self):
        """
            A state-dependent baseline calculated using the value estimator V(s,a).
        """

        return self.log_prob*self.value_estimate