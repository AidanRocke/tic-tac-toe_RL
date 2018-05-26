#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:58:06 2018

@author: aidanrocke
"""

import tensorflow as tf

class policy_gradients:
    
    def __init__(self,seed,batch_size):
        self.batch_size = batch_size ## number of rollouts
        
        self.state = tf.placeholder(tf.float32, [None, 9]) ## the board representation
        self.action = tf.placeholder(tf.float32, [None, 9]) ## the agent's action
        self.state_action = tf.placeholder(tf.float32, [None, 18]) ## (state,action)
        self.reward = tf.placeholder(tf.float32, [None, 1]) ## the time-discounted reward signal
        self.seed = seed ## the random seed
        
        ## define output of policy network:
        self.policy = self.controller()
        
        ## define the probability distribution:
        self.dist = self.multinomial()
        self.log_prob = self.log_prob()
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
        self.optimizer = tf.train.AdagradOptimizer(0.01)
        
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.TV]                                        
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
        
        self.gvs_ = self.optimizer.compute_gradients(self.average_loss, self.TV)
        #self.gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs_] ## clip gradients
        self.gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val) for grad,val in self.gvs_]
        
        self.accum_ops = [self.accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs)]
        
        self.train_step = self.optimizer.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(self.gvs)])
    
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
    
    #def log_prob(self):
        
    #    log_p = tf.log(self.dist.prob(self.action)+tf.constant(1e-8))
                
    #    return tf.where(tf.is_nan(log_p), tf.ones_like(log_p) * 0.1, log_p)
    
    def log_prob(self):
        
        ## use softmax to calculate probabilities:
        probs = tf.nn.softmax(self.policy)
        
        ## approximate the maximum probability:
        max_p = tf.pow(tf.reduce_sum(tf.pow(probs,100)),1/100)
        
        return tf.log(max_p+tf.constant(1e-8))
            
    def reinforce_loss(self):
        """
            The REINFORCE loss without subtracting a baseline. 
        """
                                                
        return self.log_prob*self.reward
    
    def baseline(self):
        """
            A state-dependent baseline calculated using the value estimator V(s,a).
        """
                
        return  self.log_prob*self.value_estimate