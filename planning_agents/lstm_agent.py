#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:16:05 2018

@author: aidanrocke
"""
import numpy as np
import tensorflow as tf

class lstm_agent:
    
    def __init__(self,random_seed,max_iter,depth=1):
        
        self.seed = random_seed
        self.max_iter = max_iter
        self.depth = depth
                        
        self.X_t = tf.placeholder(tf.float32, [1,5,5,1]) 
                
        self.policy, self.variance = self.policy_iteration()
                    

    def conv2d(self,input_tensor,weight_init,name,filter_shape):
                
            if weight_init == 'glorot':
                
                ## set random seed
                np.random.seed(self.seed)
                
                H,W,c_in,c_out = filter_shape
        
                weight_init = np.random.rand(H,W,c_in,c_out).astype(np.float32) * \
                              np.sqrt(6.0/(c_in+c_out))
                bias_init =   np.zeros([c_out]).astype(np.float32)
                
                kernel = tf.Variable(initial_value=weight_init)   
                bias = tf.Variable(initial_value=bias_init)
            
            conv = tf.nn.conv2d(input=input_tensor,filter=kernel,padding="SAME",
                                strides=(1,1,1,1),name=name)
            
            return tf.nn.bias_add(conv, bias)
    
    def hadamard(self,input_tensor,name):
        
        w,x,y,z = input_tensor.get_shape().as_list()
        
        W = tf.get_variable("W", shape=[w,x,y,z],
               initializer=tf.contrib.layers.xavier_initializer())
        
        return tf.multiply(input_tensor,W)
    
    def input_gate(self,h_prev,c_prev):
        
        with tf.variable_scope("input_gate",reuse=tf.AUTO_REUSE):
    
            input_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            input_2 = self.conv2d(input_tensor=h_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            input_3 = self.hadamard(c_prev,"hadamard_1")
            
            input_sum = tf.add_n([input_1,input_2,input_3])
        
        return tf.nn.sigmoid(input_sum)

    def forget_gate(self,h_prev,c_prev):
        
        with tf.variable_scope("forget_gate",reuse=tf.AUTO_REUSE):
        
            forget_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            forget_2 = self.conv2d(input_tensor=h_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            forget_3 = self.hadamard(c_prev,"hadamard_1")
            
            forget_sum = tf.add_n([forget_1,forget_2,forget_3])
        
        return tf.nn.sigmoid(forget_sum)

    def cell_state(self,h_prev,c_prev,input_,forget_):
        
        with tf.variable_scope("cell_state",reuse=tf.AUTO_REUSE):
        
            cell_1 = tf.multiply(forget_,c_prev)
            
            cell_2 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,self.depth,5])
            
            cell_3 = self.conv2d(input_tensor=h_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            cell_4 = tf.nn.tanh(tf.add(cell_2,cell_3))
            
            cell_5 = tf.multiply(input_,cell_4)
        
        return tf.add(cell_1,cell_5)

    def output_gate(self,h_prev,cell_):
        
        with tf.variable_scope("output_gate",reuse=tf.AUTO_REUSE):
        
            out_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            out_2 = self.conv2d(input_tensor=h_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            out_3 = self.hadamard(cell_,"hadamard_1")
            
            out_sum = tf.add_n([out_1,out_2,out_3])
        
        return tf.nn.sigmoid(out_sum)
    
    def next_move(self,H):
        """
            A function that approximates the probability map for the next move. 
        """
        
        out = tf.layers.conv2d(inputs = H,filters=3,
                                        padding="same",
                                        kernel_size=[3, 3],
                                        activation=None)
            
        flat = tf.layers.Flatten()(out)
                
        probabilities = tf.layers.dense(inputs=flat,units=9,activation=tf.nn.softmax)
        
        return probabilities
    
    
    def policy_iteration(self):
            
            with tf.variable_scope("loops",reuse=tf.AUTO_REUSE):
                
                def body(iter_,p_array,h_prev,c_prev):
                                        
                    input_ = self.input_gate(h_prev,c_prev)
                    forget_ = self.forget_gate(h_prev,c_prev)
                    cell_ = self.cell_state(h_prev,c_prev,input_,forget_)
                    output_ = self.output_gate(h_prev,cell_)
                    hidden = tf.multiply(output_,tf.nn.tanh(cell_))
                    
                    next_move = self.next_move(hidden)
                    
                    p_array = p_array.write(iter_,next_move)
                    
                    h_prev, c_prev = hidden, cell_
                                                                            
                    return iter_+1, p_array,h_prev, c_prev
    
                def condition(iter_,p_array,h_prev,c_prev):
                    
                    return iter_ < self.max_iter
                
                ## reset the iterator and the arrays:
                iter_ = 0
                policy_array = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
                    
                H_prev = tf.zeros(shape=[1,5,5,5],name='H_prev')
                C_prev = tf.zeros(shape=[1,5,5,5],name='C_prev')
                
                _, policy_,hidden, cell = tf.while_loop(condition,body,[iter_,policy_array,H_prev,C_prev])
                
                policies = policy_.stack()
                
                #log_squared = tf.square(tf.log(policies))
                
                mu, variance = tf.nn.moments(policies,axes=0)
                
            return policies[-1], tf.reduce_mean(variance)