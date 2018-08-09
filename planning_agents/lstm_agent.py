#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:16:05 2018

@author: aidanrocke
"""
import numpy as np
import tensorflow as tf

class lstm_agent:
    
    def __init__(self,random_seed,depth=1):
        
        self.seed = random_seed
        self.depth = depth
                        
        self.X_t = tf.placeholder(tf.float32, [1,5,5,1]) 
        
        self.H_prev = tf.Variable(initial_value = tf.zeros(shape=[1,5,5,5]),dtype=tf.float32)
        self.C_prev = tf.Variable(initial_value = tf.zeros(shape=[1,5,5,5]),dtype=tf.float32)
        
        self.input_ = self.input_gate()
        self.forget_ = self.forget_gate()
        self.cell_ = self.cell_state()
        self.output_ = self.output_gate()
        self.H = tf.multiply(self.output_,tf.nn.tanh(self.cell_))
        self.next_move = self.next_move()
        
        self.update = self.update()
    

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
    
    def input_gate(self):
        
        with tf.variable_scope("input_gate",reuse=tf.AUTO_REUSE):
    
            input_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            input_2 = self.conv2d(input_tensor=self.H_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            input_3 = self.hadamard(self.C_prev,"hadamard_1")
            
            input_sum = tf.add_n([input_1,input_2,input_3])
        
        return tf.nn.sigmoid(input_sum)

    def forget_gate(self):
        
        with tf.variable_scope("forget_gate",reuse=tf.AUTO_REUSE):
        
            forget_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            forget_2 = self.conv2d(input_tensor=self.H_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            forget_3 = self.hadamard(self.C_prev,"hadamard_1")
            
            forget_sum = tf.add_n([forget_1,forget_2,forget_3])
        
        return tf.nn.sigmoid(forget_sum)

    def cell_state(self):
        
        with tf.variable_scope("cell_state",reuse=tf.AUTO_REUSE):
        
            cell_1 = tf.multiply(self.forget_,self.C_prev)
            
            cell_2 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,self.depth,5])
            
            cell_3 = self.conv2d(input_tensor=self.H_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            cell_4 = tf.nn.tanh(tf.add(cell_2,cell_3))
            
            cell_5 = tf.multiply(self.input_,cell_4)
        
        return tf.add(cell_1,cell_5)
    
    def output_gate(self):
        
        with tf.variable_scope("output_gate",reuse=tf.AUTO_REUSE):
        
            out_1 = self.conv2d(input_tensor=self.X_t,
                             weight_init='glorot',
                             name='conv_1',
                             filter_shape=[3,3,self.depth,5])
            
            out_2 = self.conv2d(input_tensor=self.H_prev,
                             weight_init='glorot',
                             name='conv_2',
                             filter_shape=[3,3,5,5])
            
            out_3 = self.hadamard(self.C_prev,"hadamard_1")
            
            out_sum = tf.add_n([out_1,out_2,out_3])
        
        return tf.nn.sigmoid(out_sum)
    
    def next_move(self):
        """
            A function that approximates the probability map for the next move. 
        """
        
        out = tf.layers.conv2d(inputs = self.H,filters=3,
                                        padding="same",
                                        kernel_size=[3, 3],
                                        activation=None)
            
        flat = tf.layers.Flatten()(out)
                
        probabilities = tf.layers.dense(inputs=flat,units=9,activation=tf.nn.softmax)
        
        return probabilities
    
    def update(self):
        """
            A simple method for updating the hidden and cell variables.
        """
        
        self.H_prev = self.H
        self.C_prev = self.cell_
        