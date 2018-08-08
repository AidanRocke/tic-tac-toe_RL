#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:33:06 2018

@author: aidanrocke
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial


def multinomial(policy,game_state):
            
    ## identify the free positions:    
    free_positions = tf.to_float(tf.equal(game_state,tf.zeros((1,9))))

    fm_mapping = lambda x: tf.diag(tf.reshape(x,(9,)))

    free_matrices = tf.map_fn(fm_mapping,free_positions)


    ## calculate probability vector:
    pvec_mapping = lambda x: tf.transpose(tf.matmul(x,tf.transpose(policy)))
    
    prob_vec = tf.map_fn(pvec_mapping,free_matrices)
    prob = prob_vec/(tf.reduce_sum(prob_vec)+tf.constant(1e-5))

    return Multinomial(total_count=1., probs=prob)

game_state = tf.constant(value=np.random.choice(a=[0,1],size=(9,9),p=[0.5,0.5]),dtype=tf.float32)
policy = tf.nn.softmax(game_state)

multi = multinomial(policy,game_state)

with tf.Session() as sess:
    
    M = sess.run(multi.sample())
    
    #print(np.shape(M))
    
    print(np.sum(M) == 81.0)