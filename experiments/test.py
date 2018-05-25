#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:36:01 2018

@author: aidanrocke
"""

import tensorflow as tf
import numpy as np


from clever_stochastician import clever_stochastician as opponent
model = policy_gradients(42,10)

tic_tac = tic_tac_system(G,model,opponent,100,5,0.9)

scores = simulator(tic_tac)


"""
state = tf.placeholder(tf.float32, [None, 9])
policy = tf.placeholder(tf.float32, [None, 9])
        
## identify the free positions:
free_positions = tf.to_float(tf.equal(state,tf.zeros((1,9))))

fm_mapping = lambda x: tf.diag(tf.reshape(x,(9,)))

free_matrices = tf.map_fn(fm_mapping,free_positions)


## calculate probability vector:
pvec_mapping = lambda x: tf.transpose(tf.matmul(x,tf.transpose(policy)))

prob_vec = tf.map_fn(pvec_mapping,free_matrices)
prob = prob_vec/(tf.reduce_sum(prob_vec)+tf.constant(1e-5))

    
dist = tf.contrib.distributions.Multinomial(total_count=1., probs=prob)

X = dist.sample()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    train_feed = {state: np.ones((9,9)),policy: np.zeros((9,9))}
    
    print(sess.run(X,feed_dict = train_feed))
"""