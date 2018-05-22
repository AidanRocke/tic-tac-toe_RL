#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:33:51 2018

@author: aidanrocke
"""

import tensorflow as tf
import numpy as np
## checking that this boolean mask thing works:

Z = tf.to_float(tf.multinomial(tf.log([[10., 10.]]), 9,seed=tf.set_random_seed(42)))

q = tf.placeholder(tf.float32, [None, 9])

samples = tf.multinomial(tf.log([[.5, 0.5, 0.5, 0.5,0.5]]), 1)

y = tf.zeros((1,9))

m = tf.to_float(tf.equal(Z,q))

N = tf.diag(tf.reshape(m,(9,)))

out = tf.nn.softmax(tf.transpose(tf.matmul(N,tf.transpose(q))))

dist = tf.contrib.distributions.Multinomial(total_count=1., probs=out)


free_positions = tf.to_float(tf.equal(Z,tf.zeros((1,9))))

free_matrix = tf.diag(tf.reshape(free_positions,(9,)))

## calculate probability vector:
prob_vec = tf.transpose(tf.matmul(free_matrix,tf.transpose(tf.random_normal((1,9)))))
prob = prob_vec/tf.reduce_sum(prob_vec)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    #print(sess.run(tf.nn.softmax(Z)))
    
    print(sess.run(prob))
    
    #print(sess.run(samples)[0][0])
    
    print(sess.run(dist.sample(),feed_dict={q:np.random.choice((0,1),size=(1,9))}))
    
    #print(sess.run(tf.einsum('n,nm->m', N, Z).eval()))
    
    #print(sess.run(out,feed_dict={q:np.random.choice((0,1),size=(1,9))}))
    
    #print(sess.run(tf.multiply(N,tf.transpose(Z))))
