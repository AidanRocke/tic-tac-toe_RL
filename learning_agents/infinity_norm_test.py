#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 19:25:05 2018

@author: aidanrocke
"""

import numpy as np

quality = np.zeros(100)

for i in range(100):
    
    Z = np.random.rand(10)

    Z_ = Z/np.sum(Z)
    
    max_Z = np.max(Z_)
    
    ## use an approximation to the infinity norm:
    max_approx = np.sum(Z_**100)**(1/100)
    
    vals = [max_Z,max_approx]
    
    quality[i] = np.min(vals)/(np.max([np.max(vals),1e-10]))
    
    
## tensorflow infinity test:
import tensorflow as tf

# Use softmax on vector.
x = [0., -1., 2., 3.]
softmax_x = tf.nn.softmax(x)

# Create 2D tensor and use soft max on the second dimension.
max_p = tf.pow(tf.reduce_sum(tf.pow(softmax_x,100)),1/100)

session = tf.Session()
print("SOFTMAX X")
print(session.run(softmax_x))

print("max prob")
print(session.run(max_p))
    
    
    