#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:28:04 2018

@author: aidanrocke
"""

import tensorflow as tf

## The simulator is used to train the policy gradient model. 
## More details in vanilla_system. 

#tic_tac = tic_tac_system(model,opponent,epochs,depth,gamma)
# model = policy_gradients(seed,batch_size)

def simulator(tic_tac):    
        
    with tf.Session() as sess:
        
        ### initialise the variables:
        sess.run(tf.global_variables_initializer())
        
        for i in range(tic_tac.epochs-1):
            
            tic_tac.batch_update(sess)   
            
    scores = tic_tac.score         
            
    return scores

