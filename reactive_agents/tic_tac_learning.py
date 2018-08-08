#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:28:04 2018

@author: aidanrocke
"""

import tensorflow as tf
import numpy as np

## The simulator is used to train the policy gradient model. 
## More details in tic_tac_system. 

#from clever_stochastician import clever_stochastician as CS

# model = policy_gradients(lr=0.01,seed=42,batch_size=30)
#tic_tac = tic_tac_system(model,opponent,epochs,depth,gamma)


def simulator(tic_tac):    
        
    with tf.Session() as sess:
        
        ### initialise the variables:
        sess.run(tic_tac.model.init_g)
        sess.run(tic_tac.model.init_l)
        
        for i in range(tic_tac.epochs-1):
            
            tic_tac.batch_update(sess) 
            
            if i % 10 == 0:

            	print(i)
            
    scores = tic_tac.score         
            
    return scores

