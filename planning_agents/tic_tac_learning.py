#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:28:04 2018

@author: aidanrocke
"""

#import numpy as np
import tensorflow as tf

## The simulator is used to train the policy gradient model. 
## More details in tic_tac_system. 


def simulator(tic_tac):    
        
    with tf.Session() as sess:
        
        ### initialise the variables:
        sess.run(tic_tac.model.init_g)
        #sess.run(tic_tac.model.init_l)
        
        for i in range(tic_tac.epochs-1):
            
            tic_tac.batch_update(sess) 
            
            if i % 100 == 0:
                
                print(i)

                #print(np.mean((tic_tac.score[i*100:(i+1)*100]+5.0)/10))
            
    scores = tic_tac.score         
            
    return scores

