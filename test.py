#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:07:48 2018

@author: aidanrocke
"""

import numpy as np

Z = np.zeros((3,3))
X, O = np.random.choice(np.arange(9),2,replace=False)
Z[int(X/3)][X % 3] = 1.0
Z[int(O/3)][O % 3] = -1.0

depth = 5
gamma = 0.5


P = stochastic_play(Z,depth,gamma)


## step 1:
matrices = P.matrix_generation(P.initial)
batch = np.array([P.matrix_generation(matrices[i]) for i in range(len(matrices))])

batch_shape = np.shape(batch)

batch_indices = np.array([np.ones(batch_shape[1])*i for i in range(batch_shape[0])]).flatten()

batch = batch.reshape((batch_shape[0]*batch_shape[1],3,3))

def simulator(batch,batch_indices):

    while P.max_depth >= 1:
                
        ## min phase:
        P.update_turn()
        batch, batch_indices = P.matrix_selection(batch,batch_indices)
        
        ## max phase:
        P.update_turn()
        batch, batch_indices = P.matrix_selection(batch,batch_indices)
    
        ## update depth:
        P.max_depth = int(np.min([P.max_depth,int(P.num_positions/2)]))
        
simulator(batch,batch_indices)
    
## step 2: first moves
    
Z += P.move()

P = stochastic_play(-1.0*Z,depth,gamma)
            
Z += -1.0*P.move()

## step 3: second moves:

P = stochastic_play(Z,depth,gamma)

Z += 1.0*P.move()

P = stochastic_play(-1.0*Z,depth,gamma)
            
Z += -1.0*P.move()

