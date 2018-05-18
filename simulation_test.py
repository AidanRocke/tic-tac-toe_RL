#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:21:34 2018

@author: aidanrocke
"""

import numpy as np
from perfect_play import perfect_play

np.random.seed(0)

Q1 = np.array([[-1,0,0],[0,1,0],[0,0,0]])

## first move
P = perfect_play(Q1,5)

P.move()

## second move:
Q2 = np.array([[-1,-1,0],[0,1,0],[0,0,1]])

P = perfect_play(Q2,5)

P.move()

## third move:
Q3 = np.array([[-1,-1,1],[0,1,-1],[0,0,1]])

P = perfect_play(Q3,5)

P.move()




for i in range(P.max_depth):
            
    ## after each iteration we have fewer options:
                
    for j in range(P.num_actions):
                        
        ## update value of each position
        P.values[j] = P.evaluation(matrices[j]) ## I should halt as soon as abs(R) is maximal
        
        ## opponent's turn(min phase):
        P.update_turn()
        
        if P.num_actions <= 1:
            break
        
        ## adversarial generation and selection:
        selection, R = P.matrix_selection(P.matrix_generation(matrices[j]))
        
        if abs(R) == 3.0:
            break
        
        ## player's turn to select(max phase):
        P.update_turn()
        
        ## favorable generation and selection:
        M, R = P.matrix_selection(P.matrix_generation(selection[0]))
        
        matrices[j] = M[0]
        
    ## update depth:
    P.max_depth = int(9 - np.sum(np.abs(M)))