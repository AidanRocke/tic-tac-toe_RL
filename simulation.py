#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:35:57 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation
from perfect_play import perfect_play

num_games = 10


def game_simulation(num_games):

    outcomes = np.zeros(num_games)
    
    
    initial = []
    
    for i in range(num_games):
        
        game = 1.0
        
        Z = np.zeros((3,3))
        X, O = np.random.choice(np.arange(9),2,replace=False)
        Z[int(X/3)][X % 3] = 1.0
        Z[int(O/3)][O % 3] = -1.0
        
        initial.append(np.copy(Z))
    
        while game == 1.0:
            
            ## computer A move:
            P = perfect_play(Z,5,0.5)
        
            Z += P.move()
            
            if np.sum(np.abs(Z)) >= 9.0:
                game_eval = game_evaluation(Z)
                outcomes[i] = game_eval.X_score
                
                game = 0.0
                
                break
                
            ## computer B move:
            P = perfect_play(-1.0*Z,5,0.5)
            
            Z += -1.0*P.move()
            
    return initial, outcomes
        

    
    