#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:35:57 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation
from simple_play import simple_play
from stochastic_play import stochastic_play

num_games = 10
random_start = 0.0
depth = 5
gamma = 0.5

def game_simulation(num_games,random_start,depth,gamma):

    outcomes = np.zeros(num_games)
    
    
    initial = []
    
    for i in range(num_games):
        
        game = 1.0
        
        Z = np.zeros((3,3))
        X, O = np.random.choice(np.arange(9),2,replace=False)
        Z[int(X/3)][X % 3] = 1.0
        
        if random_start == 1.0:
            ## the second player plays randomly:
            Z[int(O/3)][O % 3] = -1.0
            
        else:
            ## the second player doesn't play randomly:
            P2 = simple_play(-1.0*Z,depth,gamma)
            Z += -1.0*P2.move()
        
        
        initial.append(np.copy(Z))
    
        while game == 1.0:
            
            ## computer A move:
            P1 = stochastic_play(Z,depth,gamma)
        
            Z += P1.move()
            
            if abs(P1.reward(Z)) >= 50.0:
                outcomes[i] = P1.reward(Z)/100.0
                
                game = 0.0
                
                break
                
            ## computer B move:
            P2 = simple_play(-1.0*Z,depth,gamma)
            
            Z += -1.0*P2.move()
            
    return initial, outcomes
        

    
    