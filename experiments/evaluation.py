#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:23:03 2018

@author: aidanrocke
"""

import numpy as np

class game_evaluation:
    
    def __init__(self):
        self.R = np.array([[0,0,1],[0,1,0],[1,0,0]]) ## reflection about y=x

    def scores(self,matrix):
        """
            On an NxN board there are only 2*(N+1) ways of winning.
        """
        scores = np.zeros(8)
        
        scores[:3] = np.sum(matrix,0) ## rows
        scores[3:6] = np.sum(matrix,1) ## columns
        scores[6] = np.trace(matrix) ## first diagonal
        scores[7] = np.trace(np.matmul(self.R,matrix)) ## second diagonal
        
        return scores
    
    def X_score(self,matrix):
        
        scores, draw = self.scores(matrix), self.draw(matrix)
        
        if np.max(scores) == 3:
            return 1.0 + draw
        
        elif np.min(scores) == -3.0:
            return -1.0 + draw
        
        else:
            return 0.0 + draw
        
    def draw(self,matrix):
                        
        if np.min(self.scores(np.abs(matrix))) >= 2.0:
            
            return np.max(np.abs(self.scores(matrix))) <= 1.0
        
        else:
            return 0
    
    def reward(self,matrix):
        """
         We use a risk-averse heuristic for calculating the reward. 
        """
        
        scores = self.scores(matrix)
        
        if np.min(scores) <= -2.0:
        
            return np.min(scores)*5
    
        elif np.max(scores) >= 2.0:
        
            return np.max(scores)*5
        
        else:            
            return np.mean(scores[np.nonzero(scores)])
            
    
    