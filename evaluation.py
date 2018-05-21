#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:23:03 2018

@author: aidanrocke
"""

import numpy as np

class game_evaluation:
    
    def __init__(self,matrix):
        self.matrix = matrix
        self.R = np.array([[0,0,1],[0,1,0],[1,0,0]]) ## reflection about y=x

        self.score = self.scores(matrix)
        self.draw = self.draw()
        self.X_score = self.X_score()+0.5*self.draw
        self.O_score = -1.0*self.X_score+0.5*self.draw
        self.reward = self.reward()

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
    
    def X_score(self):
        
        if np.max(self.score) == 3:
            return 1.0
        
        elif np.min(self.score) == -3.0:
            return -1.0
        
        else:
            return 0.0
        
    def draw(self):
                
        if np.min(self.scores(np.abs(self.matrix))) >= 2.0:
            
            return np.max(np.abs(self.score)) <= 1.0
        
        else:
            return 0
    
    def reward(self):
        """
         We use a risk-averse heuristic for calculating the reward. 
        """
        
        if np.min(self.score) <= -2.0:
        
            return np.min(self.score)*5
    
        elif np.max(self.score) >= 2.0:
        
            return np.max(self.score)*5
        
        else:
            return np.mean(self.score[np.nonzero(self.score)])
            
    
    