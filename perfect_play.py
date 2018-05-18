#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:33:13 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation

Q = np.array([[1,-1,-1],[-1,1,1],[0,1,-1]])

## define multi-dimensional array:

n = 100
X = np.zeros((n,3,3))

class perfect_play:
    
    def __init__(self,initial_matrix,max_depth,max_breadth):
        self.initial = initial_matrix
        self.max_depth = np.min([max_depth,self.num_select])
        self.max_breadth = np.max([100,max_breadth])
        
        ## num_positions*num_select ~ max_breadth
        self.num_positions = 9 - np.sum(np.abs(self.initial))
        self.num_select = int(self.max_breadth/self.num_positions)
        self.num_actions = 9 - np.sum(np.abs(self.initial))
                
        ## running values for each action:
        self.values = np.zeros(self.num_actions)
        
        self.turn = 1.0
        self.epsilon = np.finfo(np.float32).eps
        
    def update_turn(self):
        
        self.turn = 1.0*self.turn + -1.0*(1.0-self.turn)
        
    def evaluation(matrix):
    
        game_state = game_evaluation(matrix)
            
        return game_state.reward

    def matrix_generation(self,matrix):
            
        positions = np.where(matrix.flatten() == 0.0)[0]
        self.num_positions = len(positions)
        
        matrices = np.zeros((self.num_positions,3,3))
        
        for i in range(self.num_positions):
            delta = np.zeros(9)
            delta[positions[i]] = self.turn
            matrices[i] = matrix + delta.reshape((3,3))
                    
        return matrices
    
    def matrix_selection(self,matrices):
        
        N = len(matrices)
        rewards = np.zeros(N)
                
        for i in range(N):
            rewards[i] = self.evaluation(matrices[i])
            
        ## rewards are defined with respect to player/opponent:
        rewards = self.turn*rewards
            
        return matrices[np.argsort(rewards)][:self.num_select], self.turn*np.max(rewards)
        
    def simulation(self):

        ## player generates possible positions for each atomic action:
        matrices = self.matrix_generation(self.initial)
        
        for i in range(1,self.max_depth):
            
            ## after each iteration we have fewer options:
            self.num_select -= 1
                        
            for j in range(self.num_actions):
                
                ## evaluation phase:
                
                ## update value of each position
                self.values[j] = self.evaluation(matrices[j]) ## I should halt as soon as abs(R) is maximal
                
                ## opponent's turn(min phase):
                self.update_turn()
                self.num_select -= 1
                
                ## adversarial generation and selection:
                selection, R = self.matrix_selection(self.matrix_generation(matrices[j]))
                
                ## player's turn to select(max phase):
                self.update_turn()
                self.num_select -= 1
                
                ## favorable generation and selection:
                M, R = self.matrix_selection(self.matrix_generation(selection[0]))
                
                matrices[j] = M[0]