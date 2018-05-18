#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:33:13 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation

class perfect_play:
    
    def __init__(self,initial_matrix,max_depth,gamma):
        self.initial = initial_matrix
        self.gamma = gamma
        
        ## num_positions*num_select ~ max_breadth
        self.num_positions = 9 - np.sum(np.abs(self.initial))
        self.max_depth = int(np.min([max_depth,int(self.num_positions/2)]))

        self.num_actions = int(9 - np.sum(np.abs(self.initial)))
                
        ## running values for each action:
        self.values = np.zeros(self.num_actions)
        
        self.turn = 1.0
        
    def update_turn(self):
        
        self.turn *= -1.0
        
    def reward(self,matrix):
    
        game_state = game_evaluation(matrix)
        
        value = game_state.reward*(game_state.X_score == 0.0) + \
                100*game_state.X_score*(game_state.X_score != 0.0)
            
        return value

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
            rewards[i] = self.reward(matrices[i])
            
        ## rewards are defined with respect to player/opponent:
        #rewards = self.turn*rewards
            
        return matrices[np.argsort(-1.0*rewards)][:self.num_positions], self.turn*np.max(-1.0*rewards)
        
    def simulation(self):

        ## player generates possible positions for each atomic action:
        matrices = self.matrix_generation(self.initial)
        
        for i in range(self.max_depth):
                        
            for j in range(self.num_actions):
                                
                ## update value of each action
                #self.values[j] = self.gamma*self.values[j] + self.reward(matrices[j]) 
                
                ## opponent's turn(min phase):
                self.update_turn()
                
                if self.num_actions <= 1:
                    break
                
                ## adversarial generation and selection:
                selection, R = self.matrix_selection(self.matrix_generation(matrices[j]))
                
                ## update value of each action
                self.values[j] = self.gamma*self.values[j] + self.reward(selection[0]) 
                
                if abs(R) >= 50.0:
                    break
                
                ## player's turn to select(max phase):
                self.update_turn()
                
                ## favorable generation and selection:
                M, R = self.matrix_selection(self.matrix_generation(selection[0]))
                
                matrices[j] = M[0]
        
            ## update depth:
            self.max_depth = int(9 - np.sum(np.abs(matrices[0])))
            
    def move(self):
        
        positions = np.where(self.initial.flatten() == 0.0)[0]
        
        self.simulation()
        
        delta = np.zeros(9)
        delta[positions[np.argmax(self.values)]] = 1.0
            
        return delta.reshape((3,3))