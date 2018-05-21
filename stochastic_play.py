#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:15:02 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation

class stochastic_play:
    
    def __init__(self,initial_matrix,max_depth,gamma):
        self.initial = initial_matrix
        self.gamma = gamma
        self.R = np.array([[0,0,1],[0,1,0],[1,0,0]])
        self.state = 0.0
        
        ## num_positions*num_select ~ max_breadth
        self.num_positions = int(9 - np.sum(np.abs(self.initial)))
        self.max_depth = int(np.min([max_depth,int(self.num_positions/2)]))

        self.num_actions = int(9 - np.sum(np.abs(self.initial)))
                
        ## running values for each action:
        self.values = np.zeros(self.num_actions)
        
        self.turn = 1.0
        
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
    
        return matrices[np.argsort(-1.0*rewards)][:self.num_positions], self.turn*np.max(-1.0*rewards)

    def matrix_evolution(self,batch,batch_indices):
                
        ix, B = [], []
                    
        N = len(batch_indices)
        
        batch_values = np.array([self.turn*self.reward(batch[i]) for i in range(N)])
        mu_batch  = np.mean(batch_values)
    
        sub_index = np.where(batch_values > mu_batch)[0]
    
        rewards, counts = np.zeros(self.num_actions), np.zeros(self.num_actions)
        
        if len(sub_index) > 2:
    
            for i in sub_index:
                    M, R = self.matrix_selection(self.matrix_generation(batch[i]))
                                
                    ## calculate index:
                    j = int(batch_indices[i])
                
                    ## choose the top 3:
                    B.extend(M[:3])
                    ix += [j]*len(M[:3])
                        
                    ## update rewards using running average:
                    rewards[j] = rewards[j]*(counts[j]/(counts[j]+1))+(self.turn*R)/(counts[j]+1)
                    counts[j] += 1
                        
        ## update the value of each action:
        for k in range(self.num_actions):
            self.values[k] = self.gamma*self.values[k] + rewards[k]
            
            if abs(rewards[k]) >= 50.0: ## halt the game if the player has won/lost/drawn. 
                self.state = 1.0
                break
    
        return np.array(B), np.array(ix)
                            
    def simulation(self):
        
        ## player generates possible positions for each atomic action:
        matrices = self.matrix_generation(self.initial)
        
        batch = np.array([self.matrix_generation(matrices[i]) for i in range(len(matrices))])
        
        batch_shape = np.shape(batch)
        
        batch_indices = np.array([np.ones(batch_shape[1])*i for i in range(batch_shape[0])]).flatten()
        
        batch = batch.reshape((batch_shape[0]*batch_shape[1],3,3))
                
        while self.max_depth >= 1:
            
            if len(batch_indices) <= 1 or self.state == 1.0:
                break
            
            ## switch turn for min_phase:
            self.update_turn()
            batch, batch_indices = self.matrix_evolution(batch,batch_indices)
                        
            if len(batch_indices) <= 1 or self.state == 1.0:
                break
            
            ## switch turn for max phase:
            self.update_turn()
            batch, batch_indices = self.matrix_evolution(batch,batch_indices)
        
            ## update depth:
            self.max_depth = int(np.min([self.max_depth,int(self.num_positions/2)]))
            
    def move(self):
        
        positions = np.where(self.initial.flatten() == 0.0)[0]
        
        self.simulation()
        
        delta = np.zeros(9)
        delta[positions[np.argmax(self.values)]] = 1.0
            
        return delta.reshape((3,3))