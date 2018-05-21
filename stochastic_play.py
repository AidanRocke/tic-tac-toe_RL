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
    
    def matrix_evolution(self,batch,batch_indices):
            
            ix, B = [], []
            
            N = len(batch_indices)
            
            mu_batch  = np.median([self.turn*self.reward(batch[i]) for i in range(N)])
            
            rewards = np.zeros(self.num_actions)
            
            ## sort values:
            order = self.values.argsort()
            ranks = order.argsort()

            
            for i in range(self.num_actions): 
                
                #prob = np.max([(self.values[i]*(self.values[i]*self.turn > 0))/(np.sum(self.values*(self.values*self.turn > 0))+1e-20),0.5])
                #prob = np.max([(np.exp(self.values[i]))/(np.sum(np.exp(self.values))+1e-20),0.5])
                prob = np.max([ranks[i]/self.num_actions,0.2])
                
                ## running average reward counter:
                count = 0
                
                indices = np.where(batch_indices==i)[0]
                
                if len(indices) > 10 and len(ix) < 150:
                
                    for k in np.random.choice(indices,int(len(indices)*prob[i]),replace=False):
                        ## adversarial matrix generation:
                        M = self.matrix_generation(batch[i])
                                                
                        for m in M:
                            
                            if self.turn*self.reward(m) > mu_batch or np.random.rand() > 0.1:
                                
                                B.append(m)
                                ix.append(i)
                                
                                ## update rewards using running average:
                                rewards[i] = rewards[i]*(count/(count+1))+ \
                                            (self.turn*self.reward(m))/(count+1)
                                
            ## update the value of each action:
            for i in range(self.num_actions):
                self.values[i] = self.gamma*self.values[i] + rewards[i]
                
                if abs(rewards[i]) >= 50.0: ## halt the game if the player has won/lost/drawn. 
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
            
            if len(batch_indices) <= 1:
                break
            
            ## switch turn for min_phase:
            self.update_turn()
            batch, batch_indices = self.matrix_evolution(batch,batch_indices)
            
            if len(batch_indices) <= 1:
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