#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:11:24 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation as G

G = G()

class tic_tac_system:
    
    def __init__(self,G,model,opponent,epochs,depth,gamma):
        self.G = G
        self.model = model
        self.opponent = opponent
        self.epochs = epochs
        self.Z = np.zeros((3,3))
        
        ## add opponent's specific hyper-parameters:
        self.depth = depth ## the maximum search depth of the opponent
        self.gamma = gamma ## the rate at which the opponent discounts rewards
        
        ## track learning progress:
        self.iter = 0
        self.score = np.zeros(epochs)
        
        
    def restart(self):
        """
            Start with a board that has two pieces with variable initialisation.
        """
        ## start with an empty board:
        self.Z = np.zeros((3,3))
        
        ix = np.random.choice(9)
        self.Z[int(ix/3)][ix % 3] = 1.0
        
        P = self.opponent(self.G,-1.0*self.Z,self.depth,self.gamma)
        
        self.Z += -1.0*P.move()
        
    def rollouts(self,sess):
        """
            Collects a mini-batch of recent game experience and their outcomes:
                
                input: sess variable for tensorflow operations
                
                output: 
                    rollouts: state-action pairs
                    rewards: rewards associated with each state-action pair
        """
        
        
        rollouts = np.zeros((self.model.batch_size,9,18)) ## rollout array
        rewards = np.zeros((self.model.batch_size,9))
        
        N = 0 ## an iterator to keep track of running average of scores
        mu_score = 0.0
        
        for i in range(self.model.batch_size):
        
            count = 0
                        
            while self.G.X_score(self.Z) == 0.0:
                
                action = sess.run(self.model.action,feed_dict={self.model.state:self.Z.flatten().reshape((1,9))})
                
                self.Z += action.reshape((3,3))
                
                print(sess.run(self.model.dist.sample(), feed_dict={self.model.state:self.Z.flatten().reshape((1,9))}))
                
                ## update rollout:
                rollouts[i][count] = np.concatenate((self.Z.flatten(),action.reshape((9,))))
                count += 1
                
                ## reward check-pointing:
                q = self.G.X_score(self.Z)
                
                if q != 0.0:
                    rewards[i][:count] = np.geomspace(q/np.abs(q),q, num=count)
                    
                    ## update the average score:
                    mu_score = mu_score*(N/(N+1))+q/(N+1)
                    N += 1
                
                ## add the consequence of the opponent's move:
                player_2 = self.opponent(self.G,-1.0*self.Z,self.depth,self.gamma)
                self.Z += -1.0*player_2.move()
                
                ## reward check-pointing:
                q = self.G.X_score(self.Z)
                
                if q != 0.0:
                    ## discount the reward in a geometric manner:
                    rewards[i][:count] = np.geomspace(q/np.abs(q),q, num=count)
                    
                    ## update the average score:
                    mu_score = mu_score*(N/(N+1))+q/(N+1)
                    N += 1
                    
        ## restart the system:
        self.restart()
        
        ## keep track of average learning progress:           
        self.score[self.iter] = mu_score
        self.iter += 1
                    
        return rollouts, rewards
                
    def batch_update(self,sess):
        """
            Update on a mini batch of rollouts.
        """
        
        sess.run(self.model.zero_ops)
        
        batch, rewards = self.rollouts(sess)
                    
        for i in range(self.model.batch_size):
            
            for j in range(9):
                
                state_action = batch[i][j].reshape((1,18))
                state = batch[i][:,:9][j].reshape((1,9))
                R = rewards[i][j].reshape((1,1))
            
                train_feed = {self.model.state_action : state_action,self.model.state:state \
                              ,self.model.reward: R}
            
            #train_feed = {self.model.state_action : np.zeros((1,18)),self.model.state: np.zeros((1,9)) \
            #              ,self.model.reward: np.zeros((1,1))} 
            
            sess.run(self.model.accum_ops,feed_dict = train_feed)
                    
        sess.run(self.model.train_step)
            
    