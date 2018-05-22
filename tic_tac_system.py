#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:11:24 2018

@author: aidanrocke
"""

import numpy as np

class tic_tac_system:
    
    def __init__(self,model,opponent,epochs,depth,gamma):
        self.model = model
        self.opponent = opponent
        self.epochs = epochs
        self.Z = np.zeros((3,3))
        
        ## add opponent's specific hyper-parameters:
        self.depth ## the maximum search depth of the opponent
        self.gamma ## the rate at which the opponent discounts rewards
        
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
        
        P = self.opponent(-1.0*self.Z,self.depth,self.gamma)
        
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
            
            while self.opponent.score(self.Z)[1] == 0.0:
                
                action = sess.run(self.model.action,feed_dict={self.state:self.Z.flatten().reshape((1,9))})
                
                self.Z += action.reshape((3,3))
                
                ## update rollout:
                rollouts[i][count] = np.concatenate((self.Z.flatten(),action))
                count += 1
                
                ## reward check-pointing:
                q = self.opponent.score(self.Z)[1]
                
                if q != 0.0:
                    rewards[i][:count] = np.geomspace(q/np.abs(q),q, num=count)
                    
                    ## update the average score:
                    mu_score = mu_score*(N/(N+1))+q/(N+1)
                    N += 1
                
                ## add the consequence of the opponent's move:
                player_2 = self.opponent(-1.0*self.Z,self.depth,self.gamma)
                self.Z += -1.0*player_2.move()
                
                ## reward check-pointing:
                q = self.opponent.score(self.Z)[1]
                
                if q != 0.0:
                    ## discount the reward in a geometric manner:
                    rewards[i][:count] = np.geomspace(q/np.abs(q),q, num=count)
                    
                    ## update the average score:
                    mu_score = mu_score*(N/(N+1))+q/(N+1)
                    N += 1
        
        ## keep track of average learning progress:           
        self.score[self.iter] = mu_score
        self.iter += 1
                    
        return rollouts, rewards
                
    def batch_update(self,sess):
        """
            Update on a mini batch of rollouts.
        """
        
        sess.run(self.model.zero_ops)
        
        rollouts, rewards = self.rollout(sess)
                    
        for i in range(self.model.batch_size):
            
            train_feed = {self.model.state_action : rollouts[i],self.model.state:rollouts[i][:,:9] \
                          ,self.reward: rewards[i].reshape((1,9))}
            
            sess.run(self.model.accum_ops,feed_dict = train_feed)
                    
        sess.run(self.model.train_step)
            
    