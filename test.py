#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:17:22 2018

@author: aidanrocke
"""

import numpy as np
from evaluation import game_evaluation

## define random seed:
np.random.seed(32)

X1, Y1, Z1 = np.random.choice([0,-1.0,1.0],(3,3)),np.random.choice([0,-1.0,1.0],(3,3)), np.random.choice([0,-1.0,1.0],(3,3)) 

X2, Y2, Z2 = np.ones((3,3)), np.identity(3), -1.0*np.identity(3)

Q = np.array([[1,-1,-1],[-1,1,1],[0,1,-1]])

## evaluating X1:
game_state = game_evaluation(X1)

game_state.draw ## no draw

game_state.X_score ## no win/loss

## evaluating Y1:
game_state = game_evaluation(Y1)

game_state.draw ## no draw

game_state.X_score ## no win/loss

## evaluating Q:
game_state = game_evaluation(Q)

game_state.draw ## yes, there's a draw

game_state.X_score ## no win/loss

## evaluate transpose of Q:

game_state = game_evaluation(Q.transpose())

game_state.draw ## yes, there's a draw

game_state.X_score ## no win/loss

## evaluating X2:
game_state = game_evaluation(X2)

game_state.draw ## no draw

game_state.X_score ## win

## evaluating Y2:
game_state = game_evaluation(Y2)

game_state.draw ## no draw

game_state.X_score ## win

## evaluating Z2:
game_state = game_evaluation(Z2)

game_state.draw ## no draw

game_state.X_score ## loss


