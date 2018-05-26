#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:36:01 2018

@author: aidanrocke
"""
import numpy as np
from tic_tac_learning import simulator
from tic_tac_pg import policy_gradients
from clever_stochastician import clever_stochastician as opponent
from tic_tac_system import tic_tac_system
from evaluation import game_evaluation as G

G = G()

model = policy_gradients(42,40)

tic_tac = tic_tac_system(G,model,opponent,600,5,0.9)

scores = simulator(tic_tac)

for i in range(5):
    
    print(np.mean((scores[i*100:(i+1)*100]+5.0)/10))


