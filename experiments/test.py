#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:36:01 2018

@author: aidanrocke
"""

import tensorflow as tf
import numpy as np


from clever_stochastician import clever_stochastician as opponent
model = policy_gradients(42,10)

tic_tac = tic_tac_system(G,model,opponent,100,5,0.9)

scores = simulator(tic_tac)

