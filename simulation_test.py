#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:21:34 2018

@author: aidanrocke
"""

import numpy as np
from perfect_play import perfect_play

Q = np.array([[1,-1,-1],[-1,1,1],[0,1,-1]])

P = perfect_play(Q,5,10)

P.simulation()

P.values

