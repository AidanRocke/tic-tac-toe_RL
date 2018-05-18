#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:21:34 2018

@author: aidanrocke
"""

import numpy as np
from perfect_play import perfect_play

Q1 = np.zeros((3,3))

## first move
P = perfect_play(Q1,5,0.5)

P.move()

## second move:
Q2 = np.array([[1,0,0],[0,-1,0],[0,0,0]])

P = perfect_play(Q2,5,0.5)

P.move()

## third move:
Q3 = np.array([[1,1,-1],[0,-1,0],[0,0,0]])

P = perfect_play(Q3,5,0.5)

P.move()

## fourth move:
Q4 = np.array([[1,1,-1],[-1,-1,0],[1,0,0]])

P = perfect_play(Q4,5,0.5)

P.move()

## second game:

Q1 = np.array([[1,0,0],[0,-1,0],[0,0,0]])

## first move
P = perfect_play(Q1,5,0.5)

P.move()

## second move:
Q2 = np.array([[1,0,0],[1,-1,0],[-1,0,0]])

P = perfect_play(Q2,5,0.5)

P.move()

## third move:
Q3 = np.array([[1,-1,1],[1,-1,0],[-1,0,0]])

P = perfect_play(Q3,3,0.5)

P.move()


test_case_1 = np.array([[1,1,-1],[0,-1,0],[1,0,0]])
test_case_2 = np.array([[1,1,-1],[1,-1,0],[0,0,0]])