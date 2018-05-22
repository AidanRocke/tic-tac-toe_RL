## load everything:

import numpy as np
from simple_play import simple_play
from stochastic_play import stochastic_play

### there are four possible permutations where player A moves first
### and so we'll use a 2-d boolean vector(ex. [0,1]) to denote the
### combination of interest:
## 1. simple(player A) vs simple(player B)
## 2. stochastic(player A) vs stochastic(player B)
## 3. simple(player A) vs stochastic(player B)
## 4. stochastic(player A) vs simple(player B)

def game_simulation(player_combo,num_games,random_start,depth,gamma):

    outcomes = np.zeros(num_games)
    
    initial_conditions = []
    
    for i in range(num_games):
        
        game = 1.0
        
        Z = np.zeros((3,3))
        X, O = np.random.choice(np.arange(9),2,replace=False)
        Z[int(X/3)][X % 3] = 1.0
        
        if random_start == 1.0:
            ## the second player plays randomly:
            Z[int(O/3)][O % 3] = -1.0
            
        else:
            ## the second player doesn't play randomly:
            if player_combo[1] == 1.0:
                P2 = stochastic_play(-1.0*Z,depth,gamma)
                Z += -1.0*P2.move()
            else:   
                P2 = simple_play(-1.0*Z,depth,gamma)
                Z += -1.0*P2.move()
        
        initial_conditions.append(np.copy(Z))
    
        while game == 1.0:
            ## player A move:
            if player_combo[0] == 1.0:
                P1 = stochastic_play(Z,depth,gamma)
                Z += P1.move()
            else:  
                P1 = simple_play(Z,depth,gamma)
                Z += P1.move()
            
            if P1.score(Z)[1] != 0.0:
                outcomes[i] = P1.score(Z)[1]
                
                game = 0.0
                
                break
            
            ## player B move:
            if player_combo[1] == 1.0:
                P2 = stochastic_play(-1.0*Z,depth,gamma)
                Z += -1.0*P2.move()
            else:   
                P2 = simple_play(-1.0*Z,depth,gamma)
                Z += -1.0*P2.move()
            
    return initial_conditions, outcomes