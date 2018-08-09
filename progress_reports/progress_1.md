Accomplishments:
1. Using the maximum log probability and softmax with temperature to calculate probabilities appears
to perform strictly worse than simply using the softmax. 
2. Reducing the learning rate from 0.1 to 0.01 appears to stabilise learning and a learning rate schedule might be even better
   but this has not yet been tested.
3. Training on 60k games takes about 5.5 mins. 

Next steps:
1. Try using a recurrent policy which might be useful for simulating the future. 
2. Check that this produces comparable or better performance. 