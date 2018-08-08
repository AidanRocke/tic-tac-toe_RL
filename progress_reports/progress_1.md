Observations:
1. Using the maximum log probability and softmax with temperature to calculate probabilities appears
to perform strictly worse than simply using the softmax. 
2. Reducing the learning rate from 0.1 to 0.01 appears to stabilise learning and a learning rate schedule might be even better
   but this has not yet been tested. 