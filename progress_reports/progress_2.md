Accomplishments:
1. Using a convolutional lstm as a policy appears to lead to more stable learning. The win-rate hovers around 72%. 
2. It's not clear that I have properly implemented recurrent policy gradients.
3. Training on 60k games takes about 30 mins.  

Next steps:
1. Print the win-rate every 3k games i.e. every 100 iterations where each iteration involves a batch of size 30.
2. Refactor the code. 
3. Add unit tests. 