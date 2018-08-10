Accomplishments:
1. Assuming that the convolutional lstm policy can plan N steps ahead I decided to treat these N policies 
as an ensemble that I can average. 
2. Interestingly, by using this average instead of the last policy I get much more stable results. 
3. Within 30k games the convolutional lstm controller converges to a 82% win-rate. 
