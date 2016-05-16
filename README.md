# PhD Thesis II - Reproduce comparative environment to benchmark against
## Rockblocks so far
### Jan 2016: 
Located and downloaded the AMIA Datasets, understood the data from dataset A, converted it from microarray to .nat format with only intensities

### Feb 2016: 
1Mx1M was a huge thing for a conventional machine to compute 1M times, so thought of moving to cluster or cloud

### Mar 2016: 
???? Cluster was tried but configuration issues were a major hurdle in getting over, so moved to Internal GPU server

### Apr 2016: 
Needed to explore options and learn CUDA, and introduced dynamic programming to avoid recomputations of the base 1Mx1M matrix but 1Mx1Mxfloat32 was too much 
for the RAM/GPURAM as well so devised an algorithm to compute it by parts by splitting it into 32 parts (the smallest part which can be done within RAM limits)

### May 2016: 
Reverted to storing them in a compressed file after computing them byparts 
but Precomputed 1Mx1Mxfloat32 was too much of a space for the system, so decided to go for hybrid pipeline approach 
i.e. from each of the 32 parts of computed corr matrix, pick 100 most correlated vectors, considering an index only once (because after first merge operation, it will be useless anyway), in a hope to have found a big chunk of usable cache. we will have 3200 such entries already with their vector indeces, we will keep this array populated upon each new iteration over the full dataset.
