# PhD Thesis II - Reproduce comparative environment to benchmark against
## Rockblocks so far
### Jan 2016: 
Located and downloaded the MAQC Datasets, understood the data from dataset A, converted it from microarray to .nat format with only intensities

### Feb 2016: 
1Mx1M was a huge thing for a conventional machine to compute 1M times, so thought of moving to cluster or cloud

### Mar 2016: 
???? Cluster was tried but configuration issues were a major hurdle in getting over, so moved to Internal GPU server

### Apr 2016: 
* Needed to explore options and learn CUDA, and introduced dynamic programming to avoid recomputations of the base 1Mx1M matrix but 1Mx1Mxfloat32 was too much for the RAM/GPURAM as well so devised an algorithm to compute it by parts by splitting it into 32 parts (the smallest part which can be done within RAM limits)

### May 2016: 
* Reverted to storing them in a compressed file after computing them byparts 
* but Precomputed 1Mx1Mxfloat32 was too much of a space for the system, so decided to go for hybrid pipeline approach 
* i.e. from each of the 32 parts of computed corr matrix, pick 100 most correlated vectors, considering an index only once (because after first merge operation, it will be useless anyway), in a hope to have found a big chunk of usable cache. we will have 3200 such entries already with their vector indeces, we will keep this array populated upon each new iteration over the full dataset.
* Reverted to caching approach, where top X highest correlation based values will be picked up from each of the computed blocks and stored in a cache to be used as first few iterations. The cache will need to be updated at each iteration of merging to remove the already used vectors and add back top Y from the computation of metagene with each of the blocks
* Doing argmax repeatedly was causing the system to slow down to death so redone the logic again as below
* From each computed block, pick up the upper triangular matrix i.e. (1M/32x1M/32)/2x3x4 bytes memory requirements per block, then sort once on correlation value, then pick top correlated unique vectors (if any one is used already, ignore that pair altogather). These will be definitive local maximas from each block. Also passing in global cache to see if the chosen vectors find a higher correlation elsewhere globally?
* Now the issue is with local vs global highest correlation; to find that out we need lots of memory again and the process is being killed by the OS; even if this is resolved it will become very slow again.
* Handled cross block corr conflicts through globalHash and globalindexes to refer to the vectors, throughout the calculation. Also added a concept of 'superceeded' selection to indicate that the cache has been messed up beyond this point.
