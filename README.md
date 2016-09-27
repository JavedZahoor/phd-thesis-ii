# PhD Thesis III - Research Experimentation
## Roadblocks so far
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

### Jun/Jul 2016: 
* The overall execution of this step would take more than 24 hours (not sure what exactly, it never completed), the server would kill the process after a couple of hours just saying "Killed". Research into it revealed that the process was taking upto 97% of RAM and eventhough the server was dedicated to me, the OS would decide after sometime to suspect and kill the process just because it is taking too much resources.
* Tuned the Part Size and Top X cache etc. such that now it takes more iterations to complete but the RAM has reduced down to less than 50% and atleast the meta-gene-generation-process can complete (in upto 2 days).
* Since the server would not be up always and since the OS taught me a good lesson, decided to incorporate partial-state-save/resume-on-restart functionality so even if the server crashes uninformed, we can atleast resume. After all no-one can just sit there for days watch the process to complete and we need to run the process several times for sure.
* While trying to implement Step 2 of Feature Selection using LDA, realized that the dataset (the .cel files i am using) doesnt have any clue/information about the class/label of the sample. Tried multiple times to read the MAQC-II paper again and again without any success about where to find the class/label information.

### Aug 2016:
* Finally found the class/label information using the GEO2R tool provided for online data analysis, which definitely means the information is available somewhere else, which when used through this utility gets populated otherwise it is not directly evident where to find it. Anyway found the required class info from http://www.ncbi.nlm.nih.gov/geo/geo2r?acc=GSE24061. This can be opened from the individual dataset links from the main topic.
* So finally started putting together the bits and pieces for Feature Selection step!

### Sep 2016:
* Running and Rerunning both the steps for troubleshooting and validation

### Sep 2016 (Feedback Received and Pivoting):
* No need to exactly replicate the environment first. Just start pivoting.
* Validate your Algo by using a smaller dataset and using library vs your program to compare the results
* Dont go for brute force as per the paper being implemented, for such data random-walk and approximation is okay. So probably implement mRMR or some other technique as a preprocessing step and perhaps then do Treelet Clustering
* Generate an ensemble already by generating smaller datasets with random walk, then use this ensemble for classification with different boosting techniques
* Found https://github.com/nlhepler/mrmr as mrmr lib for python and going to implement it as preprocessing step.
