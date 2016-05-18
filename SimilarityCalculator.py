from GlobalUtils import *
from scipy import sparse
#import numpy as np


import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from timeit import default_timer as timer
from MachineSpecificSettings import Settings

from math import sqrt
import numpy

from tempfile import *

from ref_1_distances import pearson_correlation
#Imported from https://github.com/vinigracindo/pycudaDistances/
#from ref_1_distances import *

class SimilarityCalculator(object):
    def CalculateSimilarity(self, theMatrix):
        return theMatrix;

class PairwisePearsonCorrelationCalculator(SimilarityCalculator):
    @timing
    def CalculateSimilarity(self, theMatrix, aPart, cacheTopXPerPart):
        theMatrixTranspose = theMatrix.transpose();#convert from 88x1M to 1Mx88 matrix
        #we will be saving the corr matrix by parts in different files and use files for processing
        totalParts = theMatrix.shape[1] / aPart;
        totalParts = totalParts if theMatrix.shape[1] % aPart ==0 else totalParts + 1;
        globalHash = {};
        settings = Settings();
        #worked till 32000 but to make it completely divisible i did 1004004/36=27889 there is a hard limit on 2^27 processing at a time http://stackoverflow.com/questions/13187443/nvidia-cufft-limit-on-sizes-and-batches-for-fft-with-scikits-cuda
        #will need to calculate it using in a loop of 32 OR launch such 32 warps
        
        vectorCache = [];#numpy.array([], dtype=[('i', int), ('j', int), ('corr', float)]);#(-1,-1,0.0)
        for i in range (0, totalParts-1):
            print("i="+str(i));
            for j in range(i, totalParts-1):
                print("j=" + str(j));
                A = theMatrixTranspose[i*aPart:(i+1)*aPart,:].tolist();
                ## Last part should not take full space if it is not required to do so
                B = theMatrixTranspose[j*aPart:(j+1)*aPart,:].tolist();
                print("A = " + str(len(A)));
                print("B = " + str(len(B)));
                if settings.isLocalMachine():
                    Result = A[:][0:1000];
                else:
                    Result = pearson_correlation(A, B)
                print('going to concatenate vectorCache with the main list');
                newlist, localHash = self.ExtractTopCorrValues(Result, cacheTopXPerPart, aPart, globalHash, i, j);
                globalHash.update(localHash);
                vectorCache.append(newlist);#http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
                print('concatenation done...');

        #return sorted list
        print('going to sort...');
        sortedVectorCache = vectorCache.sort(axis=0, kind='quicksort', order='corr');
        print('Calculate Similarity done.');
        return sortedVectorCache;
    
    @timing
    def UpdateSimilarity(self, theUpdatedMatrix, theVector, a, b):
        
        return theUpdatedMatrix;
    
    @timing
    def ExtractTopCorrValues(self, thematrix, thingsToExtract, dim, globalHash, blockI, blockJ):
        """
        This function is called on a smaller chunk of the vectors to calculate pairwise correlation and to save space, keep the relevant data only
        http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.sort.html
        
        
        Things we need to do in this function are
        1. Pickup the upper triangle from the block, excluding diagonal if it was AxA, else including diagnoal
        2. Sort the extracted data on descending order of corr values
        3. From this list, extract non-overlapping pairs based on the highest possible corr values for them - this is local maxima for each vector
        4. if any of the members of the selected pair is already chosen with higher corr value with some other partner, we are fine to ignore it
        5. otherwise we need to choose this as well and update the corr value in the global cache and mark this pair as superceeded
        6. in the caller, if a superceeded pair is found, we stop using cache there and restart this computation again
        """
        ignoreDiagonal = (blockI == blockJ);
        #extractedList = numpy.array([], dtype=[('i', int), ('j', int), ('corr', float)]);
        extractedList=[];
        upperTriangle = []; #numpy.array([], dtype=[('i', int), ('j', int), ('corr', float)]);
        c=0;
        #Extract upper triangle from the corr matrix
        #1. Pickup the upper triangle from the block, excluding diagonal if it was AxA, else including diagnoal
        print("Extracting upper triangle...");
        t0 = timer();
        for i in range(0, dim):
            print ":" + str(i) + ";",
            t1 = timer();
            start = i+1 if ignoreDiagonal else i;#http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator
            for j in range (start, dim):
                #extract only the upper triangle of the block with their local indeces and then sort                
                upperTriangle.append((dim * blockI + i, dim * blockJ + j, thematrix[i,j]));
            print(timer()-t1);
            
        #now we have the required elements only i.e. dim*dim/2 + dim (if diagonal is included)
        print("Extracting upper triangle... done in " + str(timer() - t0));
        
        #2. Sort the extracted data on descending order of corr values
        #now we can sort it once before filtering it #http://stackoverflow.com/questions/20183069/how-to-sort-multidimensional-array-by-column
        print("Sorting extracted list on corr value");
        t1 = timer();
        upperTriangle.sort(key=lambda x: x[2], reverse=True);
        print("done in " + str(timer()-t1));
        
        
        print("Going to perform the main logic of finding the most correlated vectors from sorted list...");
        localHash = {};
        #now lets filter it out the vectors based on their appearance with highest corr
        #every vector is allowed to appear exactly once in desc order of their corr value
        """
        3. From this list, extract non-overlapping pairs based on the highest possible corr values for them 
            - this is local maxima for each vector
        4. if any of the members of the selected pair is already chosen with higher corr value with some other partner, 
            we are fine to ignore it
        5. else we need to choose this as well and update the corr value in the global cache and mark this pair as superceeded
        """
        #globalHash[str(upperTriangle[c][0])][Fb][Corr]
        for c in range(0, len(upperTriangle)):#for each member
            if globalHash.has_key(str(upperTriangle[c][0])) or globalHash.has_key(str(upperTriangle[c][1])):
                print("Disruption Found: old " + str(globalHash[str(upperTriangle[c][0])]) + "; new " + str(upperTriangle[c]));
                #First check if it exists in the globalHash then check if we can still consider it
                if globalHash[str(upperTriangle[c][0])][1] < upperTriangle[c][2]:
                    #we just found a good candidate, but later in the loop
                    #release the other end of this vector
                    globalHash.pop(str(upperTriangle[c][1]));
                    #override the binding for this vecor
                    globalHash[str(upperTriangle[c][0])] = (upperTriangle[c][1], upperTriangle[c][2],'superceeded');					
                    #create the new second end
                    globalHash[str(upperTriangle[c][0])] = (upperTriangle[c][0], upperTriangle[c][2],'superceeded');
					extractedList.append((upperTriangle[c][0], upperTriangle[c][1], upperTriangle[c][2]));
                    break;#no use of going forward, we need recalculation after this point anyway
                elif globalHash[str(upperTriangle[c][1])][2] < upperTriangle[c][2]:
                    #we just found a good candidate, but later in the loop
                    #release the other end of this vector
                    globalHash.pop(str(upperTriangle[c][1]));
                    #override the binding for this vecor
                    globalHash[str(upperTriangle[c][1])] = (upperTriangle[c][1], upperTriangle[c][2],'superceeded');					
                    #create the new second end
                    globalHash[str(upperTriangle[c][1])] = (upperTriangle[c][0], upperTriangle[c][2],'superceeded');
					extractedList.append((upperTriangle[c][0], upperTriangle[c][1], upperTriangle[c][2]));
                    break;#no use of going forward, we need recalculation after this point anyway
                else:
                    print('never mind...');
                    continue;#this corr is less than the previous recorded corr for both the vectors
                
            else:#not already chosen globally, just go ahead with the normal logic
                """
                In this block, we dont need to check for bigger corr if something reappears, 
                we know the first comer was chosen based on higher corr value
                """
                if c%1000==0:
                    print str(c)+"/"+str(len(upperTriangle)),";";
                if localHash.has_key(str(upperTriangle[c][0])) or localHash.has_key(str(upperTriangle[c][1])): #ignore we already have higher local maxima
                    continue;#this one is definitely with lower corr value
                else:
                    #create unique candidate vectors from within this block i.e. local indexes only
                    localHash[str(upperTriangle[c][0])]=(upperTriangle[c][1], upperTriangle[c][2],'');#remember this vector's localIndex
                    localHash[str(upperTriangle[c][1])]=(upperTriangle[c][0], upperTriangle[c][2],'');#also remember this vector's localIndex
                    #add this vector's global indeces and their corr values to the extracted list
                    extractedList.append((upperTriangle[c][0], upperTriangle[c][1], upperTriangle[c][2]));
        
        
        return (extractedList, localHash);