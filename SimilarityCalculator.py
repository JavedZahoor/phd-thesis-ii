from scipy import sparse
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from math import sqrt
import numpy
from ref_1_distances import pearson_correlation

#Imported from https://github.com/vinigracindo/pycudaDistances/
from ref_1_distances import *

class SimilarityCalculator(object):
    def CalculateSimilarity(self, theMatrix):
        return theMatrix;

class PairwisePearsonCorrelationCalculator(SimilarityCalculator):
    def CalculateSimilarity(self, theMatrix):
        howmany=16
        Result=[];
        for t in range(howmany):
            print("t=")
            print(t)
            Result = [Result, numpy.zeros((theMatrix.shape[1]/howmany, theMatrix.shape[1]/howmany), dtype=float)];
        print(len(Result));
        theMatrixTranspose = theMatrix.transpose();#convert from 88x1M to 1Mx88 matrix
        aPart = 27889;
        totalParts = theMatrix.shape[1] / aPart;
        ##worked till 32000 but to make it completely divisible i did 1004004/36=27889 there is a hard limit on 2^27 processing at a time http://stackoverflow.com/questions/13187443/nvidia-cufft-limit-on-sizes-and-batches-for-fft-with-scikits-cuda
        #so now i will need to calculate it using in a loop of 32 OR launch such 32 warps
        #GDash = theMatrixTranspose[0:aPart,:].tolist();
        """"""
        for i in range (0, totalParts):
            print("i=")
            print(i)
            for j in range(i, totalParts):
                print("j=")
                print(j)
                ##I think this needs to be done for rows now as we have taken a transpose above
                A = theMatrixTranspose[i*aPart:(i+1)*aPart,:].tolist();
                B = theMatrixTranspose[j*aPart:(j+1)*aPart,:].tolist();
                if(i==j):
                    #the input is a aPartx88 matrices
                    #H = pearson_correlation(GDash, GDash);
                    #the output is going to be a aPartxaPart matrix; with diagnoal = 1
                    #need to place it in the upper 0:aPart area
                    #for i in range (0, aPart):
                    #    print(H[i,i]);
                    R1 = pearson_correlation(A, B)
                    Result[i*aPart:(i+1)*aPart, j*aPart:(j+1)*aPart] = R1
                else:
                    Result[j*aPart:(j+1)*aPart, i*aPart:(i+1)*aPart] = R1.transpose()
        """"""
        return Result;
    def UpdateSimilarity(self, theUpdatedMatrix, theVector, a, b):
        #remove the two indexes a and b from both the rows and cols of the corr matrix
        ##These rows and columns should be removed altogether OR a should be used for the new meta gene as well to keep things intact
        ##Also theMatrix should now be the updated theMatrix after adding the new meta gene?
        blank = numpy.zeros((corrMatrix.shape[0], theUpdatedMatrix.shape[1]),dtype=float);
        corrMatrix[a, :] = blank;
        corrMatrix[:, a] = blank.transpose();
        corrMatrix[b, :] = blank;
        corrMatrix[:, b] = blank.transpose();
        ##now use index a to fill in the pair-wise correlation of the gene m with all the remaining vectors except a & b???? think over it
        R1 = pearson_correlation(theUpdatedMatrix, theVector);
        corrMatrix[a, :] = R1;
        corrMatrix[:, a] = R1.transpose();
        return corrMatrix;