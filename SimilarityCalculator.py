from GlobalUtils import *
from scipy import sparse
#import numpy as np


import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

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
        #The following line was causing the system to give MEM Error due to 1Mx1Mxfloat32 matrix space requierments
        #Result = [Result, numpy.zeros((theMatrix.shape[1]/howmany, theMatrix.shape[1]/howmany), dtype=float)];
        #so now we will be saving the corr matrix by parts in different files and use files for processing
        ######Result = numpy.zeros((aPart, aPart), dtype=float);
        totalParts = theMatrix.shape[1] / aPart;
        hashtable = {};
        #http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.chararray.html
        #fileNames = numpy.chararray((totalParts, totalParts), itemsize=12);#each item will be 12 char long corr_00i_00j at max
        
        #worked till 32000 but to make it completely divisible i did 1004004/36=27889 there is a hard limit on 2^27 processing at a time http://stackoverflow.com/questions/13187443/nvidia-cufft-limit-on-sizes-and-batches-for-fft-with-scikits-cuda
        #so now i will need to calculate it using in a loop of 32 OR launch such 32 warps
        #GDash = theMatrixTranspose[0:aPart,:].tolist();
        """"""
        #outfile = NamedTemporaryFile(delete=False);#TemporaryFile();#will be used to save all the parts of the corr matrix
        #print(outfile.name);
        vectorCache = numpy.array([], dtype=[('i', int), ('j', int), ('corr', float)]);#(-1,-1,0.0)
        for i in range (0, totalParts-1):
            print("i="+str(i));
            for j in range(i, totalParts-1):
                print("j=" + str(j));
                A = theMatrixTranspose[i*aPart:(i+1)*aPart,:].tolist();
                ## Last part should not take full space if it is not required to do so
                B = theMatrixTranspose[j*aPart:(j+1)*aPart,:].tolist();
                print("A = " + str(len(A)));
                print("B = " + str(len(B)));
                Result = pearson_correlation(A, B)
                print('going to concatenate vectorCache with the main list');
                newlist, hashtable = self.ExtractTopCorrValues(Result, cacheTopXPerPart, aPart, hashtable, i, j);
                vectorCache = numpy.concatenate((vectorCache, newlist));
                print('concatenation done...');
                """
                rejection this because of mem limitations on the GPU RAM for the whole 1Mx1Mxfloat32 requirements
                #if(i==j):
                    #the input is a aPartx88 matrices
                    #H = pearson_correlation(GDash, GDash);
                    #the output is going to be a aPartxaPart matrix; with diagnoal = 1
                    #need to place it in the upper 0:aPart area
                    #for i in range (0, aPart):
                    #    print(H[i,i]);
                    #R1 = pearson_correlation(A, B)
                    #Result[i*aPart:(i+1)*aPart, j*aPart:(j+1)*aPart] = R1
                #else:
                    #Result[j*aPart:(j+1)*aPart, i*aPart:(i+1)*aPart] = R1.transpose();
                """
                
                """
                rejecting because of diskspace limitations on the server for the whole 1Mx1Mxfloat32 requirements
                #Generate file name to make it self explanatory and keep it in a matrix
                #fileNames[i,j] = "corr_"+str(i)+"_"+str(j);
                #print(fileNames[i,j]);
                #a = fileNames[i,j];
                #save this part in a file http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savez.html#numpy.savez
                #b = Result;
                #numpy.savez(outfile,a=a, b=b); commented temporarily to see if diskspace was the only issue.
                """
        #return sorted list
        print('going to sort...');
        sortedVectorCache = vectorCache.sort(axis=0, kind='quicksort', order='corr');
        print('Calculate Similarity done.');
        return sortedVectorCache;
    
    @timing
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
    
    @timing
    def ExtractTopCorrValues(self, thematrix, thingsToExtract, dim, hashtable, originalI, originalJ):
        """
        http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.sort.html
        """
        ignoreDiagonal = originalI == originalJ;
        #extractedList = numpy.array([], dtype=[('i', int), ('j', int), ('corr', float)]);
        extractedList=[]
        c=0;
        while len(extractedList) < thingsToExtract and c<dim*dim:
            print('before argmax ' + str(len(extractedList)) + ' ' + str(c))
            loc = numpy.argmax(thematrix);
            print('after argmax =' + str(loc) + " value=" +str(numpy.max(thematrix)));
            #find out indeces from this value and create a touple
            i= loc/dim + originalI * dim;
            j= loc%dim + originalJ * dim;
            locali = i;
            localj = j;#these are needed to remove this value later on in the last line of this loop
            print("thematrix["+str(i)+"]["+str(j)+"]="+str(thematrix[i][j]));
            i += originalI * dim;
            j += originalJ * dim;
            
            if ignoreDiagonal and locali==localj:
                continue;
            if thematrix[locali,localj]<=0: #if this is the best you can do then we dont need your services, thanks!
                return extractedList;
            #ensure atleast block level uniqueness of the vectors i.e. no vector should reappear ever
            
            if True!=hashtable.has_key(str(i)) and True!=hashtable.has_key(str(j)): #we need this one
                print "inside hastable wala if and value of c is = "+str(c)
                hashtable[str(i)]='y';
                hashtable[str(j)]='y';
                #need to stop when thingsToExtract values have been added to the list actually, not when they have been checked
                if c==0:
                    print('before extracted list');
                    #extractedList=numpy.array([(i, j, thematrix[i,j])], dtype=[('i', int), ('j', int), ('corr', float)]);
                    extractedList.append((i, j, thematrix[locali,localj]))
                    print('after extracted list: ');
                    print len(extractedList)
                else:
                    #http://stackoverflow.com/questions/9775297/append-a-numpy-array-to-a-numpy-array
                    print('before concatenate');
                    #extractedList = numpy.concatenate((extractedList,(i, j, thematrix[i,j])));
                    extractedList.append((i, j, thematrix[locali,localj]))
                    print len(extractedList)
                    print('after concatenate');
            elif hashtable.has_key(str(i))==True:
                print('*****Going to flush off the row now*****');
                for t in range(0, dim):
                    thematrix[locali, t]=-1;
                    thematrix[t, locali]=-1;
            elif hashtable.has_key(str(j))==True:
                print('|||||Going to flush off the row now|||||');
                for t in range(0, dim):
                    thematrix[t, localj]=-1;
                    thematrix[localj, t]=-1;
            #forget about this one so we could move ahead
            thematrix[locali,localj]=0;
            thematrix[localj,locali]=0;
            c+=1;
            print('========================================')
        return (extractedList, hashtable);