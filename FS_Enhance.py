from ref_2_pca import PCA
from ref_3_angle import findAngle
import scipy.io
import numpy
import math
import time
import json
#from matplotlib.mlab import PCA
#import traceback
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
from SimilarityCalculator import *
from GlobalUtils import *

#from GlobalUtils import *

"""FUNCTION generateNewMetaGene
	Use Local PCA i.e.
	for the two features, calculate local PCA, then calculate Jacobi rotation and then calculate (coarse grain) m = Fa cos ThetaL + Fb sin ThetaL 
	and fine grained d = fa cos ThetaL - Fb sin ThetaL
	
	we need ThetaL for this purpose; which is the angle between the first two components of the PCA result
"""
@timing
def generateNewMetaGene(Fa, Fb):
	pca = PCA(numpy.column_stack((Fa, Fb)));
	logDebug("pca = " + str(pca));
	# need to use either .a or .Y from this object; a is the original but normalized matrix while Y is the transformed one.	
	ThetaL = findAngle(pca[0][0], pca[0][1]); #find the angle of rotation between the two transformed vectors
	# for now we will just use m as new meta gene
	m = Fa * math.cos (ThetaL) +  Fb * math.sin(ThetaL)
	#d = Fa * math.cos (ThetaL) -  Fb * math.sin(ThetaL)	
	return m;
"""End of generateNewMetaGene"""	

"""Function performTreeletClustering"""
@timing
def performTreeletClustering(DatasetName):
	#% manually calculating correlation
	d = DataSetLoader();
	G = d.LoadDataSet(DatasetName);
	F = G;
	M = [];
	cacheTopXPerPart = d.CacheTopXPerPart(DatasetName);
	#calculate pairwise pearson correlation once which we will keep on changing with each new iteration
	corrCalculator = PairwisePearsonCorrelationCalculator();
	corrMatrix = corrCalculator.CalculateSimilarity(G, d.GetPartSize(DatasetName), cacheTopXPerPart);
	logInfo("starting off with " + str(len(corrMatrix)) );
	p = F[0,:].size
	for i in range (0, p):
		#steps 1 & 2 of the fig.1 of 20160224 - find pair wise correlation of F and pick the two most correlated columns
		theVectors = corrMatrix[0];#this is always the max corr so the element we want to process
		logDebug(theVectors);
		#logDebug (theVectors);
		Fa = F[:,theVectors[0]];
		logDebug("fa = " + str(Fa));
		Fb = F[:,theVectors[1]];		
		logInfo(" find a pair of most correlated vectors " + str(i));		
		#generate a new meta gene using the two picked up columns
		m = generateNewMetaGene(Fa, Fb);
		logDebug("m = " + str(m));
		logDebug(m[0]);
		logDebug(m[1]);
		#print(" generate one new gene ");
		#delete the two selected columns from the F and add the newly generate m to F; scipy.delete(F, 0 based index of col, 0=row and 1=col)
		# REUSE THIS PLACE FOR m F = scipy.delete(F, theVectors[0], 1)		
		logDebug("Before delete Fa & Fb " + str(F[0,:].size));
		F = scipy.delete(F, theVectors[1], 1)
		logDebug("After delete Fa & Fb " + str(F[0,:].size));
		#F = numpy.column_stack((m, F)) #include in the main feature set
		
		if not len(M): #if this is the first meta gene in this matrix
			M = m;
		else:
			M = numpy.column_stack((m, M)) #include in the meta genes set as well
			logDebug("M = "+str(M));
		logInfo(" append meta gene ");
		##remove the tuple at corrPointer and keep the pointer at 0 or increment the pointer and adjust with the update call
		corrMatrix.pop(0);
		#corrPointer = corrPointer+1;
		#now update the corrMatrix for this new vector m; remove the two vectors and related values, use one of them for m
		corrMatrix = corrCalculator.UpdateSimilarity(corrMatrix, F, list(m), theVectors[0], theVectors[1]);
		#print(F[theVectors[0]]);
		#print(m);
		F[:,theVectors[0]]=m;
		logDebug("theVectors = " + str(theVectors));
		logWarning('theVectors MUST INCLUDE superceeded as well, it is 3-dim so far')#theVectors[3]=="superceeded" or
		if len(corrMatrix)<=0: #everything after this is potentially incorrect so lets recalculate the matrix
			corrMatrix = corrCalculator.CalculateSimilarity(F, d.GetPartSize(DatasetName), cacheTopXPerPart);
			
		
	F = numpy.column_stack((G, M)) #scipy.append(G, M, 1) #define a new expanded featureset F = G U M	
	logInfo(" generate treelet clustering ")
	return F
"""END OF FUNCTION performTreeletClustering"""



""" MAIN CODE FOR TREELET CLUSTERING """
def main():
	F = performTreeletClustering("A");
	json.dump(F, open('treelet.json','w'));
	numpy.save('treelet.npy', F)	
	with open("log.txt", "a") as logfile:
		logfile.write("--- %s completed the treelet clustering " + tag + "---" % (time.time() - start_time));
		logfile.close();


if __name__=="__main__":
	main();
	
""" REFERENCES

Very Good GPU Tutorial: http://people.duke.edu/~ccc14/sta-663/CUDAPython.html

https://documen.tician.de/pycuda/metaprog.html#metaprog

PCA
http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python
[sarim]http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
** [sarim]http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
[sarim]http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""