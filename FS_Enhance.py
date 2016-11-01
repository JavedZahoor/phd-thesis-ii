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
#For saving and restoring states
import pickle
import os.path

@timing
def checkCorr(corrMatrix,p):
	for i in corrMatrix:
		if i[0]>p:
			print i[0]
			print "ERROR"
			return 0
		if i[1]>p:
			print i[1]
			print "ERROR"
			return 0

@timing
def save_F(F):
 with open('F.pickle', 'w') as f:  
        pickle.dump([F], f)

#used for loading data of the variables
@timing
def read():
	try:
		with open('objs.pickle') as f:  
        		return pickle.load(f)
	except:
		print "main objs file is corrupt i will now load from backup"
		with open('backup.pickle') as f:  
        		return pickle.load(f)		
#use this function in your usual code and call it when you want to save
@timing
def save(G,F,M,i,corrMatrix,cacheTopXPerPart):
    with open('objs.pickle', 'w') as f:  
        pickle.dump([G,F,M,i,corrMatrix,cacheTopXPerPart], f)
    with open('backup.pickle', 'w') as f:  
        pickle.dump([G,F,M,i,corrMatrix,cacheTopXPerPart], f)

"""FUNCTION generateNewMetaGene
	Use Local PCA i.e.
	for the two features, calculate local PCA, then calculate Jacobi rotation and then calculate (coarse grain) m = Fa cos ThetaL + Fb sin ThetaL 
	and fine grained d = fa cos ThetaL - Fb sin ThetaL
	
	we need ThetaL for this purpose; which is the angle between the first two components of the PCA result
"""
@timing
def generateNewMetaGene(Fa, Fb):
	pca = PCA(numpy.column_stack((Fa, Fb)));
	#logDebug("pca = " + str(pca));
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
	saveFreq=1000
	#temp value for i
	x=-1
	if (not(os.path.isfile("objs.pickle"))):
		print "New Start"
		d = DataSetLoader();
		G = d.LoadDataSet(DatasetName);
		F = G;
		M = [];
		cacheTopXPerPart = d.CacheTopXPerPart(DatasetName);
		corrCalculator = PairwisePearsonCorrelationCalculator();
		print "calling corr calculator"
		corrMatrix = corrCalculator.CalculateSimilarity(G, d.GetPartSize(DatasetName), cacheTopXPerPart);
		
	else:
		print "continuing from where we left off"
		d = DataSetLoader();
		G,F,M,x,corrMatrix,cacheTopXPerPart=read()
		corrCalculator = PairwisePearsonCorrelationCalculator();
	p = F[0,:].size
	#because we have already done the previous iteration and loaded that one
	i=x+1
	
	while i<p:
	#for i in range (x+1, p):
		if checkCorr(corrMatrix,p)==0:
			print "ERROR IN CORRMATRIX INDEX"
			return 0;
		#calculating value of p
		p = F[0,:].size
		print "Value of i is : "+str(i)+" out of "+str(p)
		theVectors = corrMatrix[0];#this is always the max corr so the element we want to process
		if(corrMatrix[0][3]=='')
			recalc = False;
		else
			recalc = True;
		Fa = F[:,theVectors[0]];
		Fb = F[:,theVectors[1]];
		print "calling generate metagene"		
		m = generateNewMetaGene(Fa, Fb);
		print "calling scipy delete on F"
		F = scipy.delete(F, theVectors[1], 1)
		if not len(M): #if this is the first meta gene in this matrix
			M = m;
		else:
			M = numpy.column_stack((m, M)) #include in the meta genes set as well
		corrMatrix.pop(0);		
		corrMatrix = corrCalculator.UpdateSimilarity(corrMatrix, F, list(m), theVectors[0], theVectors[1]);
		F[:,theVectors[0]]=m;
		if len(corrMatrix)<=0 or recalc==True: #everything after this is potentially incorrect so lets recalculate the matrix
			corrMatrix = corrCalculator.CalculateSimilarity(F, d.GetPartSize(DatasetName), cacheTopXPerPart);
		if i % saveFreq==0:
			save(G,F,M,i,corrMatrix,cacheTopXPerPart)
		i+=1
	F = numpy.column_stack((G, M)) #scipy.append(G, M, 1) #define a new expanded featureset F = G U M	
	return F

#END OF FUNCTION performTreeletClustering"""



# MAIN CODE FOR TREELET CLUSTERING """
def main():
	F = performTreeletClustering("A");
	save_F(F);
	#json.dump(F, open('treelet.json','w'));
	#numpy.save('treelet.npy', F)	
	#with open("log.txt", "a") as logfile:
		#logfile.write("--- %s completed the treelet clustering " + tag + "---" % (time.time() - start_time));
		#logfile.close();


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
