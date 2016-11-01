"""
http://sebastianraschka.com/Articles/2014_python_lda.html
Load the Enhanced Dataset, G = F U M, Expected to have size 88x(1004004+(1004004/2)) = 88x(1004004+502002) = 88x1506006
Feature Selection:
    Step 1: Generate all possible subsets of indexes; 
        heuristics:
            0. Limit around the reported length of subsets i.e. less than 1000 or even below.
            1. use metagenes first
            2. know which genes are used to generate a meta gene so we dont try that combination
            3. also use biological information first
            4. Try different other FS techniques here e.g. mRMR, Correlation as heuristics
        Training DataSet:
            1. Split the dataset of 88 patients into Training+Validation and Testing Dataset
            2. Use LOOCV i.e. pick n-1 for training and 1 for validation
    Step 2: Apply LDA to each subset, calculate the error rate and reliability factor for each subset
    Step 3: Apply thining algo to setup the ensemble of classifiers
    Step 4: Save Feature Selector Ensemble Configurations
"""

import numpy
from DataSetLoaderLib import DataSetLoader
from itertools import *
from Supervised_LDA import *
import random
#from Combinator import *
from GlobalUtils import *

def getNextCombination():
    logWarning('HARDCODED COMBINATION');
    return [1,4,7,34,2,6,8,900];
""" MAIN CODE FOR TREELET CLUSTERING """
def main():
    """
        for each of the subsets s of indexes from 0-1004003 of length between 1 and 10
        #use biological info of known genes and mutual information and top down level of tree nodes [top down means ok interpretation and vague idea of root cause. bottom up means poor interpretation but pin pointed root cause identification]
        #check which subset is the best one by sorting them on desc order of error and then reliability
            create d as vertical projection of dataset using s indexes only
            for partition = 1 to length-2
                create trainingSet of size partition
                create testSet of size length-partition
                calculate Error Rate & Reliability using CV10
                calculate avg Error Rate and Avg Reliability 
        pick the best
    """
    print("\n\n\n\n\n");
    datasetLoader = DataSetLoader();
    setSize = 3;
    CVSetting = 2;
    classLabels = [];
    enhancedGeneSet = [];
    classLabels.extend(datasetLoader.GetClassLabels("A"));
    enhancedGeneSet.extend(datasetLoader.LoadEnhancedDataSet("A"));
    enhancedGeneSet = np.array(enhancedGeneSet);
    logInfo('Loaded the datasets');
    
    for s in range(1, 1+setSize):
        for i in range(0, np.array(enhancedGeneSet).shape[1]): 
            allCombinations = combinations(range(1,1+enhancedGeneSet.shape[1]-1),s);#TODO: go from 1 to setSize and for the selected top X from amongst one level, make sure the next level subset contains them as prefix so we 
            logInfo("allCombinations generated...")
            for aCombination in allCombinations:
                logDebug('aCombination');
                logDebug(aCombination);
                #on this combination, perform LOOCV (Leave one out cross validation)
                tempDataSet = enhancedGeneSet[:, aCombination[:]];
                
                logDebug ('temp Data Set ');
                logDebug (tempDataSet.shape);

                logInfo('going to partition the tempDataSet');
                logDebug(tempDataSet.shape[0]);
                
                for partition in range(CVSetting, 1+tempDataSet.shape[0]): #Using CV1
                    logDebug("Partition");
                    logDebug(partition);
                    
                    trainingLabels = classLabels[0:partition];
                    trainingSet = tempDataSet[0:partition, :];
                    
                    logDebug('training set');
                    logDebug(trainingSet.shape);
                    
                    testSet = tempDataSet[partition:tempDataSet.shape[0], :];
                    testLabels = classLabels[partition:tempDataSet.shape[0]];
                    
                    logDebug('test set');
                    logDebug(testSet.shape);
                    print (trainingLabels);
                    print (trainingSet);
                    classifier = Train(trainingSet, trainingLabels);
                    errorRate, reliability, jScore = Evaluate(classifier, tempDataSet, testSet, testLabels, 1);
                    
                    print errorRate, ";", reliability, ";", jScore;
                    return;

if __name__=="__main__":
    main();
