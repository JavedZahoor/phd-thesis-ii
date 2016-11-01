import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#TODO: Citation http://scikit-learn.org/0.16/about.html#citing-scikit-learn
from timeit import default_timer as timer
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
from GlobalUtils import *

@timing
def J_Linear(r_normalized, error): #works best with Euclidean Metagenes
    sigma = 0.5; #sigma = [0,1], where 0 means use only error and 1 means use only reliability. I have set it to 0.5 for equal weightage to both
    return sigma * r_normalized + (1-sigma) * (1-error); #eq 4.3 of thesis

@timing
def J_Exp(r, error, sigma): #works better with correlation based Metagenes
    #sigma = penalization parameter
    sign = 1;
    if (r > 0):
        sign = -1;
    return r * np.exp(sign * e * 100 / sigma); #eq 4.2 of thesis

@timing
def reliability(classifier, testSet, overallSigma):
    n_test = np.shape(testSet)[0];
    #split the test dataset into two subsets 
    lastIndex = testSet.shape[1]-1;
    class1Data = testSet[numpy.where(testSet[:,lastIndex]==0)[0],:];
    size_class1 = class1Data.shape()[0];
    
    
    class2Data = testSet[numpy.where(testSet[:,lastIndex]==1)[0],:];
    size_class2 = class2Data.shape()[0];
    
    prob_class1 = size_class1/(size_class1 +  size_class2);
    prob_class2 = size_class2/(size_class1 +  size_class2);
    #loop through each partition and calculate sum of D_i/p(C_i)
    d1 = classifier.decision_function(class1Data);
    d1sum = d1.sum();
    d2 = classifier.decision_function(class2Data);
    d2sum = d2.sum();
    return (d1sum/(prob_class1*size_class1)+d2sum/(prob_class2*size_class2)) * 1/(n_test * overallSigma);

@timing
def OverallSigma(dataSet):
    lastIndex = numpy.shape(dataSet)[1]-1;
    class1Data = dataSet[numpy.where(dataSet[:,lastIndex]==0)[0],:];
    class2Data = dataSet[numpy.where(dataSet[:,lastIndex]==1)[0],:];
    class1Var=np.var(class1Data);
    class2Var=np.var(class2Data);
    return np.sqrt(class1Var/np.shape(class1Data)[0] + class2Var/np.shape(class2Data)[0]);

@timing    
def ErrorRate(TP, TN, FP, FN):
    return (FP+FN)/(TP+TN+FP+FN);

@timing
def Train(enhancedGeneSet, classLabels):
    enhancedGeneSet = np.array(enhancedGeneSet);
    classLabels = np.array(classLabels);
    classifier = LinearDiscriminantAnalysis();
    classifier.fit(enhancedGeneSet, classLabels);
    #del enhancedGeneSet;
    #del classLabels;
    return classifier;

@timing
def PredictClass(classifier, sample):
    return classifier.predict(sample);

@timing
def Evaluate(classifier, dataSet, testSet, testLabels, positiveLabel):
    testLabels = numpy.array(testLabels);
    predictions = classifier.predict(testSet);
    TP = TN = FP = FN = 0; 
    for i in range(0, testLabels.shape[0]):
        if testLabels[i]==positiveLabel:
            if predictions[i]==testLabels[i]:
                TP = TP + 1;
            else:
                FN = FN + 1;
        else:
            if predictions[i]==testLabels[i]:
                TN = TN + 1;
            else:
                FP = FP + 1;
    errorRate = ErrorRate(TP, TN, FP, FN);
    r = reliability(classifier, testSet, OverallSigma(dataSet));
    return [errorRate, r, J_Exp(r, errorRate, 0.5)];
    

""" MAIN CODE FOR TREELET CLUSTERING """
def main():
    datasetLoader = DataSetLoader();
    classLabels = []
    classLabels.append(datasetLoader.GetClassLabels("A"))
    
    enhancedGeneSet = []
    
    test = []
    enhancedGeneSet.append(datasetLoader.LoadDataSet("A"))#TODO:  Load Enhanced DataSet using DataSetLoaderLib
    
    test.append(randomarr())#TODO: This will be extracted from the DataSet Itself i.e. Exclusive 10% of the dataset for CV10



if __name__=="__main__":
    main();
