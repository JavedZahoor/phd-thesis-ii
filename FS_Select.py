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
from DataSetLoaderLib import EnhancedDataSetLoader

""" MAIN CODE FOR TREELET CLUSTERING """
def main():
    enhancedDSLoader = new EnhancedDataSetLoader();
    enhancedDSLoader.LoadEnhancedDataset("A");


if __name__=="__main__":
    main();
