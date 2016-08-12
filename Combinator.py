from itertools import *
import random

#array is the var that is used to host current level vars that have been elevated
#targetDepth is basically the depth of recurssion that we have to goto
#currentDepth is the current depth that we are on
#others is the var that hosts other genes that have been elevated
def gencombinations(array, targetDepth, currentDepth, others):
    array2 = []
    theCombinations = combinations(array, 2)
    for aCombination in theCombinations:
        print "aCombination = ", aCombination;
        if True:
            memberStringArray = ""
            for aMember in aCombination:
                print "aMember = ", aMember
                memberStringArray = memberStringArray + str(aMember)
                others.append(aMember)
                array2.append(memberStringArray)
        
    if (currentDepth == targetDepth):
        array2.extend(others)
        return array2
    currentDepth = currentDepth + 1
    return gencombinations(array2, targetDepth, currentDepth, others)

""" Main function
    Read the dataset FUG
    
"""
array = []
#inserting the intial data
for i in range(0, 10): #TODO: This should be dataset Size from Settings. I still have a strong feeling that it should not be string
    array.append(str(i))
#arg1: indeces of the FUG matrix
#arg2: length of combinations, e.g. 10 his avg was 2.7
#arg3: currently we are starting from 0
#arg4: we start with 0 selected elements thus emtpy array
#also pass in the dataset to apply LDA from
array3 = gencombinations(array, 2, 0, [])