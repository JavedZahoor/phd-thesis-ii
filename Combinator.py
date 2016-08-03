from itertools import *
import random

#Pseudo fitness function which has a probablility of 50% we can tune it further down by utilizing those that are above
#A specific values for example if value is higher than 8 in a rand in range between 1 to 10 then only elevate
def fitness():
    return random.randint(0, 1)

#array is the var that is used to host current level vars that have been elevated
#targetDepth is basically the depth of recurssion that we have to goto
#currentDepth is the current depth that we are on
#others is the var that hosts other genes that have been elevated
def gencombinations(array, targetDepth, currentDepth, others):
    array2 = []
    x = combinations(array, 2)
    for i in x:
        if fitness() == 1:
            temp2 = ""
            for temp in i:
                temp2 = temp2 + str(temp)
                others.append(temp)
                array2.append(temp2)
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
for i in range(0, 10):
    array.append(str(i))
#arg1: indeces of the FUG matrix
#arg2: length of combinations, e.g. 10 his avg was 2.7
#arg3: currently we are starting from 0
#arg4: we start with 0 selected elements thus emtpy array
#also pass in the dataset to apply LDA from
array3 = gencombinations(array, 2, 0, [])
print len(array3)