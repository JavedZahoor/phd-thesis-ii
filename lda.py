print "Loading numpy"
import numpy as np
print "Loading LDA"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
print "Loading timer"
from timeit import default_timer as timer
print "Loading Random"
import random
print "Loading done"


# import scipy

# mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
# data=mat['G0'][:, 0:s.sampleSize()];
def fitness():
    return random.randint(0, 1)


def randomarr():
    temp = []
    for i in range(0, 1000):#001
        temp.append(random.random())
    return temp


print "generating data"
arr1 = []
arr2 = []
test = []
for i in range(0, 20):
    arr1.append(randomarr())
    arr2.append(fitness())
    print i
print "done generating"

X = np.array(arr1)
y = np.array(arr2)
del arr1
del arr2
print "done deleting arr1 arr2"
start = timer()
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
end = timer()

del X
del y

print "Time it took to train:"
print(end - start)
print "Time it took to Predict:"
test.append(randomarr())
test.append(randomarr())
test.append(randomarr())
test.append(randomarr())
start = timer()
print "class prediction";
print(clf.predict(test))
end = timer()
print(end - start)
print "decision function value"
#print test;
print (clf.decision_function(test))
