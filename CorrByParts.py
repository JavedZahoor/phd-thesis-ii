from GlobalUtils import * 
import numpy
from LoadDataSetA import LoadDataSet
from MachineSpecificSettings import Settings
s = Settings()
p=20
w=2;
G = numpy.matrix((88,p), dtype=float)
Result = numpy.zeros([p,p], dtype=float)
logDebug ("going to start the loops now")
LoadDataSet()
for i in range (0, p/w):
    print("i=")
    print(i)
    for j in range(i, p/w):
        print("j=")
        print(j)
        A = G[:, i*w:(i+1)*w]
        B = G[:, j*w:(j+1)*w]
        if(i==j):
            R1 = numpy.corrcoef(A, B)
            Result[i*w:(i+1)*w, j*w:(j+1)*w] = R1
        else:
            Result[j*w:(j+1)*w, i*w:(i+1)*w] = R1.transpose()