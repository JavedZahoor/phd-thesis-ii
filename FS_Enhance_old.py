import scipy.io
import numpy
import math
import time
import json
import traceback
from DataSetLoader import *
#Imported from https://github.com/vinigracindo/pycudaDistances/

from ref_1_distances import pearson_correlation

"""GLOBALS"""
logTimings=1
logToFile=1
start_time = time.time()
logFileName = 'log.txt'
""""""
with open(logFileName, "w") as logfile:
    logfile.write("started at %s " % start_time);
    logfile.close();

"""UTILITY FUNCTIONS"""
def logDebug(tag, msg):
	if(logTimings):
		print((time.time() - start_time))
		print("--- to " + tag + "---");
	if(logToFile):
		with open(logFileName, "a") as logfile:
			logfile.write("--- to " + tag + "---");
			logfile.write("%s" %(time.time() - start_time));
			logfile.write("\n");
			logfile.close();
	print(msg)
"""END UTILITY FUNCTIONS"""

""" FUNCTION FindMaxCorelatedFeatures"""
def FindMaxCorelatedFeatures( F ):
	pearsonCorr = pearson_correlation(F, F)
	numOfVectors = F[0,:].size
	val = -1	
	for i in range (0, numOfVectors): #since j starts from i+1 anyway so last iteration will compare second last with last vector		
		for j in range(i+1, numOfVectors):
			try:
				Fa = F[:,i]
				Fb = F[:,j]
				"""manual
				# http://stackoverflow.com/questions/3045040/python-how-to-find-a-correlation-between-two-vectors
				nX = 1/(sum([x*x for x in Fa]) ** 0.5)
				nY = 1/(sum([y*y for y in Fb]) ** 0.5)
				cor = sum([(x*nX)*(y*nY)  for x,y in zip(Fa,Fb) ])
				"""
				# through numpy
				corr = numpy.corrcoef(Fa, Fb)
				corVal = corr[0,1]
				#print("i=%s, " %i + "j=%s, " %j + "corVal = %s " %corVal);
				if val < corVal:
				   val = corVal
				   firstVector=i
				   secondVector=j            
			except:
				traceback.print_exc()
				logDebug("failed for  i=", i)
				logDebug("failed for  j=", j)
				#return [i, j, val];
	return [firstVector, secondVector, val];
""" END OF FUNCTION FindMaxCorelatedFeatures"""
"""Ref: http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
""""""
"""FUNCTION generateNewMetaGene
	Use Local PCA i.e.
	for the two features, calculate local PCA, then calculate Jacobi rotation and then calculate (coarse grain) m = Fa cos ThetaL + Fb sin ThetaL 
	and fine grained d = fa cos ThetaL - Fb sin ThetaL
	
	we need ThetaL for this purpose; which is the angle between the first two components of the PCA result
"""
def generateNewMetaGene(Fa, Fb):
	pca = PCA(numpy.column_stack((Fa, Fb)));
	# need to use either .a or .Y from this object; a is the original but normalized matrix while Y is the transformed one.	
	ThetaL = angle(pca.Y[0], pca.Y[1]); #find the angle of rotation between the two transformed vectors
	# for now we will just use m as new meta gene
	m = Fa * math.cos (ThetaL) +  Fb * math.sin(ThetaL)
	d = Fa * math.cos (ThetaL) -  Fb * math.sin(ThetaL)	
	return m;
"""End of generateNewMetaGene"""	
"""Function performTreeletClustering"""
def performTreeletClustering(fileBasePath):
	mat = scipy.io.loadmat(fileBasePath + '../data/DataSetA-Loaded-88x1004004.mat')
	logDebug("load the file "," fileLoaded");
	#% manually calculating correlation
	G = mat['G0'];
	F = G;
	M = [];
	p = 1 #F[0,:].size
	for i in range (0, p):
		#steps 1 & 2 of the fig.1 of 20160224 - find pair wise correlation of F and pick the two most correlated columns
		theVectors = pearson_correlation(F, F)
		#theVectors = FindMaxCorelatedFeatures(F)
		print (theVectors);
		return F
		Fa = F[:,theVectors[0]];
		Fb = F[:,theVectors[1]];		
		logDebug(" find a pair of most correlated vectors ", i);		
		#generate a new meta gene using the two picked up columns
		m = generateNewMetaGene(Fa, Fb)
		logDebug(" generate one new gene ","");
		#delete the two selected columns from the F and add the newly generate m to F; scipy.delete(F, 0 based index of col, 0=row and 1=col)
		F = scipy.delete(F, theVectors[0], 1)		
		F = scipy.delete(F, theVectors[1], 1)
		logDebug(" delete Fa & Fb ",F[0,:].size)
		F = numpy.column_stack((m, F)) #include in the main feature set
		if not len(M):
			M = m
		else:
			M = numpy.column_stack((m, M)) #include in the meta genes set as well
		logDebug(" append meta gene ", "")

	F = numpy.column_stack((G, M)) #scipy.append(G, M, 1) #define a new expanded featureset F = G U M	
	logDebug(" generate treelet clustering ", "")
	return F
"""END OF FUNCTION performTreeletClustering"""
"""From PyCudaDistances Open Source Lib"""
def pearson_correlation(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_1, n_features]

    Y : {array-like}, shape = [n_samples_2, n_features]

    Returns
    -------
    distances : {array}, shape = [n_samples_1, n_samples_2]

    Examples
    --------
    >>> from pycudadistances.distances import pearson_correlation
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> pearson_correlation(X, X)
    array([[ 1., 1.],
           [ 1., 1.]])
    >>> pearson_correlation(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.39605904],
               [ 0.39605904]])
    """
    X, Y = check_pairwise_arrays(X,Y)
    
    rows = X.shape[0]
    cols = Y.shape[0]
    
    dx, mx = divmod(cols, BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    solution = numpy.zeros((rows, cols))
    solution = solution.astype(numpy.float32)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void pearson(float *x, float *y, float *solution) {
        
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NDIM)s ) ) {
                float sum_xy, sum_x, sum_y, sum_square_x, sum_square_y;
                
                sum_x = sum_y = sum_xy = sum_square_x = sum_square_y = 0.0f;
                
                for(int iter = 0; iter < %(NDIM)s; iter ++) {
                    float x_e = x[%(NDIM)s * idy + iter];
                    float y_e = y[%(NDIM)s * idx + iter];
                    sum_x += x_e;
                    sum_y += y_e;
                    sum_xy += x_e * y_e;
                    sum_square_x += pow(x_e, 2);
                    sum_square_y += pow(y_e, 2);
                }
                int pos = idx + %(NCOLS)s * idy;
                float denom = sqrt(sum_square_x - (pow(sum_x, 2) / %(NDIM)s)) * sqrt(sum_square_y - (pow(sum_y, 2) / %(NDIM)s));
                if (denom == 0) {
                    solution[pos] = 0;
                } else {
                    float quot = sum_xy - ((sum_x * sum_y) / %(NDIM)s);
                    solution[pos] = quot / denom;
                }
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        'NCOLS': cols,
        'NDIM': X.shape[1]
    }
    
    mod = SourceModule(kernel_code)
    
    func = mod.get_function("pearson")
    func(drv.In(X), drv.In(Y), drv.Out(solution), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    
    return solution
def check_pairwise_arrays(X, Y, dtype=numpy.float32):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the second dimension of the two arrays is equal.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples_a, n_features]

    Y : {array-like}, shape = [n_samples_b, n_features]

    Returns
    -------
    safe_X : {array-like}, shape = [n_samples_a, n_features]
        An array equal to X, guarenteed to be a numpy array.

    safe_Y : {array-like}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guarenteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    if Y is X or Y is None:
        X = Y = safe_asarray(X, dtype=dtype)
    else:
        X = safe_asarray(X, dtype=dtype)
        Y = safe_asarray(Y, dtype=dtype)
    
    if len(X.shape) < 2:
        raise ValueError("X is required to be at least two dimensional.")
    if len(Y.shape) < 2:
        raise ValueError("Y is required to be at least two dimensional.")
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))
    
    return X, Y
# from Utils.py
def safe_asarray(X, dtype=None, order=None):
    """Convert X to an array or sparse matrix.

    Prevents copying X when possible; sparse matrices are passed through."""
    if sparse.issparse(X):
        assert_all_finite(X.data)
    else:
        X = np.asarray(X, dtype, order)
        assert_all_finite(X)
    return X
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    if X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum()) \
      and not np.isfinite(X).all():
            raise ValueError("Array contains NaN or infinity.")


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    _assert_all_finite(X.data if sparse.issparse(X) else X)

"""END OF PyCudaDistances Open Source Lib"""
""" MACHINE SPECIFIC SETTINGS """
fileBasePath = '/home/javedzahoor/research/mattia/code/'

""" MAIN CODE FOR TREELET CLUSTERING """
F = performTreeletClustering(fileBasePath)
#json.dump(F, open('treelet.json','w'));
numpy.save('treelet.npy', F)	
with open("log.txt", "a") as logfile:
	logfile.write("--- %s completed the treelet clustering " + tag + "---" % (time.time() - start_time));
	logfile.close();