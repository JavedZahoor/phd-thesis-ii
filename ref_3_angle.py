#https://newtonexcelbach.wordpress.com/2014/03/01/the-angle-between-two-vectors-python-version/
#http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
import numpy
import numpy.linalg as la
def findAngle(u, v): 
    cosTheta = numpy.dot(u, v);
    sinTheta = la.norm(numpy.cross(u, v));
    return numpy.arctan2(sinTheta,cosTheta);