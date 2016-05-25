import numpy
import math
#import time
from timeit import default_timer as timer
import json
import traceback

"""GLOBALS"""
debugMessages = True
warnings = False
infoMessages = True

logTimings=1
logToFile=1
start_time = timer()
logFileName = 'log.txt'
""" """
with open(logFileName, "w") as logfile:
    logfile.write("started at %s " % start_time);
    logfile.close();

def logDebug(msg):
    if debugMessages:
        print "DEBUG>>>>> " + str(msg);

def logWarning(msg):
    if warnings:
        print "WARNING===== " + str(msg);

def logInfo(msg):
    if infoMessages:
        print "INFO..... " + str(msg);
    
"""UTILITY FUNCTIONS
def logDebug(tag, msg):
	if(logTimings):
		print((timer() - start_time))
		print("--- to " + tag + "---");
	if(logToFile):
		with open(logFileName, "a") as logfile:
			logfile.write("--- to " + tag + "---");
			logfile.write("%s" %(timer() - start_time));
			logfile.write("\n");
			logfile.close();
	print(msg)
END UTILITY FUNCTIONS"""

##http://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = timer()
        #change this to timer to get more precise estimate of timings
        print("Calling %s" %f.func_name);
        ret = f(*args)
        time2 = timer()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap
    

    