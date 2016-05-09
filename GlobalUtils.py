import numpy
import math
import time
import json
import traceback

"""GLOBALS"""
logTimings=1
logToFile=1
start_time = time.time()
logFileName = 'log.txt'
""" """
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

##http://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap