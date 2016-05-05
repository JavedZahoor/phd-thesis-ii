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
