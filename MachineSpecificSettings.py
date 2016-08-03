from GlobalUtils import *
class Settings(object):
    def getBasePath(self):
        return '/home/javedzahoor/research/mattia/code/';
    
    def getDatasetAFileName(self):
        return 'DataSetA-Loaded-88x1004004.mat';
        
    def getInterimPath(self):
        return '../data/';
    def isLocalMachine(self):
        simulating = False;
        if simulating:
            logWarning("WORKING FOR LOCAL SETTINGS");
        return simulating;
    def sampleSize(self):
        logWarning("HARD CODED VALUE IN MachineSpecificSettings.sampleSize()");
        return 1004004;#;10040