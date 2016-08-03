from GlobalUtils import *
import scipy
from MachineSpecificSettings import Settings

class DataSetLoader(object):
    def LoadDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
            mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
            return mat['G0'][:, 0:s.sampleSize()];
        else:
            logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSet()");
            return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
            
    def GetPartSize(self, dataSetType):
        if dataSetType == "A":
            ## HARDCODING
            logWarning("HARD CODED VALUE from DataSetLoaderLib.GetPartSize()");
            return 2510;#20080; #should be 1004004/36
            
    def CacheTopXPerPart(self, dataSetType):
        if dataSetType == "A":
            ## HARDCODING
            logWarning("HARD CODED VALUE from DataSetLoaderLib.CacheTopXPerPart()");
            return 10;#1000