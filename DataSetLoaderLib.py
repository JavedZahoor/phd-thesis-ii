import scipy
from MachineSpecificSettings import Settings

class DataSetLoader(object):
    def LoadDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
            mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
            return mat['G0'][:, 0:100];
        else:
            return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
            
    def GetPartSize(self, dataSetType):
        if dataSetType == "A":
            return 20080; #should be 1004004/36
            
    def CacheTopXPerPart(self, dataSetType):
        if dataSetType == "A":
            return 10000;