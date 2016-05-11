import scipy
from MachineSpecificSettings import Settings

class DataSetLoader(object):
    def LoadDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
            mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
            return mat['G0'][:, :];
        else:
            return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
            
    def GetPartSize(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
            return 27889;