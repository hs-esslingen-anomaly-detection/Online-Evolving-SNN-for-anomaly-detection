# distutils: language = c++
# distutils: sources = ./src/OeSNN.cpp ./include/sxdgrf/grf.cpp ./include/sxdgrf/kmeans/kmeans.cpp
# distutils: include_dirs = ./include/sxdgrf/include
# cython: language_level=3, boundscheck=False
# Cython interface file for wrapping OeSNN.cpp

from libcpp cimport bool
from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "OeSNN.h" namespace "snn":
  cdef cppclass OeSNN:
        OeSNN(int, int, int, int, double, double, double, double, double, double, int, bool, bool) except +
        vector[double] Predict(vector[double])
        vector[vector[double]] PredictAll(vector[vector[double]])
        double GetClassification()
        int GetUnfiredTrials()
        int GetNumberOutputRepoNeurons()
        void Flush()
        
        # functions for debugging and testing
        vector[int] GetUpdateTypeLog()
        vector[vector[double]] GetSpikingTimes()
        vector[vector[double]] GetExc()
        vector[int] GetFiredLog()


# cython wrapper class
cdef class PyOeSNN:
    cdef OeSNN *thisptr # hold a C++ instance which we're wrapping
            
    def __cinit__(self, int dim, int Wsize, int NOsize, int NIsize, double tau, double weightBias, double sim, double C,
                  double errorCorrection, double anomalyThreshold, int scoreWindowSize, bool random, bool debug):
        self.thisptr = new OeSNN(dim, Wsize, NOsize, NIsize, tau, weightBias, sim, C, errorCorrection, anomalyThreshold, scoreWindowSize, random, debug)
    
    def __dealloc__(self):
        del self.thisptr
    
    def Predict(self, x_t):
        return self.thisptr.Predict(x_t)
    
    def PredictAll(self, values): # values[timesteps, dim]
        return self.thisptr.PredictAll(values)
    
    def GetClassification(self):
        return self.thisptr.GetClassification()

    def GetUnfiredTrials(self):
        return self.thisptr.GetUnfiredTrials()

    def GetNumberOutputRepoNeurons(self):
        return self.thisptr.GetNumberOutputRepoNeurons()

    def Flush(self):
        self.thisptr.Flush()
    

    # functions for debugging and testing
    def GetUpdateTypeLog(self):
        return self.thisptr.GetUpdateTypeLog()
    
    def GetSpikingTimes(self):
        return self.thisptr.GetSpikingTimes()
    
    def GetExc(self):
        return self.thisptr.GetExc()
    
    def GetFiredLog(self):
        return self.thisptr.GetFiredLog()

