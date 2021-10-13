# distutils: language = c++
# distutils: sources = ./src/OeSNN.cpp
# Cython interface file for wrapping the object

from libcpp cimport bool
from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "OeSNN.h" namespace "snn":
  cdef cppclass OeSNN:
        OeSNN(int, int, int, double, double, double, double, double, double, bool) except +
        double Predict(double)
        double GetClassification()
        vector[int] GetLOG()

# creating a cython wrapper class
cdef class PyOeSNN:
    cdef OeSNN *thisptr # hold a C++ instance which we're wrapping

    def __cinit__(self, int Wsize, int NOsize, int NIsize, double TS, double sim, double C,
                  double mod, double errorCorrection, double scoreWindowSize, bool random, bool debug):
        self.thisptr = new OeSNN(Wsize, NOsize, NIsize, TS, sim, C, mod, errorCorrection, scoreWindowSize, random)

    def __dealloc__(self):
        del self.thisptr

    def Predict(self, values):
        return self.thisptr.Predict(values)

    def GetClassification(self):
        return self.thisptr.GetClassification()

    def GetFiredLog(self):
        return self.thisptr.GetLOG()
