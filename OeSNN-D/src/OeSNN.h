#ifndef ESNN_RTAD_OESNN_H
#define ESNN_RTAD_OESNN_H

/*
 * src: 
 * - https://arxiv.org/pdf/1912.08785v1.pdf
 * - https://ieeexplore.ieee.org/document/7342967
 * - https://link.springer.com/chapter/10.1007%2F978-3-540-69158-7_9
 */

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cfloat>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include "../include/sxdgrf/grf.h"
#include "MovingAverage.h"

using namespace std;

namespace snn {
    
    class OeSNN {
        /* here are the important functions (interfaces to external code) */
        public:
            
            struct OeSNNConfig {
                unsigned int dimensions; // input signal dimension
                unsigned int Wsize; // size of input window
                unsigned int NOsize; // max number of output neurons in repository
                unsigned int NIsize; // number of input neurons
                double sim; // similarity threshold to merge neurons (0,1]
                double C; // threshold of PSP_max when neuron fired (0,1]
                double tau; // time constant to calculate weights of synapses
                double weightBias; // output neuro initialize weight [0,0.5]
                double errorCorrectionFactor; // xi (0,1]
                double anomalyThreshold; // (0,1)
                unsigned int scoreWindowSize; // outlier score moving average window size (>= 2)
                bool random = false; // seed random generator with time?
            };
            
            OeSNN(int dim, int Wsize, int NOsize, int NIsize, double tau, double weightBias, double sim, double C,
                     double errorCorrection, double anomalyThreshold, int scoreWindowSize, bool random, bool debug);
            OeSNN(OeSNN::OeSNNConfig config);

            ~OeSNN();

            std::vector<double> Predict(const std::vector<double>& x_t); // predict one time step X[t] -> Y[t]
            std::vector< std::vector<double> > PredictAll(const vector<vector<double>> &values); // predict a complete time series at once
            double GetClassification(); // readout of the internal classification for X[t] (Optional)
            int GetUnfiredTrials();
            int GetNumberOutputRepoNeurons();
            void Flush();
            
            /* functions for debugging and testing */
            vector<int> GetUpdateTypeLog();
            vector<vector<double>> GetSpikingTimes();
            vector<vector<double>> GetExc();
            vector<int> GetFiredLog();
            
        private:
            struct OutputNeuron {
                vector<double> weights; // vector of synaptic weights
                vector<double> outputValues; // ny_ni (output of the neuron if fired first)
                double gamma; // actual post-synaptic threshold 
                double M = 1.0; // number of updates
                double PSP = 0.0; // lower approximation of its Postsynaptic potential
                double additionTime; // tau, initialization or update time
                unsigned int ID = 0; // neuron id
            };

            //-------------------------------------
            struct WelfordAggregate {
                unsigned int count = 0;
                double mean = 0;
                double m2 = 0;
                double variance = 0;
                double sample_variance = 0;
            };


            WelfordAggregate _aggregate;
            MovingAverage<double> _movingAverage;

            double ClassifyAnomaly(double error);
            inline void UpdateWelford(double x);
            inline void FinalizeWelford();
            //--------------------------------------

            struct InputNeuron { unsigned int ID; double firingTime; };

            unsigned int _initializationCounter = 0;

            unsigned int _CNOsize = 0; // current output repository size
            unsigned int _neuronIDcounter = 1; // assingn each neuron an unique id
            int _unfiredTrials = 0; // count unfired trials
            long _neuronAge = 0; // counter memory variable to assing the neuron age
            bool _debug = false; // is debugging mode enabled

            default_random_engine _randomGenerator;

            OeSNN::OeSNNConfig _hyperParam; // all OeSNN Hyperparameter
            vector<OutputNeuron *> _outputNeurons; // output repository
            vector<double> _G; // classification
            vector<double> _E; // error between predicted Y[t] and X[t]
            grf _grf; // gaussian receptive field
            vector<InputNeuron> _spikeOrder; // firing order of input neurons for X[t]
            
            /* variables and functions for debugging and testing */
            int _debugCounter = 0;
            vector<vector<double>> _spikingTimes;
            vector<vector<double>> _exc;
            vector<int> _firedLog;
            vector<vector<double>> _initVector;
            vector<int> _updateTypeLog;

            //static inline double CalculateAvg(const vector<double> &vec);
            //static inline double CalculateStd(const vector<double> &vec);
            static inline double CalculateDistance(const vector<double> &v1, const vector<double> &v2);

            void InitializeNeuron(OutputNeuron *n_i, unsigned int dim, const vector<double>& x_t);
            void UpdateNeuron(OutputNeuron *n_i, OutputNeuron *n_s);
            void CalculateSpikeOrder(std::vector<double>& exc_vals);
            void ValueCorrection(OutputNeuron *n_e, const std::vector<double>& x_t);
            void ReplaceOldest(OutputNeuron *n_i);
            void UpdateRepository(const vector<double>& x_t);
            
            std::vector<double> UpdateOutput(const std::vector<double>& x_t);
            double CalculateMaxDistance();

            OutputNeuron *FiresFirst();
            OutputNeuron *FindMostSimilar(OutputNeuron *n_i);

            void Init();
    };
}

#endif //ESNN_RTAD_OESNN_H
