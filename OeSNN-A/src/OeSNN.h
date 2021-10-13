#ifndef ESNN_RTAD_OESNN_H
#define ESNN_RTAD_OESNN_H

/*
 * src:
 * - https://arxiv.org/pdf/1912.08785v1.pdf
 * - https://github.com/jethrokuan/esnn
 */

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cfloat>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

using namespace std;

namespace snn {

    class OeSNN {
        private:
            struct HyperParam {
                int Wsize; // Size of input window
                int NOsize; // max number of output neurons in repository NO
                int NIsize; // Number of input neurons
                double TS; // Synchronisation time of input neurons firing
                double sim; // Similarity threshold (0,1]
                double C; // Fraction of PSP_max for output neurons output value, gamma_ni (0,1]
                double mod; // Modulation factor of weights of synapses (0,1)
                double errorCorrectionFactor; // xi (0,1]
                double anomalyClassificationFactor; // epsilon (>= 2)
            };
            struct Neuron {
                vector<double> weights; // Vector of synaptic weights
                double outputValue; // ny_ni
                double gamma; // actual post-synaptic threshold (of output neuron)
                double M = 1.0; // Number of updates (of output neuron)
                double PSP = 0.0; // lower approximation of its Postsynaptic potential
                double additionTime; // tau, initialization or update time (of output neuron)
                double PSP_max = 0.0; // maximal post-synaptic threshold (of output neuron)
            };
            struct InputValue { int timeStamp; double value; };
            struct GRF { double mu, sigma; };
            struct InputNeuron { int ID; double firingTime; };

            int _CNOsize = 0; // current output repository size
            int _neuronAge = 0; // counter memory variable
            int _counter = 0;

            default_random_engine _randomGenerator;

            HyperParam _hyperParam;
            vector<Neuron *> _outputNeurons;
            vector<InputValue> _X; // input dataset
            vector<double> _Y; // predicted values
            vector<bool> _G; // classification
            vector<int> _LOG; // debug stuff
            vector<double> _E; // error between predicted Y[t] and X[t]
            vector<GRF> _GRFs; // input GRFs
            vector<InputNeuron> _spikeOrder; // firing order of input neurons for X[t]
            vector<double> _Window;

            static inline double CalculateAvg(const vector<double> &vec);
            static inline double CalculateStd(const vector<double> &vec);
            static inline double CalculateDistance(const vector<double> &v1, const vector<double> &v2);

            void InitializeGRFs(vector<double> &Window);
            void InitializeNeuron(Neuron *n_i, const vector<double> &Window);
            void UpdateNeuron(Neuron *n_i, Neuron *n_s);
            void CalculateSpikeOrder(vector<double> &Window);
            void ValueCorrection(Neuron *n_e, double x_t);
            void ReplaceOldestNeuron(Neuron *n_i);
            void UpdateRepository(int t, bool g_t, const vector<double> &Window);
            bool ClassifyAnomaly();
            bool PredictNext(int t);
            double CalculateMaxDistance();

            Neuron *GetNeuronSpikeFirst();
            Neuron *FindMostSimilarNeuron(Neuron *n_i);

        public:
            OeSNN(int Wsize, int NOsize, int NIsize, double TS, double sim, double C, double mod,
                    double errorCorrection, double anomalyFactor, bool random);

            ~OeSNN();

            double Predict(const double &value);
            bool GetClassification();
            vector<int> GetLOG();
    };
}

#endif //ESNN_RTAD_OESNN_H
