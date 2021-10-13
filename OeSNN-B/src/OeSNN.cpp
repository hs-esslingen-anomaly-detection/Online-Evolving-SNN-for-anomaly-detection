#include "OeSNN.h"

using namespace snn;

OeSNN::OeSNN(int Wsize, int NOsize, int NIsize, double TS, double sim,  double C, double mod,
        double errorCorrection, double scoreWindowSize, bool random) : _movingAverage(scoreWindowSize) {

    _randomGenerator = default_random_engine();
    _randomGenerator.seed( random ? (unsigned)std::chrono::system_clock::now().time_since_epoch().count() : 0);

    _hyperParam = {
        .Wsize = (Wsize < 2 ? 2 : Wsize),
        .NOsize = (NOsize < 1 ? 1 : NOsize),
        .NIsize = (NIsize < 1 ? 1 : NIsize),
        .TS = (TS < 1.0 ? 1.0 : TS),
        .sim = (sim <= 0.0 ? 1e-9 : ( sim > 1.0 ? 1.0 : sim )),
        .C = (C <= 0.0 ? 1e-9 : ( C > 1.0 ? 1.0 : C )),
        .mod = (mod <= 0.0 ? 1e-9 : ( mod >= 1.0 ? (1-1e9) : mod )),
        .errorCorrectionFactor = (errorCorrection <= 0.0 ? 1e-9 : ( errorCorrection > 1.0 ? 1.0 : errorCorrection )),
        .scoreWindowSize = scoreWindowSize
    };
}

OeSNN::~OeSNN() {
    for (auto & _outputNeuron : _outputNeurons) delete _outputNeuron;
    _outputNeurons.clear();
    _X.clear();
    _Y.clear();
    _E.clear();
    _G.clear();
    _GRFs.clear();
    _spikeOrder.clear();
    _CNOsize = 0;
    _neuronAge = 0;
}

double OeSNN::CalculateAvg(const vector<double> &vec) {
    return std::accumulate(vec.begin(), vec.end(), (double)0.0) / vec.size();
}

double OeSNN::CalculateStd(const vector<double> &vec) {
    double sqSum = 0.0;
    double avg = OeSNN::CalculateAvg(vec);
    for (double k : vec) sqSum += pow(k - avg, 2);
    return ( vec.size() > 1 ? sqrt(sqSum / (vec.size() - 1.0)) : sqrt(sqSum / 1) );
}

double OeSNN::CalculateDistance(const vector<double> &v1, const vector<double> &v2) {
    double diffSq = 0.0;
    for (int j = 0; j < v1.size(); j++) diffSq += pow(v1[j] - v2[j], 2);
    return sqrt(diffSq);
}

void OeSNN::InitializeGRFs(vector<double> &Window) {
    double I_max = *max_element(Window.begin(), Window.end());
    double I_min = *min_element(Window.begin(), Window.end());

    for (int j = 0; j < _GRFs.size(); j++) {
        _GRFs[j].mu = I_min + ( (2.0 * j - 3.0) / 2.0 ) * ( (I_max - I_min) / (_hyperParam.NIsize - 2.0) );
        _GRFs[j].sigma = (I_max - I_min) / (_hyperParam.NIsize - 2.0);
    }
}

void OeSNN::InitializeNeuron(OeSNN::Neuron *n_i, const vector<double> &Window) {
    /* ensure that all weights exists */
    for (int i = 0; i < _hyperParam.NIsize; i++) n_i->weights.push_back(0.0);

    int order = 0;
    for (auto & n_x : _spikeOrder) {
        /* calculate and set the correct weights */
        n_i->weights[n_x.ID] += pow(_hyperParam.mod, order);
        n_i->PSP_max += n_i->weights[n_x.ID] * pow(_hyperParam.mod, order++);
    }

    normal_distribution<double> distribution(CalculateAvg(Window), CalculateStd(Window));

    n_i->gamma = n_i->PSP_max * _hyperParam.C;
    n_i->outputValue = distribution(_randomGenerator);
    n_i->M = 1.0;
    n_i->additionTime = _neuronAge++;
}

void OeSNN::UpdateNeuron(OeSNN::Neuron *n_i, OeSNN::Neuron *n_s) {
    for (int j = 0; j < n_s->weights.size(); j++) {
        n_s->weights[j] = (n_i->weights[j] + n_s->weights[j] * n_s->M ) / (n_s->M + 1.0 );
    }
    n_s->gamma = ( n_i->gamma + n_s->gamma * n_s->M ) / ( n_s->M + 1.0 );
    n_s->outputValue = ( n_i->outputValue + n_s->outputValue * n_s->M ) / ( n_s->M + 1.0 );
    n_s->additionTime = ( n_i->additionTime + n_s->additionTime * n_s->M ) / ( n_s->M + 1.0 );
    n_s->M += 1.0;
    delete n_i;
}

void OeSNN::CalculateSpikeOrder(vector<double> &Window) {
    _spikeOrder.clear();

    for (int j = 0; j < _GRFs.size(); j++) {
        if (_GRFs[j].sigma == 0.0) _GRFs[j].sigma = 1.0;
        double exc = (exp(-0.5 * pow(((Window[Window.size() - 1] - _GRFs[j].mu) / _GRFs[j].sigma), 2)));
        _spikeOrder.push_back({ j, _hyperParam.TS * (1 - exc) });
    }

    sort(_spikeOrder.begin(), _spikeOrder.end(), [](auto a, auto b) { return a.firingTime < b.firingTime; } );
}

void OeSNN::ValueCorrection(OeSNN::Neuron *n_c, double x_t) {
    n_c->outputValue += ( x_t - n_c->outputValue ) * _hyperParam.errorCorrectionFactor;
}

double OeSNN::ClassifyAnomaly(double error) {
    UpdateWelford(error);
    FinalizeWelford();
    auto standard_deviation = (_aggregate.sample_variance > 0) ? std::sqrt(_aggregate.sample_variance) : 1.0;
    auto movingAverage = _movingAverage.UpdateAverage(error);
    auto cdf = 1 - std::erfc(-(movingAverage - _aggregate.mean)/standard_deviation/std::sqrt(2))/2;
    return 2.0 * std::abs(cdf - 0.5);
    return error;
}

void OeSNN::UpdateWelford(double x) {
    _aggregate.count++;
    auto delta = (x - _aggregate.mean);
    _aggregate.mean += delta/_aggregate.count;
    auto delta2 = (x - _aggregate.mean);
    _aggregate.m2 += delta * delta2;
}

void OeSNN::FinalizeWelford() {
    //Avoid zero division and senseless results from division by -1
    if (_aggregate.count < 2) {
        _aggregate.variance = 0;
        _aggregate.sample_variance = 0;
        return;
    }

    _aggregate.variance = _aggregate.m2/_aggregate.count;
    _aggregate.sample_variance = _aggregate.m2/(_aggregate.count - 1.0);
}

OeSNN::Neuron* OeSNN::FindMostSimilarNeuron(Neuron *n_i) {
    Neuron* simPtr = _outputNeurons[0];
    for (auto & n_x : _outputNeurons) {
        if(CalculateDistance(n_i->weights, n_x->weights) < CalculateDistance(n_i->weights, simPtr->weights)) {
            simPtr = n_x;
        }
    }
    return simPtr;
}

void OeSNN::ReplaceOldestNeuron(Neuron *n_i) {
    double oldest = _outputNeurons[0]->additionTime;
    int oldestIdx = 0;

    for (int k = 1; k < _outputNeurons.size(); k++) {
        if (oldest > _outputNeurons[k]->additionTime) {
            oldest = _outputNeurons[k]->additionTime;
            oldestIdx = k;
        }
    }

    delete _outputNeurons[oldestIdx];
    _outputNeurons[oldestIdx] = n_i;
}


OeSNN::Neuron* OeSNN::GetNeuronSpikeFirst() {
    for (auto & _outputNeuron : _outputNeurons) _outputNeuron->PSP = 0.0;

    Neuron* maxPtr;
    vector<Neuron*> toFire;

    int order = 0;
    for (auto & l : _spikeOrder) {
        for (auto & n_o : _outputNeurons) {
            n_o->PSP += pow(_hyperParam.mod, order) * n_o->weights[l.ID];
            if (n_o->PSP > n_o->gamma) toFire.push_back(n_o);
        }
        if (!toFire.empty()) {
            maxPtr = toFire[0];
            for (auto & n_o : toFire) {
                if ((n_o->PSP - n_o->gamma) > (maxPtr->PSP - maxPtr->gamma)) {
                    maxPtr = n_o;
                }
            }
            return maxPtr;
        }
        order++;
    }

    return nullptr;
}

double OeSNN::CalculateMaxDistance() {
    vector<double> v1, v2;
    for(int i = 0; i < _hyperParam.NIsize; i++) {
        v1.push_back(pow(_hyperParam.mod, _hyperParam.NIsize - 1 - i));
        v2.push_back(pow(_hyperParam.mod, i));
    }

    double diffSq = 0.0;
    for (int i = 0; i < v1.size(); i++) diffSq += pow(v1[i] - v2[i], 2);

    return sqrt(diffSq);
}

void OeSNN::UpdateRepository(int t, bool g_t, const vector<double> &Window) {
    auto *n_c = new Neuron;
    InitializeNeuron(n_c, Window);
    if (!g_t) ValueCorrection(n_c, _X[t].value);

    Neuron* n_s = (_CNOsize > 0 ? FindMostSimilarNeuron(n_c) : nullptr);
    if (_CNOsize > 0 && CalculateDistance(n_c->weights, n_s->weights)
        <= _hyperParam.sim*CalculateMaxDistance()) {
        UpdateNeuron(n_c, n_s);
        _LOG.push_back(1);
    } else if (_CNOsize < _hyperParam.NOsize) {
        _outputNeurons.push_back(n_c);
        _CNOsize++;
        _LOG.push_back(2);
    } else {
        ReplaceOldestNeuron(n_c);
        _LOG.push_back(3);
    }
}

bool OeSNN::PredictNext(int t) {
    double g_t = 1.0;
    Neuron *n_f = GetNeuronSpikeFirst();
    if (n_f == nullptr) {
        _Y.push_back(0);
        _E.push_back(DBL_MAX);
        _LOG.push_back(-1);
    } else {
        double y_t = n_f->outputValue;
        _Y.push_back(y_t);
        _E.push_back(abs(_X[t].value - y_t));
        g_t = ClassifyAnomaly(_E.back());
    }
    _G.push_back(g_t);
    return g_t >= 0.8;
}

double OeSNN::Predict(const double &value) {

    _X.push_back({_counter, value});

    if (_counter == 0) for (int k = 0; k < _hyperParam.NIsize; k++) _GRFs.push_back({ });

    if (_counter < _hyperParam.Wsize) {
        _Window.push_back(value);
        normal_distribution<double> distribution(CalculateAvg(_Window), CalculateStd(_Window));
        _Y.push_back(distribution(_randomGenerator));
        _E.push_back(abs(_X[_counter].value - _Y[_counter]));
        _G.push_back(0.0);
        _LOG.push_back(0);
    } else {
        _Window.erase(_Window.begin());
        _Window.push_back(_X[_counter].value);

        InitializeGRFs(_Window);
        CalculateSpikeOrder(_Window);
        UpdateRepository(_counter, PredictNext(_counter), _Window);
    }

    _counter++;
    return _Y.back();
}

double OeSNN::GetClassification() {
    return _G.back();
}

vector<int> OeSNN::GetLOG() {
    return _LOG;
}
