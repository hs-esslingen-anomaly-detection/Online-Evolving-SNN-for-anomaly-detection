#include "OeSNN.h"

using namespace snn;

void OeSNN::Init() {
    _randomGenerator = default_random_engine();
    _randomGenerator.seed( _hyperParam.random ? (unsigned)std::chrono::system_clock::now().time_since_epoch().count() : 1);
}

OeSNN::OeSNN(int dim, int Wsize, int NOsize, int NIsize, double TS, double sim,  double C, double mod,
        double neuronInitFactor, double errorCorrection, int scoreWindowSize, bool random, bool debug) :
        _grf(NIsize, dim), _movingAverage(scoreWindowSize) {

    _hyperParam = {
        .dimensions = (unsigned) dim,
        .Wsize = (unsigned) Wsize,
        .NOsize = (NOsize < 1 ? 1 : (unsigned)NOsize),
        .NIsize = (NIsize < 1 ? 1 : (unsigned)NIsize),
        .TS = (TS < 1.0 ? 1.0 : TS),
        .sim = (sim <= 0.0 ? 1e-9 : ( sim > 1.0 ? 1.0 : sim )),
        .C = (C <= 0.0 ? 1e-9 : ( C > 1.0 ? 1.0 : C )),
        .mod = (mod <= 0.0 ? 1e-9 : ( mod >= 1.0 ? (1-1e9) : mod )),
        .neuronInitFactor = (neuronInitFactor <= 0.0 ? 1e-9 : ( neuronInitFactor > 1.0 ? 1.0 : neuronInitFactor )),
        .errorCorrectionFactor = (errorCorrection <= 0.0 ? 1e-9 : ( errorCorrection > 1.0 ? 1.0 : errorCorrection )),
        .scoreWindowSize = scoreWindowSize,
        .random = random
    };

    _debug = debug;
    _aggregate = WelfordAggregate();
    OeSNN::Init();
}

OeSNN::OeSNN(OeSNN::OeSNNConfig config) :
    _grf(config.NIsize, config.dimensions), _movingAverage(config.scoreWindowSize) {
    _hyperParam = config;
    _debug = false;
    _aggregate = WelfordAggregate();
    OeSNN::Init();
}

OeSNN::~OeSNN() {
    for (auto & _outputNeuron : _outputNeurons) delete _outputNeuron;
    _outputNeurons.clear();
    _E.clear();
    _G.clear();
    _spikeOrder.clear();
    _CNOsize = 0;
    _neuronAge = 0;
}

/*
double OeSNN::CalculateAvg(const vector<double> &vec) {
    return std::accumulate(vec.begin(), vec.end(), (double)0.0) / vec.size();
}

double OeSNN::CalculateStd(const vector<double> &vec) {
    double sqSum = 0.0;
    double avg = OeSNN::CalculateAvg(vec);
    for (double k : vec) sqSum += pow(k - avg, 2);
    return ( vec.size() > 1 ? sqrt(sqSum / (vec.size() - 1.0)) : sqrt(sqSum / 1) );
}
*/

double OeSNN::CalculateDistance(const vector<double> &v1, const vector<double> &v2) {
    double diffSq = 0.0;
    for (unsigned int j = 0; j < v1.size(); j++) diffSq += pow(v1[j] - v2[j], 2);
    return sqrt(diffSq);
}

void OeSNN::InitializeNeuron(OeSNN::Neuron *n_i, unsigned int dim, const vector<double>& x_t) {
    /* ensure that all weights exists */
    for (unsigned int i = 0; i < _hyperParam.NIsize; i++) n_i->weights.push_back(0.0);

    int order = 0;
    double PSP_max = 0.0;
    for (auto & n_x : _spikeOrder) {
        /* calculate and set the correct weights */
        n_i->weights[n_x.ID] += pow(_hyperParam.mod, order);
        PSP_max += n_i->weights[n_x.ID] * pow(_hyperParam.mod, order++);
    }

    n_i->gamma = PSP_max * _hyperParam.C;
    n_i->outputValues = vector<double>(dim);


    auto distributions = _grf.get_marginal_distributions();
    for(unsigned int i = 0; i < n_i ->outputValues.size(); i++) {
        n_i->outputValues[i] = distributions[i](_randomGenerator);
    }

    /*
    for(unsigned int i = 0; i < n_i ->outputValues.size(); i++) {
        n_i->outputValues[i] = x_t[i];
    } */

    n_i->M = 1.0;
    n_i->additionTime = _neuronAge++ * 1.0;
}

void OeSNN::UpdateNeuron(OeSNN::Neuron *n_i, OeSNN::Neuron *n_s) {
    for (unsigned int j = 0; j < n_s->weights.size(); j++) {
        n_s->weights[j] = (n_i->weights[j] + n_s->weights[j] * n_s->M ) / (n_s->M + 1.0 );
    }
    n_s->gamma = ( n_i->gamma + n_s->gamma * n_s->M ) / ( n_s->M + 1.0 ); //TODO add hyperparameter?

    for(unsigned int i = 0; i < n_s->outputValues.size(); i++) {
        n_s->outputValues[i] = ( n_i->outputValues[i] + n_s->outputValues[i] * n_s->M ) / ( n_s->M + 1.0 );
    }

    n_s->additionTime = ( n_i->additionTime + n_s->additionTime * n_s->M ) / ( n_s->M + 1.0 );
    n_s->M += 1.0;
    delete n_i;
}

void OeSNN::CalculateSpikeOrder(std::vector<double>& exc_vals) {
    _spikeOrder.clear();
    for (unsigned int j = 0; j < exc_vals.size(); j++) {
        /* NOTE: There seems to be a problem due to the reference (maybe due to compiler optimization or something else) so we first copy the value
         * into a locale variable to avoid the problem of all 0 values (later we can implement this even better). */
        double exc = std::isnan(exc_vals[j]) ? 1 - 1e-9 : exc_vals[j];
        _spikeOrder.push_back({ (unsigned) j, _hyperParam.TS * (1.0 - exc) });
    }

    sort(_spikeOrder.begin(), _spikeOrder.end(), [](auto a, auto b) { return a.firingTime > b.firingTime; } );

    //TODO Sometimes we don't get back reasonable values, here we have to take a closer look.
    //TODO:
    if(_spikeOrder.front().firingTime == _spikeOrder.back().firingTime) {
        vector<double> tmp;

        auto distributions = _grf.get_marginal_distributions();
        for(unsigned int i = 0; i < _hyperParam.dimensions; i++) {
            tmp.push_back(distributions[i](_randomGenerator));
        }
        (void)_grf.update(tmp);
    }

    /*
    if (_debugCounter++ % 1000 == 0) {
        cout << "ftimes:";
        for (auto & a : _spikeOrder) cout << ", " << a.firingTime;
        cout << endl;
    } */

    if(_debug) {
        vector<double> tmp1;
        for (auto & o : _spikeOrder) tmp1.push_back( o.firingTime );
        _spikingTimes.push_back(tmp1);

        vector<double> tmp2(exc_vals);
        for (unsigned int i = 0; i < tmp2.size(); i++) {
            if(std::isnan(tmp2[i])) {
                tmp2[i] = -1;
            }
        }
        _exc.push_back(tmp2);
    }
}

//TODO test value correct with respect to M?
void OeSNN::ValueCorrection(OeSNN::Neuron *n_c, const vector<double>& x_t) {
    for (unsigned int i = 0; i < x_t.size(); i++) {
        n_c->outputValues[i] += (x_t[i] - n_c->outputValues[i]) * _hyperParam.errorCorrectionFactor * 1/n_c->M;
    }
}

double OeSNN::ClassifyAnomaly(double error) {
    UpdateWelford(error);
    FinalizeWelford();
    auto standard_deviation = (_aggregate.sample_variance > 0) ? std::sqrt(_aggregate.sample_variance) : 1.0;
    auto movingAverage = _movingAverage.UpdateAverage(error);
    auto cdf = 1 - std::erfc(-(movingAverage - _aggregate.mean)/standard_deviation/std::sqrt(2))/2;
    return 2.0 * std::abs(cdf - 0.5);
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
    }

    _aggregate.variance = _aggregate.m2/_aggregate.count;
    _aggregate.sample_variance = _aggregate.m2/(_aggregate.count - 1.0);
}

OeSNN::Neuron* OeSNN::FindMostSimilar(Neuron *n_i) {
    Neuron* simPtr = _outputNeurons[0];
    for (auto & n_x : _outputNeurons) {
        if(CalculateDistance(n_i->weights, n_x->weights) < CalculateDistance(n_i->weights, simPtr->weights)) {
            simPtr = n_x;
        }
    }
    return simPtr;
}

void OeSNN::ReplaceOldest(Neuron *n_i) {
    double oldest = _outputNeurons[0]->additionTime;
    int oldestIdx = 0;

    for (unsigned int k = 1; k < _outputNeurons.size(); k++) {
        if (oldest > _outputNeurons[k]->additionTime) {
            oldest = _outputNeurons[k]->additionTime;
            oldestIdx = k;
        }
    }

    delete _outputNeurons[oldestIdx];
    _outputNeurons[oldestIdx] = n_i;
}


OeSNN::Neuron* OeSNN::FiresFirst() {
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
    for(unsigned int i = 0; i < _hyperParam.NIsize; i++) {
        v1.push_back(pow(_hyperParam.mod, _hyperParam.NIsize - 1 - i));
        v2.push_back(pow(_hyperParam.mod, i));
    }

    double diffSq = 0.0;
    for (unsigned int i = 0; i < v1.size(); i++) diffSq += pow(v1[i] - v2[i], 2);

    return sqrt(diffSq);
}

void OeSNN::UpdateRepository(const vector<double>& x_t) {
    auto *n_c = new Neuron;
    InitializeNeuron(n_c, x_t.size(), x_t);
    if (_G.back() < 0.8) ValueCorrection(n_c, x_t); //if no anomaly --> perfom value correction

    Neuron* n_s = (_CNOsize > 0 ? FindMostSimilar(n_c) : nullptr);
    if (_CNOsize > 0 && CalculateDistance(n_c->weights, n_s->weights)
        <= _hyperParam.sim*CalculateMaxDistance()) {
        UpdateNeuron(n_c, n_s);
        if(_debug) _LOG.push_back(1);
    } else if (_CNOsize < _hyperParam.NOsize) {
        n_c->ID = _neuronIDcounter++;
        _outputNeurons.push_back(n_c);
        _CNOsize++;
        if(_debug) _LOG.push_back(2);
    } else {
        n_c->ID = _neuronIDcounter++;
        ReplaceOldest(n_c);
        if(_debug) _LOG.push_back(3);
    }
}

std::vector<double> OeSNN::UpdateOutput(const std::vector<double>& x_t) {
    vector<double> y_t;

    Neuron *n_f = FiresFirst();
    if (n_f == nullptr) {

        /*TODO: In the current implementation, this function is called at least once, exactly when the window has been read completely,
         * because the _outputRepository is still empty, we should probably implement this better. */
        /*
        auto aggregates = _Window.get_values();

        for (auto agg : aggregates) {
            y_t.push_back(agg.mean); //NOTE: Here we deviate from the paper because otherwise our outlier score does not work well.
        }
        */
        if(_debug) {
            //cout << "Waring: no neuron has fired" << endl;
            _firedLOG.push_back(-1);
        }

        auto distributions = _grf.get_marginal_distributions();
        for(unsigned int i = 0; i < _hyperParam.dimensions; i++) {
            y_t.push_back(distributions[i].mean());
        }

        _E.push_back(DBL_MAX);
        _G.push_back(1.0);
    } else {
        if(_debug) {
            _firedLOG.push_back(n_f->ID);
        }

        y_t = n_f->outputValues;
        _E.push_back(abs(CalculateDistance(x_t, y_t)));


        vector<double> squaredError;
        for (unsigned int i = 0; i < x_t.size(); i++) {
            squaredError.push_back(pow((x_t[i] - y_t[i]), 2));
        }

        _G.push_back(ClassifyAnomaly(*max_element(squaredError.begin(), squaredError.end())));
    }

    if(_E.size() > _hyperParam.Wsize + 2) _E.erase(_E.begin());
    if(_G.size() > _hyperParam.Wsize + 2) _G.erase(_G.begin());

    return y_t;
}

double OeSNN::GetClassification() {
    return _G.empty() ? 1.0 : _G.back();
}

std::vector<double> OeSNN::Predict(const std::vector<double>& x_t) {

    //TODO: "Note !!" Random initialization at first is important !!
    if (_initializationCounter < _hyperParam.Wsize) {
        vector<double> y_t;

        _initVector.push_back(x_t);
        if (_debug) {
            _LOG.push_back(0);
            _firedLOG.push_back(0);
            _spikingTimes.emplace_back( vector<double>(_hyperParam.NIsize, 0.0) );
            _exc.emplace_back( vector<double>(_hyperParam.NIsize, 0.0) );
        }

        auto distributions = _grf.get_marginal_distributions();
        for(unsigned int i = 0; i < _hyperParam.dimensions; i++) {
            y_t.push_back(distributions[i](_randomGenerator));
        }

        _E.push_back(abs(CalculateDistance(x_t, y_t)));
        _G.push_back(0.0);

        // NOTE: test implementation to improve clusters sturcture
        /*
        if (_initializationCounter % (int)(_hyperParam.Wsize/_hyperParam.NIsize) == 0) {
            (void)_grf.update(x_t);
        }*/
        srand(1);
        if(_initializationCounter == _hyperParam.Wsize - 1) {
            for (unsigned int i = 0; i < _hyperParam.NIsize; i++) {
                vector<double> clusterInit;
                for (unsigned int j = 0; j < _hyperParam.dimensions; j++) {
                    double max = _initVector[0][j];
                    double min = _initVector[0][j];
                    for(unsigned int k = 0; k < _initVector.size(); k++) {
                        if(max < _initVector[k][j]) max = _initVector[k][j];
                        if(min > _initVector[k][j]) min = _initVector[k][j];
                    }
                    clusterInit.push_back( (double)rand()/(double)RAND_MAX * (max-min) + min);
                }
                (void)_grf.update(clusterInit);
            }
        }

        _initializationCounter++;
        return y_t;
    }


    auto exc_vals = _grf.update(x_t);
    CalculateSpikeOrder(exc_vals);
    auto y_t = UpdateOutput(x_t);
    UpdateRepository(x_t);

    return y_t; //TODO: Change to (std::isnan(y_t) ? x_t : y_t)
}

vector<vector<double>> OeSNN::PredictAll(const vector<vector<double>> &values) {
    /* clear debung variables */
    _LOG.clear();
    _spikingTimes.clear();
    _exc.clear();
    _firedLOG.clear();

    vector<vector<double>> retValues;
    for (unsigned int i=0; i < values.size(); i++)
        retValues.push_back( Predict(values.at(i)) );

    return retValues;
}


/* functions for debugging and testing */
vector<int> OeSNN::GetLOG() {
    return _LOG;
}

vector<vector<double>> OeSNN::GetSpikingTimes() {
    return _spikingTimes;
}

vector<vector<double>> OeSNN::GetExc() {
    return _exc;
}

vector<int> OeSNN::GetFiredLOG() {
    return _firedLOG;
}
