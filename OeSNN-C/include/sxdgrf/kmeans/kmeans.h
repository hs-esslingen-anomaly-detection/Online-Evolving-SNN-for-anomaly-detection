//
// Created by tobias on 22.03.20.
//

#ifndef SXDGRF_LIB_CPP_KMEANS_H
#define SXDGRF_LIB_CPP_KMEANS_H

/*
//#include <stdexcept>
//#undef eigen_assert
//#define eigen_assert(x) \
//  if (!(x)) { throw (std::runtime_error("eigen_assert")); }
*/

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#define STATIONARY -1

class kmeans {
public:
    explicit kmeans(int k, double alpha, bool debug = false);
    ~kmeans() = default;

    //Basic structure for datatype of single cluster
    struct  cluster {
    public:
        explicit cluster(const Eigen::VectorXd& x, double alpha, bool debug = false, unsigned int debug_idx=-1):
            center(x), n(1), alpha(alpha), debug(debug) {

            if(debug) {
                debug_idxs.push_back(debug_idx);
            }
        };

        void update(const Eigen::VectorXd& x, unsigned int debug_idx=-1) {
            n++;
            auto alpha_val = (alpha == STATIONARY) ? 1.0/n : alpha;
            center = (1-alpha_val)*center+alpha_val*x;
            if(debug) {
                elements.push_back(x);
                debug_idx++;
                debug_idxs.push_back(debug_idx);
            }
        };

        [[nodiscard]] const Eigen::VectorXd& get_center() const {return center;}
        [[nodiscard]] const int& get_number_of_items() const {return  n;}
        [[nodiscard]] const std::vector<Eigen::VectorXd>& get_elements() const {return elements;};
        [[nodiscard]] const std::vector<unsigned int>& get_debug_idxs() const {return debug_idxs;};
    private:
        Eigen::VectorXd center; 
        int n; 
        double alpha; 
        bool debug; 
        std::vector<Eigen::VectorXd> elements{};
        std::vector<unsigned int> debug_idxs{};
    };

    [[nodiscard]] const std::vector<cluster>& get_clusters() const { return clusters; }
    [[nodiscard]] const unsigned int& get_last_updated_idx() const { return last_updated_idx; }

    std::vector<cluster> update(const std::vector<double>& x);
    std::vector<cluster> update(const Eigen::VectorXd& x);

private:
    unsigned int debug_idx{};
    unsigned int k{};
    unsigned int n{};
    unsigned int last_updated_idx{};
    double alpha;
    bool debug;

    static int combine_clusters(const Eigen::VectorXd& x, const std::vector<cluster>& cl);

    static inline double euclidean(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    static inline std::pair<int, double > calculate_min_dist(const Eigen::VectorXd& x, const std::vector<cluster>& cl);

    std::vector<cluster> clusters;
};


#endif //SXDGRF_LIB_CPP_KMEANS_H
