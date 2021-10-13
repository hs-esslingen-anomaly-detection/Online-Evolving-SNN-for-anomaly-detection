from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from os.path import join, exists
from copy import deepcopy
from os import getcwd

import numpy as np

export_path = join(getcwd(), 'results')
save_path = join(getcwd(), 'np_data')
plot_path = join(getcwd(), 'figures')
assert(exists(export_path))
assert(exists(save_path))
print("{}\n{}\n{}".format(export_path, save_path, plot_path))


def calculate_diffs(true_means, est_means, true_cov, est_cov):
    diff_mean = true_means - est_means
    diff_cov = true_cov - est_cov

    diff_cov = diff_cov.reshape(diff_cov.shape[0], diff_cov.shape[1], diff_cov.shape[2]* diff_cov.shape[3])
    np.save('np_data/oscillating_diff_cov', diff_cov)
    np.save('np_data/oscillating_diff_mean', diff_mean)
    return diff_mean, diff_cov

def plot_est_cov(est_cov, k):
    est_cov_plt = est_cov.reshape(est_cov.shape[0], k, 4)
    title = [
        "0,0",
        "0,1",
        "1,0",
        "1,1"
    ]

    for i in range(est_cov_plt.shape[1]):
        for j in range(est_cov_plt.shape[2]):
            plt.plot(est_cov_plt[:, i, j], label='$\^{C}_{%s}$' % title[j])

    ax = plt.gca()
    ax.legend(loc='lower right')
    plt.savefig(join(plot_path, 'recursive_mean_est_covariance_llyoid_kmeans_{}').format(k), dpi=300)
    plt.clf()


def plot_excitement_values(excitment_values, k):
    for i in range(excitment_values.shape[1]):
        plt.plot(excitment_values[:,i], label='mean $| f_{%s}(x)\|$' % str(i))

    plt.title('L1 distance of online estimated vs real covariance $C$ for grf with k={}'.format(k))
    plt.ylabel('manhatten distance $\| C-\^{C} \|$')

    ax = plt.gca()
    ax.legend(loc='upper right')

    plt.savefig(join(plot_path, 'excitement_values_llyoid_kmeans_{}').format(k))
    plt.clf()

def plot_diff_cov(diff_cov, k):
    x = np.arange(diff_cov.shape[0])
    for i in range(diff_cov.shape[1]):
        plt.plot(np.mean(diff_cov[:,i], axis=1), label='mean $| C_{%s} -\^{C}_{%s} \|$' % (str(i), str(i)))
        plt.fill_between(x, np.min(diff_cov[:,i], axis=1), np.max(diff_cov[:,i], axis=1), alpha=0.2, label='min-max $| C_{%s} -\^{C}_{%s} \| $' % (str(i), str(i)))

    plt.title('L1 distance of online estimated vs real covariance $C$ for grf with k={}'.format(k))
    plt.ylabel('Manhatten-Distanz $\| C-\^{C} \|$')

    ax = plt.gca()
    ax.legend(loc='upper right')

    plt.savefig(join(plot_path, 'recursive_mean_covariance_llyoid_kmeans_{}').format(k), dpi=300)
    plt.clf()

def plot_cluster_distances(diff_mean, k):
    x = np.arange(diff_mean.shape[0])
    for i in range(diff_mean.shape[1]):
        plt.plot(np.mean(diff_mean[:,i], axis=1), label='mean $| \mu_{C_{%s}} -\mu_{\^{C}_{%s}} \| $' % (str(i), str(i)))
        plt.fill_between(x, np.min(diff_mean[:,i], axis=1), np.max(diff_mean[:,i], axis=1), alpha=0.2, label='min-max $| \mu_{C_{%s}} -\mu_{\^{C}_{%s}} \| $' % (str(i), str(i)))

    #plt.title(f'L1 distance of online estimated vs real clusters $C$ for grf with k={k}')
    plt.ylabel('Manhatten-Distanz $\| \mu_{C}- \mu_{\^{C}} \|$')
    plt.legend(loc='upper right')
    plt.savefig(join(plot_path, 'recursive_mean_llyoid_kmeans_{}').format(k), dpi=300)
    plt.clf()