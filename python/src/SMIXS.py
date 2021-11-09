from typing          import Tuple

from numpy.core import numeric
from src.SMIXSHelper import init_penatly_matrix

import numpy as np

from sklearn.cluster import KMeans

class SMIXS:

    def __init__(self, *,
        number_of_measurements: int,
        number_of_subjects:     int,
        number_of_clusters:     int,
        K:    np.array = None,
        R:    np.array = None,
        Q:    np.array = None,
        QQ:   np.array = None,
        number_of_iterations: int = 50):

        if K  is None or \
            R  is None or \
            Q  is None or \
            QQ is None:

            (K, R, Q) = init_penatly_matrix(number_of_measurements)
            QQ = np.matmul(np.transpose(Q), Q)

        self.number_of_measurements = number_of_measurements
        self.number_of_iterations   = number_of_iterations
        self.number_of_clusters     = number_of_clusters
        self.number_of_subjects     = number_of_subjects

        # Internal state #

        self.cluster_mean     = np.zeros(shape = (number_of_clusters, number_of_measurements), dtype = np.float32)
        self.cluster_variance = np.zeros(shape = number_of_clusters, dtype = np.float32)
        self.cluster_alpha    = np.zeros(shape = number_of_clusters, dtype = np.float32)
        self.cluster_mixing   = np.zeros(shape = number_of_clusters, dtype = np.float32)
        self.labels           = np.zeros(shape = (number_of_subjects, number_of_clusters), dtype = np.float32)

        self.FILL_ZERO_MEMBER_CLUSTERS = True

    def fit(self, data: np.array):

        for j in range(0, 50):
            
            self._init_kmeans(data = data)
            self._fit(data = data)

    def _fit(self, data: np.array):

        old_likelihood = 0
        stop_flag      = False

        for step in range(0, self.number_of_iterations):
            
            self._expectation_step(data = data)
            # TODO 


    def _expectation_step(self, *, data: np.array) -> None:

        t1 = np.log(self.cluster_mixing)
        t2 = -self.number_of_measurements*0.5*np.log(self.cluster_variance)

        for i in range(0, self.number_of_subjects):

            t0 = np.sum(np.square(self.cluster_mean - data[i, :])/self.cluster_mean, axis = 1)
            t = t0 + t1 + t2

            c = np.max(t)
            t = np.exp(t - c)
            self.labels[i, :] = t/np.sum(t)

        if self.FILL_ZERO_MEMBER_CLUSTERS:

            n_subjects_per_cluster = np.sum(self.labels, axis = 0)
            min_memebers = 1

            for k in range(0, self.number_of_clusters):

                if n_subjects_per_cluster[k] >= min_memebers:
                    continue

                cluster_index = np.nonzero(n_subjects_per_cluster >= (min_memebers + 1))
                subject_index = np.nonzero(np.sum(self.labels[:, cluster_index], axis = 1) >= (self.number_of_clusters - 1)/self.number_of_clusters)

                mean     = self.cluster_mean[k, :]
                variance = self.cluster_variance[k]

                t = np.zeros(shape = subject_index.shape[0], dtype = np.float32)

                for i in subject_index:

                    t0 = -0.5*np.sum(np.square(mean - data[i, :]))/variance
                    
                    t[i] = t0 + t1[k] + t2[k]

                c = subject_index[np.argmax(t)]
                self.labels[c, :] = 0.0
                self.labels[c, k] = 1.0

                n_subjects_per_cluster = np.sum(self.labels, axis = 0)


    def _init_kmeans(self, *, data: np.array) -> None:

        results = KMeans(n_clusters = self.number_of_clusters).fit(data)
        self.cluster_mean = results.cluster_centers_

        for i in range(0, self.number_of_clusters):
            indices = results.labels_ == i

            self.cluster_variance[i] = np.mean(np.var(data[indices, :], axis = 0)) if np.sum(indices) > 2 else \
                (np.mean(np.var(data[indices, :]) if np.sum(indices) == 1 else 1.0))
            self.cluster_mixing[i] = np.sum(indices)/results.labels_.shape[0]
            self.cluster_alpha[i]  = 1.0
            