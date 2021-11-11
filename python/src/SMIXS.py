from typing          import Tuple

from numpy.core import numeric
from sklearn import cluster
from src.SMIXSHelper import init_penatly_matrix
from src.SMIXSHelper import plot_single, plot_one

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

class SMIXS_internal_state:

    def __init__(self, *, 
        cluster_mean:     np.array,
        cluster_variance: np.array,
        cluster_alpha:    np.array,
        cluster_mixing:   np.array,
        labels:           np.array) -> None:
        
        self.cluster_mean     = np.copy(cluster_mean)
        self.cluster_variance = np.copy(cluster_variance)
        self.cluster_alpha    = np.copy(cluster_alpha)
        self.cluster_mixing   = np.copy(cluster_mixing)
        self.labels           = np.copy(labels)

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
            QQ = np.dot(np.transpose(Q), Q)

            self.Q  = Q
            self.R  = R 
            self.K  = K
            self.QQ = QQ
        else:
            self.Q  = Q
            self.R  = R
            self.K  = K 
            self.QQ = QQ

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

        # Cholesky decomposition matrices #

        self.chol_diagonal = np.zeros(shape = QQ.shape[0], dtype = np.float32)
        self.chol_lowertri = np.identity(QQ.shape[0], dtype = np.float32)

        # Inverse diagonal matrices

        self.invd_tmp  = np.zeros(shape = (self.number_of_measurements - 2, self.number_of_measurements - 2), dtype = np.float32)
        self.invd_diag = np.zeros(shape = self.number_of_measurements, dtype = np.float32) 

        # Log likelihood buffer matrix

        self.ll_buffer      = np.zeros(shape = (self.number_of_clusters, self.number_of_subjects), dtype = np.float32)
        self.log_likelihood = 0.0

        # Try to remove empty clusters

        self.FILL_ZERO_MEMBER_CLUSTERS = True


    def fit(self, data: np.array):

        best_result = [None, None]

        for j in range(0, 50):
            
            print("Iteration {}/{}".format(j + 1, 50))

            self._init_random(data = data)
            self._fit(data = data)

            if j == 0 or self.log_likelihood > best_result[0]:

                best_result[0] = self.log_likelihood
                best_result[1] = SMIXS_internal_state(
                    cluster_mean     = self.cluster_mean,
                    cluster_variance = self.cluster_variance,
                    cluster_alpha    = self.cluster_alpha,
                    cluster_mixing   = self.cluster_mixing,
                    labels           = self.labels)

        self._restore_from_state(state = best_result[1])
        self.log_likelihood = best_result[0]

    def _fit(self, data: np.array):

        stop_buffer = np.zeros(shape = self.number_of_iterations, dtype = np.float32)
        stop_min_iterations = 5

        for step in range(0, self.number_of_iterations):
            
            self._expectation_step(data = data)
            self._maximization_step(data = data) 
            self._log_likelihood(data = data)

            stop_buffer[step] = self.log_likelihood
          
            if step + 1 >= stop_min_iterations:
        
                r = np.abs((stop_buffer[step] - stop_buffer[step - stop_min_iterations + 1])/stop_buffer[step])
                #print("Relative error magnitude: {:.5f}".format(r))

                if r < 5*1e-3:
                    break

    def _expectation_step(self, *, data: np.array) -> None:

        t1 = np.log(self.cluster_mixing)
        t2 = -self.number_of_measurements*0.5*np.log(self.cluster_variance)

        for i in range(0, self.number_of_subjects):

            t0 = -0.5*np.sum(np.square(self.cluster_mean - data[i, :]), axis = 1)/self.cluster_variance
            t = t0 + t1 + t2

            t = np.exp(t - np.max(t))
            self.labels[i, :] = t/np.sum(t)

        if self.FILL_ZERO_MEMBER_CLUSTERS:

            n_subjects_per_cluster = np.sum(self.labels, axis = 0)
            min_memebers = 1

            for k in range(0, self.number_of_clusters):

                if n_subjects_per_cluster[k] >= min_memebers:
                    continue

                cluster_index = np.nonzero(n_subjects_per_cluster >= (min_memebers + 1))[0]
                subject_index = np.nonzero((np.sum(self.labels[:, cluster_index], axis = 1) >= (self.number_of_clusters - 1)/self.number_of_clusters).flatten())[0]

                mean     = self.cluster_mean[k, :]
                variance = self.cluster_variance[k]

                t = np.zeros(shape = subject_index.shape[0], dtype = np.float32)

                for i in range(0, subject_index.shape[0]):

                    index = subject_index[i]

                    t0 = -0.5*np.sum(np.square(mean - data[index, :]))/variance
                    
                    t[i] = t0 + t1[k] + t2[k]

                c = subject_index[np.argmax(t)]
                self.labels[c, :] = 0.0
                self.labels[c, k] = 1.0

                n_subjects_per_cluster = np.sum(self.labels, axis = 0)

        self.cluster_mixing[:] = np.sum(self.labels, axis = 0)/self.number_of_subjects
        self.cluster_mixing[:] = self.cluster_mixing/np.sum(self.cluster_mixing)

    def _maximization_step(self, *, data: np.array) -> None:
        
        eps = 1e-10
        h   = 0.1

        for i in range(0, self.number_of_clusters):

            cluster_labels = self.labels[:, i]

            W  = np.sum(cluster_labels)
            Nk = W*self.number_of_measurements
            wY = np.sum(np.transpose(np.transpose(data)*cluster_labels), axis = 0)

            c0 = self._cross_validation(alpha = self.cluster_alpha[i] + h, k = i, wY = wY, W = W, data = data)
            #c1 = self._cross_validation(alpha = self.cluster_alpha[i] - h, k = i, wY = wY, W = W, data = data)
            c2 = self._cross_validation(alpha = self.cluster_alpha[i],     k = i, wY = wY, W = W, data = data)

            d0 = (c0 - c2)/h
            #d1 = (c0 - 2*c2 + c1)/h**2
            #if np.abs(d1) >= 1e-6:
            self.cluster_alpha[i] = np.clip(self.cluster_alpha[i] - d0*0.1, 1.0, 1e6)
            
            self._cholesky_decomposition(A = self.QQ*self.cluster_alpha[i] + self.R*W)
            T = np.linalg.solve(np.transpose(self.chol_lowertri), np.linalg.solve(self.chol_lowertri, np.dot(np.transpose(self.Q), wY))/self.chol_diagonal)

            self.cluster_mean[i, :]  = (wY - self.cluster_alpha[i]*np.dot(self.Q, T))/W
            self.cluster_variance[i] = (np.sum(np.sum(np.square(data - self.cluster_mean[i, :]), axis = 1)*cluster_labels))/Nk 
            
            T = np.dot(np.dot(self.cluster_mean[i, :], self.K), self.cluster_mean[i, :])

            self.cluster_variance[i] += self.cluster_alpha[i]*T/Nk

            if self.cluster_variance[i] < eps:
                self.cluster_variance[i] = np.var(data)

    def _cholesky_decomposition(self, *, A: np.array) -> None:

        N = self.QQ.shape[0]

        """ self.chol_diagonal[:] = 0.0
        self.chol_lowertri[:, :] = 0.0
        self.chol_lowertri[np.diag_indices(N)] = 1.0 """

        for j in range(0, N):

            self.chol_diagonal[j] = A[j, j]
            for k in range(np.clip(j - 2, 0, j - 2), j):
                self.chol_diagonal[j] = self.chol_diagonal[j] - (self.chol_lowertri[j, k]**2)*self.chol_diagonal[k]

            if (j + 1) < N:

                self.chol_lowertri[j + 1, j] = A[j + 1, j]/self.chol_diagonal[j]

                if (j - 1) >= 0:
                    self.chol_lowertri[j + 1, j] = self.chol_lowertri[j + 1, j] \
                        - self.chol_lowertri[j + 1, j - 1]*self.chol_lowertri[j, j - 1]*self.chol_diagonal[j - 1]/self.chol_diagonal[j]

            if (j + 2) < N:
                self.chol_lowertri[j + 2, j] = A[j + 2, j]/self.chol_diagonal[j]

    def _cross_validation(self, *, alpha: float, k: int, wY: np.array, W: float, data: np.array) -> float:

        A = 0
        self._cholesky_decomposition(A = self.QQ*alpha + self.R*W)

        T = np.linalg.solve(np.transpose(self.chol_lowertri), np.linalg.solve(self.chol_lowertri, np.dot(np.transpose(self.Q), wY))/self.chol_diagonal)
        gk = (wY - alpha*np.dot(self.Q, T))/W

        self._invert_diagonal(alpha = alpha)

        for j in range(0, self.number_of_subjects):
            A += self.labels[j, k]*np.sum(np.square((data[j, :] - gk)/(1e-10 + 1.0 - self.invd_diag*self.labels[j, k])))
            #A += self.labels[j, k]*np.sum(np.square((data[j, :] - T)))

        return A

    def _invert_diagonal(self, *, alpha: float) -> None:

        N = self.number_of_measurements - 2
        """ self.invd_diag[:] = 0.0
        self.invd_tmp[:, :] = 0.0 """

        for i in range(N - 1, -1, -1):

            self.invd_tmp[i, i] = 1.0/self.chol_diagonal[i] \
                - (self.chol_lowertri[i + 1, i]*self.invd_tmp[i, i + 1] if (i + 1) < N else 0.0) \
                - (self.chol_lowertri[i + 2, i]*self.invd_tmp[i, i + 2] if (i + 2) < N else 0.0)

            if i - 1 >= 0:
                self.invd_tmp[i - 1, i] = -self.chol_lowertri[i, i - 1]*self.invd_tmp[i, i] \
                    - (self.chol_lowertri[i + 1, i - 1]*self.invd_tmp[i, i + 1] if (i + 1) < N else 0.0)
            if i - 2 >= 0:
                self.invd_tmp[i - 2, i] = -self.chol_lowertri[i - 1, i - 2]*self.invd_tmp[i - 1, i] \
                    - self.chol_lowertri[i, i - 2]*self.invd_tmp[i, i]

        for i in range(0, self.number_of_measurements):

            T0 = self.invd_tmp[i - 2, i - 2]*self.Q[i, i - 2]*self.Q[i, i - 2] if (i - 2 >= 0) else 0.0
            T1 = self.invd_tmp[i - 1, i - 1]*self.Q[i, i - 1]*self.Q[i, i - 1] if (i - 1 >= 0) and (i < N + 1) else 0.0
            T2 = self.invd_tmp[i, i]*self.Q[i, i]*self.Q[i, i]                 if i < N else 0.0

            T3 = 2*self.invd_tmp[i - 2, i - 1]*self.Q[i, i - 2]*self.Q[i, i - 1] if (i - 2 >= 0) and (i < N + 1) else 0.0
            T4 = 2*self.invd_tmp[i - 2, i]*self.Q[i, i - 2]*self.Q[i, i]         if (i - 2 >= 0) and (i < N) else 0.0
            T5 = 2*self.invd_tmp[i - 1, i]*self.Q[i, i - 1]*self.Q[i, i]         if (i - 1 >= 0) and (i < N) else 0.0

            self.invd_diag[i] = 1.0 - (T0 + T1 + T2 + T3 + T4 + T5)*alpha

    def _log_likelihood(self, *, data: np.array) -> None:

        A = 0.0
        for k in range(0, self.number_of_clusters):

            e0 = -0.5*np.sum(np.square(data - self.cluster_mean[k, :]), axis = 1)/self.cluster_variance[k]
            e1 = (2.0*np.pi*self.cluster_variance[k])**(-0.5*self.number_of_measurements)

            self.ll_buffer[k, :] = e0 + np.log(e1) + np.log(self.cluster_mixing[k])

            A = A - 0.5*(self.cluster_alpha[k]/self.cluster_variance[k])*(np.dot(np.dot(self.cluster_mean[k, :], self.K), self.cluster_mean[k, :]))

        M = np.max(self.ll_buffer, axis = 0)

        self.log_likelihood =  A + np.sum(np.log(np.sum(np.exp(self.ll_buffer - M), axis = 0)) + M)

    def _bic(self, *, log_likelihood: float) -> None:

        k = self.number_of_clusters + self.number_of_measurements*self.number_of_clusters
        n = self.number_of_subjects*self.number_of_measurements

        self.bic = k*np.log(n) - 2*log_likelihood

    def _init_kmeans(self, *, data: np.array) -> None:

        results = KMeans(n_clusters = self.number_of_clusters).fit(data)
        self.cluster_mean[:, :] = results.cluster_centers_

        for i in range(0, self.number_of_clusters):
            indices = results.labels_ == i

            self.cluster_variance[i] = np.mean(np.var(data[indices, :], axis = 0)) if np.sum(indices) > 2 else \
                (np.mean(np.var(data[indices, :]) if np.sum(indices) == 1 else 1.0))
            self.cluster_mixing[i] = np.sum(indices)/results.labels_.shape[0]
            self.cluster_alpha[i]  = 1.0

    def _init_random(self, *, data: np.array) -> None:

        index = np.random.randint(0, high = self.number_of_subjects, size = self.number_of_clusters)

        self.cluster_mean[:, :] = data[index, :]
        self.cluster_variance[:] = 0.5
        self.cluster_mixing[:] = 1.0/self.number_of_clusters
        self.cluster_alpha[:] = 1.0

    def _restore_from_state(self, *, state: SMIXS_internal_state) -> None:

        self.cluster_mean     = np.copy(state.cluster_mean)
        self.cluster_variance = np.copy(state.cluster_variance)
        self.cluster_alpha    = np.copy(state.cluster_alpha)
        self.cluster_mixing   = np.copy(state.cluster_mixing)
        self.labels           = np.copy(state.labels)