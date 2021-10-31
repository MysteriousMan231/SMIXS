from typing          import Tuple
from src.SMIXSHelper import init_penatly_matrix

import numpy as np

from sklearn.cluster import KMeans

def SMIXS(*, 
    data: np.array,
    K:    np.array = None,
    R:    np.array = None,
    Q:    np.array = None,
    QQ:   np.array = None,
    cluster_mean:     np.array = None,
    cluster_variance: np.array = None,
    cluster_alpha:    np.array = None,
    cluster_mixing:   np.array = None,
    number_of_clusters:   int = 2,
    number_of_iterations: int = 50):
    
    number_of_measurements = data.shape[1]
    number_of_subjects     = data.shape[0]

    if K  is None or \
       R  is None or \
       Q  is None or \
       QQ is None:

       (K, R, Q) = init_penatly_matrix(number_of_measurements)
       QQ = np.matmul(np.transpose(Q), Q)


def _init_random(*, data: np.array, number_of_clusters: int) -> \
    Tuple[np.array, np.array, np.array, np.array]:

    variance = np.zeros(shape = number_of_clusters, dtype = np.float32)
    
    results = KMeans(n_clusters = number_of_clusters, max_iter = 50).fit(data)

    pass