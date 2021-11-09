from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt

def _print_matrix(*, matrix: np.array):

    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            print("{:.2f} ".format(matrix[i,j]), end = "")
        print()

def _fill_diagonal(*, matrix: np.array, vector:np.array, index: Tuple[int, int]) ->  None:
    
    x = index[0]
    y = index[1]
    for i in range(0, len(vector)):
        matrix[x,y] = vector[i]
        x += 1
        y += 1

def to_index(one_hot: np.array):
    return np.argmax(one_hot, axis = 1)

def init_penatly_matrix(n_measurements: int):
    
    t = np.zeros(shape = n_measurements, dtype = np.float32)
    for i in range(1, n_measurements):
        t[i] = i

    h = t[1:] - t[0:-1]

    # Julia linear indexing along columns
    Q = np.zeros(shape = (n_measurements,     n_measurements - 2), dtype = np.float32)
    R = np.zeros(shape = (n_measurements - 2, n_measurements - 2), dtype = np.float32)

    _fill_diagonal(matrix = Q, vector = 1.0/h[:-1],             index = (0, 0))
    _fill_diagonal(matrix = Q, vector = -(1/h[0:-1] + 1/h[1:]), index = (1, 0))
    _fill_diagonal(matrix = Q, vector = 1/h[1:],                index = (2, 0))

    _fill_diagonal(matrix = R, vector = h[1:-1]*(1/6),          index = (1, 0))
    _fill_diagonal(matrix = R, vector = h[1:-1]*(1/6),          index = (0, 1))
    _fill_diagonal(matrix = R, vector = (h[:-1] + h[1:])*(1/3), index = (0, 0))

    return (np.matmul(np.matmul(Q, np.linalg.inv(R)), np.transpose(Q)), R, Q)
    
def plot(*, predicted_clusters: np.array, predicted_labels: np.array, true_clusters: np.array, true_labels: np.array, data: np.array) -> None:

    fig, axs = plt.subplots(2, 1)

    number_of_clusters = predicted_clusters.shape[0]

    cc0 = []

    for i in range(0, number_of_clusters):

        cluster = data[true_labels == i, :]
        c = (np.random.uniform()*0.7, np.random.uniform()*0.7, np.random.uniform()*0.7, 0.25)

        for j in range(0, cluster.shape[0]):
            axs[0].plot(cluster[j,:], color = c)

        cc0.append((c[0], c[1], c[2]))

    for i in range(0, number_of_clusters):

        axs[0].plot(true_clusters[i,:], color = cc0[i], lw = 4)

    cc1 = []

    for i in range(0, number_of_clusters):

        cluster = data[predicted_labels == i, :]
        c = (np.random.uniform()*0.7, np.random.uniform()*0.7, np.random.uniform()*0.7, 0.25)

        for j in range(0, cluster.shape[0]):
            axs[1].plot(cluster[j,:], color = c)

        cc1.append((c[0], c[1], c[2]))

    for i in range(0, number_of_clusters):

        axs[1].plot(predicted_clusters[i,:], color = cc1[i], lw = 4)
    

    plt.show()