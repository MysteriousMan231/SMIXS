from numpy.core import numeric
from src.SMIXSHelper   import init_penatly_matrix
from src.DataGenerator import generate_dataset

import matplotlib.pyplot as plt
import numpy as np

from src.SMIXS import _init_random

def main():
    
    #init_penatly_matrix(10)

    number_of_clusters     = 3
    number_of_measurements = 64
    number_of_subjects     = 25

    (dataset, _, groundtruth) = generate_dataset(
        number_of_clusters     = number_of_clusters, 
        number_of_measurements = number_of_measurements,
        number_of_subjects     = number_of_subjects)

    _init_random(data = dataset, number_of_clusters = number_of_clusters)


    for j in range(0, number_of_clusters):

        cluster = dataset[np.nonzero(groundtruth[:, j])[0], :]
        c = (np.random.uniform()*0.7, np.random.uniform()*0.7, np.random.uniform()*0.7)

        for i in range(0, cluster.shape[0]):
            plt.plot(cluster[i,:], color = c)
        

    plt.show()

    

if __name__ == "__main__":
    main()