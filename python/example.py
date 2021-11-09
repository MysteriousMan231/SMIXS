from numpy.core import numeric

from src.SMIXSHelper import init_penatly_matrix, plot, to_index

from src.DataGenerator import generate_dataset

import matplotlib.pyplot as plt
import numpy as np

from src.SMIXS import _init_kmeans

def main():
    
    #init_penatly_matrix(10)

    number_of_clusters     = 10
    number_of_measurements = 64
    number_of_subjects     = 200

    (dataset, functions, groundtruth) = generate_dataset(
        number_of_clusters     = number_of_clusters, 
        number_of_measurements = number_of_measurements,
        number_of_subjects     = number_of_subjects)

    results = _init_kmeans(data = dataset, number_of_clusters = number_of_clusters)

    plot(
        predicted_clusters = results.cluster_centers_, 
        predicted_labels   = results.labels_, 
        true_clusters      = functions, 
        true_labels        = to_index(groundtruth), 
        data               = dataset)
    

    

if __name__ == "__main__":
    main()