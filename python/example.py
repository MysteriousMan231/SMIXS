from numpy.core import numeric

from src.SMIXSHelper   import plot_single
from src.DataGenerator import generate_dataset

import numpy as np

from src.SMIXS import SMIXS

def main():

    number_of_clusters     = 3
    number_of_measurements = 128
    number_of_subjects     = 200

    np.random.seed(2314233)

    (dataset, functions, groundtruth) = generate_dataset(
        number_of_clusters     = number_of_clusters, 
        number_of_measurements = number_of_measurements,
        number_of_subjects     = number_of_subjects,
        noise_level            = 0)

    np.random.seed(2)

    model = SMIXS(
        number_of_measurements = number_of_measurements,
        number_of_subjects     = number_of_subjects,
        number_of_clusters     = number_of_clusters)

    model.fit(dataset)

    plot_single(
            clusters = model.cluster_mean,
            variance = model.cluster_variance,
            labels   = np.argmax(model.labels, axis = 1),
            data     = dataset) 
    

    

if __name__ == "__main__":
    main()