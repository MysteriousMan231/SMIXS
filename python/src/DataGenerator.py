from typing import Tuple
import numpy as np

def generate_dataset(*,
    number_of_measurements: int = None,
    number_of_subjects:     int = None,
    number_of_clusters:     int = None,
    seed:        int = None,
    noise_level: int = None) -> Tuple[np.array, np.array, np.array]:

    if number_of_measurements is None:
        number_of_measurements = 32
    if number_of_subjects is None:
        number_of_subjects = 100
    if number_of_clusters is None:
        number_of_clusters = 3

    if seed is not None:
        np.random.seed(seed)

    if noise_level is None:
        noise_level = 0

    p_min = 0.5

    P = np.random.rand(number_of_clusters)*(1.0 - p_min) + p_min
    P = P/np.sum(P)

    iP = np.argsort(P)
    P  = np.cumsum(P[iP])

    dataset          = np.zeros(shape = (number_of_subjects, number_of_measurements), dtype = np.float32)
    ground_truth     = np.zeros(shape = (number_of_subjects, number_of_clusters),     dtype = np.int32)
    latent_functions = np.zeros(shape = (number_of_clusters, number_of_measurements), dtype = np.float32) 

    # Perlin noise function generator parameters
    step     = 8
    variance = 0.5
    
    x = range(step, number_of_measurements + 1, step)
    y = np.random.randn(len(x) + 1)*np.sqrt(variance)

    top = 10.0
    bot =  0.0

    for i in range(0, number_of_clusters):
        
        y0 = np.copy(y)
        v0 = np.random.randn(y.shape[0])*np.sqrt(0.5)
        v0[v0 < 0] = np.clip(v0[v0 < 0], -top, -bot)
        v0[v0 > 0] = np.clip(v0[v0 > 0],  bot,  top)
        y0 = y0 + v0

        c = 0
        for j in range(0, number_of_measurements):
            latent_functions[i, j] += cerp(y0 = y0[c], y1 = y0[c + 1], a = (j + 1)/step - 1/step)

            if ((j + 1) % step) == 0:
                c += 1

    min = np.amin(latent_functions)
    max = np.amax(latent_functions)

    latent_functions = (latent_functions - min)/(max - min)
    for j in range(0, number_of_subjects):
        idx = iP[sum(P < np.random.uniform())]

        dataset[j, :] = latent_functions[idx, :]
        
        if   noise_level == 0:
            dataset[j, :] += np.random.randn(dataset[j, :].shape[0])*0.2
        elif noise_level == 1:
            dataset[j, :] += np.random.randn(dataset[j, :].shape[0])*0.4
        elif noise_level == 2:
            dataset[j, :] += np.random.randn(dataset[j, :].shape[0])*0.6
        else:
            dataset[j, :] += np.random.randn(dataset[j, :].shape[0])*0.8

        ground_truth[j, idx] = 1

    return dataset, latent_functions, ground_truth

def cerp(*, y0: float, y1: float, a: float):
    g = (1 - np.cos(np.pi*(a - np.floor(a))))*0.5

    return (1 - g)*y0 + g*y1
