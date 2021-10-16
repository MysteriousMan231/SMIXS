import numpy as np

def generate_dataset(*,
    number_of_measurements: int = None,
    number_of_subjects:     int = None,
    number_of_clusters:     int = None,
    seed:        int = None,
    noise_level: int = None):

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
    P  = P[iP]

    dataset          = np.zeros(shape = (number_of_subjects, number_of_measurements), dtype = np.float32)
    ground_truth     = np.zeros(shape = (number_of_subjects, number_of_clusters),     dtype = np.float32)
    latent_functions = np.zeros(shape = (number_of_clusters, number_of_measurements), dtype = np.float32) 

    # Perlin noise function generator parameters
    step     = 8
    variance = 0.5
    
    x = range(step, number_of_measurements, step)
    y = np.random.randn(len(x) + 1)*np.sqrt(variance)

    top = 10.0
    bot =  0.0

    for i in range(0, number_of_clusters):
        