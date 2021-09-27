
using Printf
using Random
using Clustering
using Distributions
using GaussianMixtures

include("SMIXSSharedFunctions.jl")
include("fitSMIXS.jl")

function SMIXS(D, NCLUSTERS, NREPETITIONS = 50, initSeed = 1)
    Random.seed!(initSeed)

    K, R, Q = initPenaltyMatrix(size(D)[2])
    QQ      = Q'*Q
    K       = Symmetric(K)

    # PARAMETERS #
    NUMBER_OF_ITERATIONS = 50

    NSUBJECTS     = size(D)[1]
    NMEASUREMENTS = size(D)[2]

    bestResult = zeros(2)
    resultsZ   = BitArray(zeros(Int32, NSUBJECTS, NCLUSTERS))
    resultsCM  = zeros(Float64, NCLUSTERS, NMEASUREMENTS)
    iterations = 0

    seeds = round.(rand(1000)*10^7)

    for j = 1:NREPETITIONS
        SEED=Int32(seeds[j])
        Random.seed!(SEED)

        cMean, cVariance, cMixing, cAlpha = kMeansInitialization(D, NCLUSTERS)

        (cMean, cVariance, cAlpha, cMixing, Z, bicValue, likelihood, likelihoodArray) = fitSMIXS(D, K, R, Q, QQ, cMean, cVariance, cAlpha, cMixing, NCLUSTERS, NUMBER_OF_ITERATIONS)

        iterations += length(likelihoodArray)

        if (bestResult[1] < likelihood) || (j == 1)
            bestResult[1]  = likelihood
            bestResult[2]  = SEED
            resultsZ[:,:]  = BitArray(ZtoZHot(Z, NSUBJECTS, NCLUSTERS))
            resultsCM[:,:] = cMean

            println(cAlpha)
        end
    end

    return resultsZ, resultsCM, iterations/NREPETITIONS
end


function kMeansInitialization(D, NCLUSTERS)
    cVariance = zeros(NCLUSTERS)

    Res = kmeans(D', NCLUSTERS, maxiter=50)
    grp = v2oneHot(assignments(Res), NCLUSTERS)
    for k = 1:NCLUSTERS
        S = sum(grp[:,k])

        cVariance[k] = S < 2 ? (S < 1 ? 1 : mean(var(D[grp[:,k],:], dims=2))) : mean(var(D[grp[:,k],:], dims=1))
    end

    return Res.centers', cVariance, Res.counts./sum(Res.counts), ones(NCLUSTERS)
end
