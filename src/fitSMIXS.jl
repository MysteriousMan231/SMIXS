using Optim
using Plots
using Printf
using Random
using Statistics
using SparseArrays
using Distributions
using LinearAlgebra
using DelimitedFiles

include("SMIXSSharedFunctions.jl")

"""
    fitModel(D, C, N, iM, iMc, NCLUSTERS=1, NUMBER_OF_ITERATIONS=100, PARAMETRIC=false)

Compute the semiparametric model fit on the longitudinal
data provided by the readData.jl function set.

# Arguments
- `D::Array{Float32,2}`: Array of measurements.
- `C::Array{Float32,2}`: Array of covariates.
- `N::Array{Float32,2}`: Incidence array containing information about missing measurements for individuals.
- `iM::Array{Int32,1}`: Array containing the number of measurements per subject.
- `iMc::Array{Int32,1}`: Array containing the progressive cumulative sums of array iM.
"""

function fitSMIXS(D, K, R, Q, QQ, cMeans, cVariance, cAlpha, cMixing, NCLUSTERS=1, NUMBER_OF_ITERATIONS=100)

    NSUBJECTS = size(D)[1]
    NMEASUREMENTS = size(D)[2]

    Z = zeros(Float32, NSUBJECTS, NCLUSTERS)
    L = []

    oldLikelihood = 0
    stoppage = false
    step = 1
    while step <= NUMBER_OF_ITERATIONS

        labelEstimation(D, Z, cMeans, cVariance, cMixing, NCLUSTERS, NSUBJECTS)
        clusterMixingProportionEstimation(cMixing, Z, NCLUSTERS, NSUBJECTS)
        clusterEstimationNonParametricCorrect(step, cMeans, cVariance, cAlpha, D, Z, K, R, Q, QQ, NCLUSTERS, NMEASUREMENTS, NSUBJECTS)
        oldLikelihood = logLikelihood(cMeans, cVariance, cAlpha, [], cMixing, D, Z, K, [], NSUBJECTS, NCLUSTERS, false)

        push!(L, oldLikelihood)

        b = 5
        x = collect(1:b) .- ceil(b/2)

        t = 0
        if length(L) >= b
            h = L[end - b + 1:end]

            max = maximum(L)
            min = minimum(L)

            h = (h .- minimum(L))./(max == min ? 1 : max - min)
            h = h .- mean(h)
            t = sum(h.*x)/sum(x.^2)
        end

        if (abs(t) < 10^(-3)) && (length(L) >= b)
            break
        end

        step += 1
    end

    modelBIC = BIC(3NCLUSTERS + NMEASUREMENTS*NCLUSTERS, NSUBJECTS*NMEASUREMENTS, oldLikelihood)
    return cMeans, cVariance, cAlpha, cMixing, Z, modelBIC, oldLikelihood, L
end

@inline function crossValidation(a, ZSub, wY, Q, QQ, R, W, D, NMEASUREMENTS, NSUBJECTS)
    A = 0

    iD,L = CHOL(QQ*a + R*W)
    SD = invdiag(a, Q, iD, L, NMEASUREMENTS)./W

    gk = (wY - a*Q*(L'\((L\(Q'*wY))./iD)))/W

    for j = 1:NSUBJECTS
        A += ZSub[j]*sum(((D[j,:] .- gk)./(1 .- SD.*ZSub[j])).^2)
    end

    return A
end

@inline function clusterEstimationNonParametricCorrect(step, cMeans, cVariance, cAlpha, D, Z, K, R, Q, QQ, NCLUSTERS, NMEASUREMENTS, NSUBJECTS)

    eps = (10e-200)^(2/size(D)[2])
    h   = 0.1
    P   = []

    for i=1:NCLUSTERS

        ZSub = Z[:,i]
        W    = sum(ZSub)
        Nk   = W*NMEASUREMENTS
        wY   = sum(((D.*ZSub)), dims=1)'

        c0 = crossValidation(cAlpha[i] + h, ZSub, wY, Q, QQ, R, W, D, NMEASUREMENTS, NSUBJECTS)
        c1 = crossValidation(cAlpha[i] - h, ZSub, wY, Q, QQ, R, W, D, NMEASUREMENTS, NSUBJECTS)
        c2 = crossValidation(cAlpha[i], ZSub, wY, Q, QQ, R, W, D, NMEASUREMENTS, NSUBJECTS)
        cAlpha[i] = clamp(cAlpha[i] - ((c0 - c2)/h)/((c0 - 2c2 + c1)/h^2), 1.0, 10^6)

        iD,L        = CHOL(QQ*cAlpha[i] + R*W)
        a2          = L'\((L\(Q'*wY))./iD)
        cMeans[i,:] = (wY - cAlpha[i]*Q*a2)/W

        cVariance[i] = (sum(((D .- cMeans[i,:]').^2).*ZSub) + cAlpha[i]*(cMeans[i,:]')*K*cMeans[i,:])/Nk
    end

    cVariance[cVariance .< eps] .= var(D)
end
