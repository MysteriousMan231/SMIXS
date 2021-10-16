using Printf
using Random
using Noise
using Distributions
using LinearAlgebra

function rndNum(N)
    return Int32(round(rand()*N + 0.5))
end

function cerp(y0, y1, a)
    g = (1 - cos(pi*(a - floor(a))))*0.5

    return (1 - g)*y0 + g*y1
end

function generateData(NMEASUREMENTS, NSUBJECTS, NCLUSTERS, seed = 1, noiselvl = 0.0)
    Random.seed!(seed)

    Nf01 = map(x -> Normal(0,x), ones(Float64, NCLUSTERS).*0.2)
    Nf02 = map(x -> Normal(0,x), ones(Float64, NCLUSTERS).*0.4)
    Nf03 = map(x -> Normal(0,x), ones(Float64, NCLUSTERS).*0.6)
    Nf04 = map(x -> Normal(0,x), ones(Float64, NCLUSTERS).*0.8)

    P = clamp.(rand(NCLUSTERS), 0.1, 1)
    P = P./sum(P)
    iP = sortperm(P)
    P = cumsum(P[iP])
    GT = zeros(Int32, NSUBJECTS, NCLUSTERS)

    D = zeros(Float64, NSUBJECTS, NMEASUREMENTS)
    N = BitArray(ones(Int32, NSUBJECTS, NMEASUREMENTS))
    iM = ones(Int32, NSUBJECTS).*NMEASUREMENTS
    iMc = zeros(Int32, NSUBJECTS)
    iMc[2:end] = collect(1:NSUBJECTS - 1).*NMEASUREMENTS

    kF = zeros(Float64, NCLUSTERS, NMEASUREMENTS)

    s = 8
    var = 0.5
    x = s:s:NMEASUREMENTS
    y = rand(Normal(0, var), length(x) + 1)
    Ny = Normal(0, 5) #0.5

    top = 10.0 #0.45
    bot = 0.0 #0.2

    for i = 1:NCLUSTERS

        y0 = copy(y)
        v0 = rand(Ny, length(y))
        v0[v0 .< 0] = clamp.(v0[v0 .< 0], -top, -bot)
        v0[v0 .> 0] = clamp.(v0[v0 .> 0],  bot, top)
        y0 = y0 + v0

        c = 1
        for j in 1:NMEASUREMENTS
            kF[i,j] += cerp(y0[c], y0[c + 1], j/s - 1/s)

            if (j % s) == 0
                c += 1
            end
        end

    end

    minkF = findmin(kF)[1]
    maxkF = findmax(kF)[1]

    kF = (kF .- minkF)./(maxkF - minkF)

    #Random.seed!(noiseSeed)
    x = collect(1:NMEASUREMENTS)
    for j = 1:NSUBJECTS
        idx = iP[sum(P .< rand()) + 1]

        D[j,:] = kF[idx,:]
        D[j,:] += (noiselvl == 1) ? rand(Nf01[idx], NMEASUREMENTS) : (noiselvl == 2 ? rand(Nf02[idx], NMEASUREMENTS) : (noiselvl == 3 ? rand(Nf03[idx], NMEASUREMENTS) : rand(Nf04[idx], NMEASUREMENTS)))

        GT[j,idx] = 1
    end

    #return D,N,iM,iMc,BitArray(GT),kF
    return D, BitArray(GT), kF
end
