using ImageFiltering
using SparseArrays

#####################################################
#############     SUPPORT FUNCTIONS     #############
#####################################################

@inline function v2oneHot(vector, N)

    Z = BitArray(zeros(Int32, length(vector), N))

    for i = 1:length(vector)
        Z[i, vector[i]] = 1
    end

    return Z
end

@inline function ZtoAssignment(Z, NCLUSTERS)
    return sum(Z.*collect(1:NCLUSTERS)',dims=2)[:]
end

@inline function ZtoZHot(Z, NSUBJECTS, NCLUSTERS)

    ZHot = BitArray(zeros(Int32, NSUBJECTS, NCLUSTERS))
    for i = 1:NSUBJECTS
        ZHot[i, findmax(Z[i,:])[2]] = 1
    end

    return ZHot
end


@inline function indexToRow(a, L)
    t = Int32.(a[:].*collect(1:L))

    return t[t .!= 0]
end

@inline function BIC(k, n, lL)
    return k*log(n) - 2lL
end

@inline function rnd(N)
    return Int32(round(rand()*N + 0.5))
end

###################################################
#############    FAST DECOMPOSITION   #############
###################################################

# Identity minus diagonal
function invdiag(alpha, Q, D, L, NMEASUREMENTS)

    N = NMEASUREMENTS - 2
    iM = zeros(N, N)

    for i = N:-1:1

        iM[i,i] = 1/D[i] - ((i + 1) <= N ? L[i+1,i]*iM[i,i+1] : 0) - ((i + 2) <= N ? L[i+2,i]*iM[i,i+2] : 0)
        if i - 1 > 0
            iM[i-1,i] = -L[i,i-1]*iM[i,i] - (((i + 1) <= N) ? L[i+1,i-1]*iM[i,i+1] : 0)
        end
        if i - 2 > 0
            iM[i-2,i] = -L[i-1,i-2]*iM[i-1,i] - L[i,i-2]*iM[i,i]
        end
    end

    diagonal = zeros(NMEASUREMENTS)
    for i = 1:NMEASUREMENTS
        T0 = (i - 2) > 0 ? iM[i-2,i-2]*Q[i,i-2]*Q[i,i-2] : 0
        T1 = ((i - 1) > 0) && (i <= N + 1) ? iM[i-1,i-1]*Q[i,i-1]*Q[i,i-1] : 0
        T2 = (i <= N) ? iM[i,i]*Q[i,i]*Q[i,i] : 0

        T3 = ((i - 2) > 0) && (i <= N + 1) ? 2*iM[i-2,i-1]*Q[i,i-2]*Q[i,i-1] : 0
        T4 = ((i - 2) > 0) && (i <= N) ? 2*iM[i-2,i]*Q[i,i-2]*Q[i,i] : 0
        T5 = ((i - 1) > 0) && (i <= N) ? 2*iM[i-1,i]*Q[i,i-1]*Q[i,i] : 0

        diagonal[i] = 1 - (T0+T1+T2+T3+T4+T5)*alpha
    end

    return diagonal
end

@inline function CHOL(A)
    N = size(A,1)
    D = zeros(N,1)
    #L = diagm(ones(N))
    L = spzeros(N,N)
    L[diagind(N,N)] .= 1

    for j = 1:N
        D[j] += A[j,j]
        for k in clamp(j - 2,1,j - 2):(j - 1)
            D[j] = D[j] - L[j,k]^2*D[k]
        end

        if (j + 1) <= N
            L[j + 1,j] = A[j + 1,j]/D[j]

            if (j - 1) > 0
                L[j + 1, j] = L[j + 1, j] - L[j + 1, j - 1]*L[j, j - 1]*D[j - 1]/D[j]
            end
        end
        if (j + 2) <= N
            L[j + 2, j] = A[j + 2,j]/D[j]
        end
    end

    return D,L
end

##################################################
#############     CORE FUNCTIONS     #############
##################################################

@inline function labelEstimation(D, Z, cluster_means, cluster_variance, mixing_proportions, NCLUSTERS, NSUBJECTS)

    NMEASUREMENTS = length(cluster_means[1,:])
    T1 = log.(mixing_proportions)
    T2 = -NMEASUREMENTS.*0.5.*log.(cluster_variance)

    for i = 1:NSUBJECTS
        T0 = -0.5.*sum(((cluster_means .- D[i,:]').^2)./cluster_variance, dims=2)
        T = T0+T1+T2

        c = findmax(T)[1]
        T = exp.(T .- c)
        Z[i,:] = T./sum(T)
    end

    if true
        nSubjectsPerCluster = sum(Z, dims=1)
        eps = 1

        for k = 1:NCLUSTERS
            if nSubjectsPerCluster[k] >= eps
                continue
            end

            clusterIdx = indexToRow(nSubjectsPerCluster .>= (eps + 1), NCLUSTERS)
            subjectIdx = indexToRow(round.(sum(Z[:,clusterIdx], dims=2)), NSUBJECTS)

            T = zeros(length(subjectIdx))

            for i = 1:length(subjectIdx)
                s = subjectIdx[i]

                dMean = cluster_means[k,:]
                dS = cluster_variance[k]

                T0 = -0.5*sum(((dMean - D[s,:]).^2)./dS)
                T1 = log(mixing_proportions[k])
                T2 = -0.5*sum(log.(dS))

                T[i] = T0 + T1 + T2
            end

            c = subjectIdx[findmax(T)[2]]
            Z[c,:] .= 0
            Z[c, k] = 1

            nSubjectsPerCluster = sum(Z, dims=1)
        end
    end

end

@inline function initPenaltyMatrix(NMEASUREMENTS)

    t = zeros(Float64, NMEASUREMENTS)
    for j in range(1, length=NMEASUREMENTS)
        t[j] = j
    end
    h = t[2:end] - t[1:end-1]

    Q = zeros(Float64, NMEASUREMENTS, NMEASUREMENTS - 2)
    R = zeros(Float64, NMEASUREMENTS - 2, NMEASUREMENTS - 2)

    Q[1:NMEASUREMENTS + 1:end - 2] = h[1:end - 1].^(-1)
    Q[2:NMEASUREMENTS + 1:end - 1] = -(h[1:end - 1].^(-1) + h[2:end].^(-1))
    Q[3:NMEASUREMENTS + 1:end]     = h[2:end].^(-1)

    R[NMEASUREMENTS - 1:NMEASUREMENTS - 1:end - 1] = h[2:end - 1]*(1/6)
    R[2:NMEASUREMENTS - 1:end - NMEASUREMENTS + 2] = R[NMEASUREMENTS - 1:NMEASUREMENTS - 1:end - 1]
    R[1:NMEASUREMENTS - 1:end] = (h[1:end - 1] + h[2:end])*(1/3)

    K = Q*inv(R)*Q'

    R = sparse(R)
    Q = sparse(Q)
    dropzeros!(R)
    dropzeros!(Q)

    return K, R, Q

end

@inline function logLikelihood(cluster_means, cluster_variance, cluster_alpha, cluster_covariates, mixing_proportions, D, Z, K, C, NSUBJECTS, NCLUSTERS, FLAG)

    A = 0.0
    R = zeros(Float64, NCLUSTERS, NSUBJECTS)
    for k = 1:NCLUSTERS
        e0 = -0.5.*sum((D .- cluster_means[k,:]').^2, dims=2)./cluster_variance[k]
        e1 = (2pi*cluster_variance[k])^(-0.5*size(D, 2))

        R[k,:] = e0 .+ (log(e1) + log(mixing_proportions[k]))

        A = A - (cluster_alpha[k] != 0.0 ?
        (cluster_alpha[k]/(2cluster_variance[k]))*cluster_means[k,:]'*K*cluster_means[k,:] :
        0.0)
    end

    M = maximum(R, dims=1)
    R = log.(sum(exp.(R .- M), dims=1)) + M

    return A + sum(R)
end

@inline function clusterMixingProportionEstimation(mixing_proportions, Z, NCLUSTERS, NSUBJECTS)
    mixing_proportions = sum(Z, dims=1)./NSUBJECTS
    mixing_proportions = mixing_proportions/sum(mixing_proportions)
end
