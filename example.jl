include("src/dataGenerator.jl")
include("src/SMIXS.jl")

function main()

    numberOfMeasurements = 128
    numberOfClusters     = 3
    numberOfSubjects     = 200

    data, groudnTruth, generatorFunctionVectors = generateData(numberOfMeasurements, numberOfSubjects, numberOfClusters, 4)

    labelVectors, clusterMeans, meanIterations = SMIXS(data, numberOfClusters, 3)

    p0 = plot(dpi = 200, legend = false, title = "Fited")
    p1 = plot(dpi = 200, legend = false, title = "Original")
    for i = 1:numberOfClusters

        plot!(p0, 1:numberOfMeasurements, clusterMeans[i,:],             linewidth = 2)
        plot!(p1, 1:numberOfMeasurements, generatorFunctionVectors[i,:], linewidth = 2)

    end

    display(plot(p0, p1))

end


main()
