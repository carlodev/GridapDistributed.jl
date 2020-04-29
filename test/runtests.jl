module GridapDistributedTests

using Test

@time @testset "CartesianDiscreteModels" begin include("CartesianDiscreteModelsTests.jl") end

@time @testset "DistributedTriangulations" begin include("DistributedTriangulationsTests.jl") end

@time @testset "DistributedCellQuadratures" begin include("DistributedCellQuadratures.jl") end

@time @testset "DistributedFESpaces" begin include("DistributedFESpacesTests.jl") end

@time @testset "GloballyAddressableArrays" begin include("GloballyAddressableArraysTests.jl") end

@time @testset "SparseMatrixAssemblers" begin include("SparseMatrixAssemblersTests.jl") end

end # module
