module SpinTruncatedWigner

using LinearAlgebra, Random, SparseArrays, SpinModels, SciMLBase

export SpinHalf, spinHalfdown, spinHalfup, CoherentSpinState, NeelState, SpinProductState
export TWAParameters, TWAProblem, cTWADiscreteState, cTWAGaussianState
export ClusterBasis, lookupClusterOp, reverseLookup

abstract type AbstractTWAParameters end
function twaUpdate! end

# const MatrixTypes = Union{Matrix{Float64}, SparseMatrixCSC{Float64}}
# const SparseMatrixNonZeroEntries = Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}

# TODO Let the Q Vector also be sparse?


### CLUSTERBASIS

include("clusterBasis.jl")
include("structureConstants.jl")
include("spinstates.jl")
include("dTWA.jl")
include("cTWA.jl")


"""
    TWAParameters(Hamiltonian[, clusterbasis])

Precompute the necessary parameters for a TWA simulation. If clusterbasis is given, a cluster TWA will be prepared.
Otherwise a standard dTWA is prepared.
"""
function TWAParameters end

TWAParameters(H::SpinModels.Hamiltonian) = dTWAParameters(_derivativeMatrix(H)...)
function TWAParameters(H::SpinModels.Hamiltonian, cb::ClusterBasis)
    L,Q = _toclusterbasis(H, cb)
    return cTWAParameters(L, sparse(Q), cb)
end

"""
    TWAProblem([clustering,] Hamiltonian, ψ0, times)

Construct the DifferentialEquations' Problem for solving the TWA. If a clustering is specified, uses cTWA.
 - `clustering` is either a Vector of clusters or a `ClusterBasis`
 - `Hamiltonian` is either a `SpinModels.Hamiltonian` or an appropriate subtype of `AbstractTWAParameters`
 - `ψ0` is either a subtype of `SpinState`
"""
function TWAProblem end

TWAProblem(H, ψ0::AbstractSpinState, times; rng=Random.default_rng()) = TWAProblem(TWAParameters(H), classical(ψ0), times; rng)
TWAProblem(H::SpinModels.Hamiltonian, ψ0::Matrix, times; rng=Random.default_rng()) = TWAProblem(TWAParameters(H), ψ0, times; rng)

TWAProblem(clustering::Vector, H, ψ0, times; rng=Random.default_rng()) = TWAProblem(ClusterBasis(clustering), H, ψ0, times; rng)
TWAProblem(cb::ClusterBasis, H, ψ0::AbstractSpinState, times; rng=Random.default_rng()) = TWAProblem(cb, H, cTWAGaussianState(cb,ψ0), times; rng)
TWAProblem(cb::ClusterBasis, H, ψ0::Vector, times; rng=Random.default_rng()) = TWAProblem(cb, H, cTWAGaussianState(cb,ψ0), times; rng)
TWAProblem(cb::ClusterBasis, H::SpinModels.Hamiltonian, ψ0::AbstractCTWAState, times; rng=Random.default_rng()) = TWAProblem(TWAParameters(H,cb), ψ0, times; rng)

function TWAProblem(param::AbstractTWAParameters, ψ0, times; rng=Random.default_rng())
    problem = ODEProblem{true, SciMLBase.FullSpecialize}(twaUpdate!, ψ0, (0, maximum(times)), param; saveat=sort(times))
    ensemble = EnsembleProblem(problem;
	    prob_func = (prob, i, repeat) -> remake(prob; u0 = twaSample(prob.u0; rng)))
    return ensemble
end

end
