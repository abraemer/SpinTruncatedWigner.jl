module SpinTruncatedWigner

using LinearAlgebra, SparseArrays, SpinModels

export dTWAParameters, cTWAGaussianState, cTWAParameters, ClusterBasis, twaSample

abstract type TWAParameters end
function twaUpdate! end
# const MatrixTypes = Union{Matrix{Float64}, SparseMatrixCSC{Float64}}
# const SparseMatrixNonZeroEntries = Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}

# TODO Let the Q Vector also be sparse?
struct dTWAParameters{LT, QT} <: TWAParameters
    L::LT # linear couplings
    Q::QT # quadratic couplings
    dTWAParameters(L::AbstractVector,Q::AbstractMatrix) = new{typeof(L),typeof(Q)}(L,Q)
end

#TODO also reduce to upper triangular matrix by symmetry
# cTWAParameters(L, Q::AbstractVector{<:AbstractSparseMatrix{Float64}}) = cTWAParameters(L, vec(findnz.(Q)))

const LEVI_CIVITA = let temp = [zeros(3,3) for _ in 1:3]
    temp[1][2,3] = temp[2][3,1] = temp[3][1,2] = 1
    temp[1][3,2] = temp[2][1,3] = temp[3][2,1] = -1
    temp
end

function twaUpdate!(ds, s, p::dTWAParameters, t)
    # L,Q = p.derivs
    # mul!(loc_deriv, Q, s)
    # loc_deriv += L
    loc_deriv = 2p.Q*s
    loc_deriv += p.L
    for site in 1:length(ds)÷3
        l = @view loc_deriv[3site-2:3site]
        s2 = @view s[3site-2:3site]
        ds[3site-2] =  2dot(l, LEVI_CIVITA[1], s2)
        ds[3site-1] =  2dot(l, LEVI_CIVITA[2], s2)
        ds[3site-0] =  2dot(l, LEVI_CIVITA[3], s2)
    end
    ds
end

dTWAParameters(H::SpinModels.Hamiltonian) = dTWAParameters(_derivativeMatrix(H)...)

_derivativeMatrix(H::SpinModels.Hamiltonian) = sum(_derivativeMatrix(Val(term.kind),term.coefficient) for term in SpinModels._termlist(H))

for (sym, inds) in ((:X, :Xinds),
                    (:Y, :Yinds),
                    (:Z, :Zinds))
    @eval function _derivativeMatrix(::$(typeof(Val(sym))), coeff)
        N = size(coeff,1)
        Xinds = 1:3:3N-2
        Yinds = 2:3:3N-1
        Zinds = 3:3:3N

        L = zeros(3N)
        Q = zeros(3N, 3N)
        L[$inds] .= coeff
        return [L,Q]
    end
end

for (sym, inds) in ((:XX, :Xinds),
                    (:YY, :Yinds),
                    (:ZZ, :Zinds))
    @eval function _derivativeMatrix(::$(typeof(Val(sym))), coeff)
        N = size(coeff,1)
        Xinds = 1:3:3N-2
        Yinds = 2:3:3N-1
        Zinds = 3:3:3N

        L = zeros(3N)
        Q = zeros(3N, 3N)
        Q[$inds,$inds] .= coeff
        return [L,Q]
    end
end

_derivativeMatrix(::Val{:Hopp}, coeff) = _derivativeMatrix(Val(:XX),coeff)+_derivativeMatrix(Val(:YY),coeff)

twaSample(state::Vector, N) = twaSample(reshape(state,:,1),N)
function twaSample(state::Matrix{T}, N) where T <: Real
    orthogonal = nullspace(Matrix(state'))
    ret = Matrix{T}(undef, length(state), N)
    for i in 1:N
        ret[:, i] = state .+ sum(rand([-1, 1], 1, 2)  .* orthogonal, dims=2)
    end
    return vec(ret)
end

### CLUSTERBASIS

struct ClusterBasis
    clustering::Vector{Vector{Int64}} # clusterID -> [spinIDs]
    clusters::Vector{Int64} # spinID -> clusterID
    multiplicators::Vector{Int64} # spinID -> multiplicator for the spin (based on it's position within the cluster)
    clusterOffsets::Vector{Int64} # clusterID -> offset in cluster operator Basis
                                    # note that the offsets are sorted in ascending order
    length::Int64
    # singleBases TODO implement general Bases
    # TODO logic when there are different bases within the SAME cluster
    function ClusterBasis(clustering::Vector{Vector{Int64}})# TODO bases
        totalEntries = sum(map(length, clustering))
        numClusters = length(clustering)

        clusters = Vector{Int64}(undef, totalEntries)
        multiplicators = Vector{Int64}(undef, totalEntries)
        clusterOffsets = Vector{Int64}(undef, numClusters)

        currentOffset = 0
        for (clusterID, cluster) in enumerate(clustering)
            clusterOffsets[clusterID] = currentOffset
            multiplicator = 1
            for spinID in cluster
                clusters[spinID] = clusterID
                multiplicators[spinID] = multiplicator
                multiplicator *= 4 # length(basis[i])
            end
            currentOffset += multiplicator - 1 #only first cluster has the identity
        end

        new(clustering, clusters, multiplicators, clusterOffsets, currentOffset) # length(clusterBasis) is the last valid opIndex

    end
end

"Length of Cluster basis (amount of cluster operators)"
Base.length(cb::ClusterBasis) = cb.length

sameCluster(cb::ClusterBasis, position1, position2) = cb.clusters[position1] == cb.clusters[position2]

function lookupClusterOp(cb::ClusterBasis, posAndDirs...)
    # TODO should this return a list of numbers corresponding to a product of cluster Ops? could simplify code on other places...
    if length(posAndDirs) == 0 || all(posAndDir[2] == 0 for posAndDir in posAndDirs)
        return 0 # or error?
    end
    clusterID = cb.clusters[posAndDirs[1][1]]
    return cb.clusterOffsets[clusterID] + lookupClusterOpWithinCluster(cb, posAndDirs...)
end

function lookupClusterOpWithinCluster(cb::ClusterBasis, posAndDirs...)
    if length(posAndDirs) == 0
        return 0
    end
    clusterID = cb.clusters[posAndDirs[1][1]]
    total = 0
    for (position, direction) in posAndDirs
        # could also check wether direction has a sensible value - worth it?
        if clusterID != cb.clusters[position]
            error("Particle $position not in the same cluster as the others!")
        end
        total += cb.multiplicators[position] * direction
    end
    total
end

function reverseLookup(cb::ClusterBasis, clusterOperatorID)
    # find clusterID
    range = searchsorted(cb.clusterOffsets, clusterOperatorID-1)
    # -1 cause the operator at clusterOffset actually belongs to the PREVIOUS cluster
    clusterID = range.stop

    ret = Vector{Tuple{Int64, Int64}}(undef, 0)
    restValue = clusterOperatorID - cb.clusterOffsets[clusterID]
    cluster = cb.clustering[clusterID]
    for pos in length(cluster):-1:1
        spinID = cluster[pos]
        multiplicator = cb.multiplicators[spinID]
        direction, restValue2 = divrem(restValue, multiplicator)
        restValue = restValue2
        if direction > 0

            push!(ret, (spinID, direction))
        end
    end
    ret
end

_toclusterbasis(H::SpinModels.Hamiltonian,cb::ClusterBasis) = sum(_toclusterbasis(Val(term.kind),term.coefficient, cb) for term in SpinModels._termlist(H))

for (sym, dir) in ((:X, 1),
                    (:Y, 2),
                    (:Z, 3))
    @eval function _toclusterbasis(::$(typeof(Val(sym))), coeff, cb)
        N = size(coeff,1)

        L = zeros(length(cb))
        Q = zeros(length(cb), length(cb))
        for (spin,h) in enumerate(coeff)
            L[lookupClusterOp(cb,(spin,$dir))] = h
        end
        return [L,Q]
    end
end

for (sym, dir) in ((:XX, 1),
                    (:YY, 2),
                    (:ZZ, 3))
    @eval function _toclusterbasis(::$(typeof(Val(sym))), coeff, cb)
        N = size(coeff,1)

        L = zeros(length(cb))
        Q = zeros(length(cb), length(cb))
        for I in CartesianIndices(coeff)
            # TODO symmetry
            (I[1]==I[2] || coeff[I] == 0) && continue
            if sameCluster(cb,I[1],I[2])
                L[lookupClusterOp(cb,(I[1],$dir),(I[2],$dir))] = coeff[I]
            else
                Q[lookupClusterOp(cb, (I[1],$dir)),lookupClusterOp(cb, (I[2],$dir))] = coeff[I]
            end
        end
        return [L,Q]
    end
end

_toclusterbasis(::Val{:Hopp}, coeff, cb) = _toclusterbasis(Val(:XX),coeff,cb)+_toclusterbasis(Val(:YY),coeff,cb)

struct cTWAParameters{LT, QT} <: TWAParameters
    L::LT # linear couplings
    Q::QT # quadratic couplings
    clusterbasis::ClusterBasis
    structureconstants::Vector{Vector{Matrix{Float64}}} # clusterSize -> structure constants
    cTWAParameters(L::AbstractVector,Q::AbstractMatrix, cb::ClusterBasis) = new{typeof(L),typeof(Q)}(L,Q, cb, _precompute_structureconstants(cb))
end

#TODO also reduce to upper triangular matrix by symmetry
# cTWAParameters(L, Q::AbstractVector{<:AbstractSparseMatrix{Float64}}) = cTWAParameters(L, vec(findnz.(Q)))

function twaUpdate!(ds, s, p::cTWAParameters, t)
    # L,Q = p.derivs
    # mul!(loc_deriv, Q, s)
    # loc_deriv += L
    loc_deriv = 2p.Q*s
    loc_deriv += p.L
    cb = p.clusterbasis
    for (clusterID, cluster) in enumerate(cb.clustering)
        offset = cb.clusterOffsets[clusterID]
        lastIndex = clusterID==length(cb.clusterOffsets) ? length(cb) : cb.clusterOffsets[clusterID+1]

        structureConsts = p.structureconstants[length(cluster)]

        l = @view loc_deriv[offset+1:lastIndex]
        s2 = @view s[offset+1:lastIndex]
        # @show size(structureConsts[1]), size(l)
        for (i, stateIndex) in enumerate(offset+1:lastIndex)
            ds[stateIndex] = dot(l,structureConsts[i],s2)
        end
    end
    ds
end

function _precompute_structureconstants(cb::ClusterBasis)
    maxcluster = maximum(length.(cb.clustering))
    consts = Vector{Matrix{Float64}}[]
    for k in 1:maxcluster
        push!(consts, _full_structure_constants(k))
    end
    return consts
end

function _full_structure_constants(clustersize)
    basissize = 4^clustersize-1
    res = [zeros(basissize,basissize) for _ in 1:basissize]
    for α in 1:basissize
        αstring = _ind_to_string(α)
        for β in α+1:basissize
            βstring = _ind_to_string(β)
            γstring, val = computeStructureConstant(αstring,βstring)
            val == 0 && continue
            γ = _string_to_ind(γstring)
            res[α][β,γ] = val
            res[β][α,γ] = -val
        end
    end
    return res
end

function computeStructureConstant(αstring, βstring)
	l = max(length(αstring), length(βstring))
	αstring = [αstring;zeros(Int,l-length(αstring))]
	βstring = [βstring;zeros(Int,l-length(βstring))]
    parity = 1
    odds = 0
    γstring = Vector{Int64}(undef, length(αstring))
    for (i, (α, β)) in enumerate(zip(αstring, βstring))
        val, par = singleStructureConstant(α, β)
        γstring[i] = val
        if par != 0
            parity *= par
            odds += 1
        end
    end
    if isodd(odds)
        return γstring, parity*2 #TODO structure constant is 2?
    else
        return γstring, 0
    end
end

function singleStructureConstant(a,b)
    if a == 0 || b == 0
        return a + b, 0
    else
        val = LEVI_CIVITA_ϵ[a,b]
        return abs(val), sign(val)
    end
end

const LEVI_CIVITA_ϵ = [0 3 -2; -3 0 1; 2 -1 0]

function _ind_to_string(x)
    str = Int[]
    while x > 0
        x,dir = divrem(x,4)
        push!(str, dir)
    end
    return str
end

function _string_to_ind(xstring)
    x = 0
    mult = 1
    for dir in xstring
        x += mult*dir
        mult *= 4
    end
    return x
end

function cTWAParameters(H::SpinModels.Hamiltonian, cb::ClusterBasis)
    cTWAParameters(_toclusterbasis(H, cb)..., cb)
end

const Pauli = (;
    x = [0 1; 1 0],
    y = [0 -im; im 0],
    z = [1 0; 0 -1],
    xyz = [[0 1; 1 0],[0 -im; im 0],[1 0; 0 -1]])

"""
expBinaryOp(object, op, n)

Compute op(object, ..., object) = object op object op ... op object via exponentiation by squaring.
op needs to be associative for this to be correct!

expBinaryOp(... , 0) = one(object)
"""
function expBinaryOp(object, op::Function, n::Integer)
    if n < 0
        error("n must be non-negative!")
    elseif n == 0
        return one(object)
    end
    result = nothing
    temp = object
    while n > 0
        if n % 2 == 1
            result = result===nothing ? temp : op(result, temp)
        end
        temp = op(temp, temp)
        n >>= 1
    end
    result
end

expkron(matrix, n::Integer) = expBinaryOp(matrix, kron, n)

function cartesianToSpherical(x,y,z)
    r = sqrt(x^2 + y^2 + z^2)
    [acos(z/r), atan(y,x), r]
end

# x = r sin(θ)cos(ϕ), y = r sin(θ)sin(ϕ), z = r cos(θ)
function sphericalToCartesian(θ, ϕ, r=1.0)
    [r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ)]
end

struct SpinHalf #<: SpinState
    # |ψ⟩=cos(θ/2)|↑⟩ + exp(iϕ)sin(θ/2)|↓⟩
    θ::Number
    ϕ::Number
end

function spinHalfup(symbol = :z)
    if symbol == :x
        return SpinHalf(π/2, 0)
    elseif symbol == :y
        return SpinHalf(π/2, π/2)
    elseif symbol == :z
        return SpinHalf(0,0)
    else
        error("What dis axis? $symbol")
    end
end

function spinHalfdown(symbol = :z)
    if symbol == :x
        return SpinHalf(π/2, π)
    elseif symbol == :y
        return SpinHalf(π/2, 3π/2)
    elseif symbol == :z
        return SpinHalf(π,0)
    else
        error("What dis axis? $symbol")
    end
end

classical(ss::SpinHalf) = sphericalToCartesian(ss.θ, ss.ϕ)
quantum(ss::SpinHalf) = [cos(ss.θ/2), sin(ss.θ/2)*exp(im*ss.ϕ)]

struct CoherentSpinState
    singleSpin::SpinHalf
    N::Integer
end

function classical(css::CoherentSpinState)
    single = classical(css.singleSpin)
    hcat([copy(single) for _ in 1:css.N]...)
end

quantum(css::CoherentSpinState) = expkron(quantum(css.singleSpin), css.N)

struct SpinProductState
    spinstates::Vector{SpinHalf}
end

function classical(sps::SpinProductState)
    reduce(hcat, classical.(sps.spinstates))
end

quantum(sps::SpinProductState) = reduce(kron, quantum.(sps.spinstates))

function neelstate(N, axis=:z)
    spinup = spinHalfup(axis)
    spindown = spinHalfup(axis)
    return SpinProductState([iseven(i) ? spindown : spinup for i in 1:N])
end

struct cTWAGaussianState
    # all per cluster!
    μs::Vector{Vector{Float64}} # expectation values
    σs::Vector{Vector{Float64}} # variance eigenvalues
    Vs::Vector{Matrix{Float64}} # transformation Σ = V σ V^T, note that V may not be quadratic
end

# TODO generalize to general bases...
function cTWAGaussianState(cb::ClusterBasis, sps::SpinProductState; cutoff=1e-10)
    firstMoments = classical.(sps.spinstates) #[spinID][axis]
    # raw second moments [spinID][i,j] -> ⟨σᵢσⱼ⟩
    secondMoments = [I(3) + im*sum(LEVI_CIVITA .* fM) for fM in firstMoments]
    baseLength = length(cb)
    #spinOpLists = [reverseLookup(cb, opIndex) for opIndex in 1:baseLength]

    μs = Vector{Float64}[]
    σs = Vector{Float64}[]
    Vs = Matrix{Float64}[]
    for (clusterID, cluster) in enumerate(cb.clustering)
        clusterLastIndex = clusterID == length(cb.clustering) ? length(cb) : cb.clusterOffsets[clusterID+1]
        clusterLength = clusterLastIndex-cb.clusterOffsets[clusterID]

        μ = zeros(clusterLength)
        for opIndex1 in 1:clusterLength
            opString1 = _ind_to_string(opIndex1)
            # ⟨X_α⟩ = Π_i ⟨X_αᵢ⟩
            μ[opIndex1] = reduce(*, iszero(dir) ? 1.0 : firstMoments[spinID][dir] for (spinID,dir) in zip(cb.clustering[clusterID], opString1); init=1.0)
        end
        push!(μs, μ)

        Σ = zeros(clusterLength,clusterLength)
        for opIndex1 in 1:clusterLength
            opString1 = _ind_to_string(opIndex1)
            opString1Iter = Iterators.flatten((opString1, Iterators.repeated(0)))
            for opIndex2 in opIndex1:clusterLength
                opString2 = _ind_to_string(opIndex2)
                # Σ_ij = ⟨XᵢXⱼ + XⱼXᵢ⟩ - 2μᵢμⱼ
                # ⟨XᵢXⱼ + XⱼXᵢ⟩ = 2 Re⟨XᵢXⱼ⟩
                cov = complex(1.0)
                for (spinID, dir1, dir2) in zip(cb.clustering[clusterID], opString1Iter, opString2)
                    if dir1 != 0 && dir2 != 0
                        cov *= secondMoments[spinID][dir1,dir2]
                    elseif dir1 != 0
                        cov *= firstMoments[spinID][dir1]
                    elseif dir2 != 0
                        cov *= firstMoments[spinID][dir2]
                    end
                end
                Σ[opIndex1,opIndex2] = Σ[opIndex2,opIndex1] = 2real(cov)-2μ[opIndex1]*μ[opIndex2]
            end
        end
        eigenDecomp = eigen(Symmetric(Σ))
        nonZeros = @. eigenDecomp.values .> cutoff
        push!(σs, sqrt.(eigenDecomp.values[nonZeros]))
        push!(Vs, eigenDecomp.vectors[:, nonZeros])
    end
    return cTWAGaussianState(μs, σs, Vs)
end

function cTWAGaussianState(cb::ClusterBasis, css::CoherentSpinState)
    cTWAGaussianState(cb, SpinProductState(fill(css.singleSpin,css.N)))
end

function twaSample(state::cTWAGaussianState)
    percluster = Vector{Float64}[]
    for (μ,σ,V) in zip(state.μs, state.σs, state.Vs)
        sample = randn(length(σ))
        push!(percluster, μ .+ V * (σ .* sample))
    end
    return reduce(vcat, percluster)
end
end
