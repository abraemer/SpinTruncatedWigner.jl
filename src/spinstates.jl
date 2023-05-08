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

twaSample(state::Vector, N) = twaSample(reshape(state,:,1),N)
function twaSample(state::Matrix{T}, N) where T <: Real
    orthogonal = nullspace(Matrix(state'))
    ret = Matrix{T}(undef, length(state), N)
    for i in 1:N
        ret[:, i] = vec .+ sum(rand([-1, 1], 1, 2)  .* orthogonal, dims=2)
    end
    return vec(ret)
end

twaSample(state::Vector) = twaSample(reshape(state, 3, :))
function twaSample(state::Matrix{T}) where T <: Real
    ret = similar(state)
    for (i, vec) in enumerate(eachcol(state))
        orthogonal = nullspace(Matrix(vec'))
        ret[:, i] = vec .+ sum(rand([-1, 1], 1, 2)  .* orthogonal, dims=2)
    end

    return vec(ret)
end

abstract type AbstractSpinState end

function classical end
function quantum end
function twaSample end

struct SpinHalf <: AbstractSpinState
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

twaSample(ss::SpinHalf) = twaSample(classical(ss))

struct CoherentSpinState <: AbstractSpinState
    singleSpin::SpinHalf
    N::Integer
end

function classical(css::CoherentSpinState)
    single = classical(css.singleSpin)
    hcat([copy(single) for _ in 1:css.N]...)
end

quantum(css::CoherentSpinState) = expkron(quantum(css.singleSpin), css.N)

twaSample(css::CoherentSpinState) = twaSample(classical(css))

struct SpinProductState <: AbstractSpinState
    spinstates::Vector{SpinHalf}
end

function neelstate(N, axis=:z)
    spinup = spinHalfup(axis)
    spindown = spinHalfdown(axis)
    return SpinProductState([iseven(i) ? spindown : spinup for i in 1:N])
end

function classical(sps::SpinProductState)
    reduce(hcat, classical.(sps.spinstates))
end

quantum(sps::SpinProductState) = reduce(kron, quantum.(sps.spinstates))

twaSample(sps::SpinProductState) = twaSample(classical(sps))

convert(::Type{SpinProductState}, ss::SpinHalf) = SpinProductState([ss])
convert(::Type{SpinProductState}, css::CoherentSpinState) = SpinProductState(fill(css.singleSpin,css.N))

struct cTWAGaussianState
    # all per cluster!
    μs::Vector{Vector{Float64}} # expectation values
    σs::Vector{Vector{Float64}} # variance eigenvalues
    Vs::Vector{Matrix{Float64}} # transformation Σ = V σ V^T, note that V may not be quadratic
end


cTWAGaussianState(cb::ClusterBasis, spinstate; cutoff=1e-10) = cTWAGaussianState(cb, convert(SpinProductState, spinstate); cutoff)
cTWAGaussianState(cb::ClusterBasis, sps::SpinProductState; cutoff=1e-10) = cTWAGaussianState(cb, classical.(sps.spinstates); cutoff=1e-10)
function cTWAGaussianState(cb::ClusterBasis, firstMoments::Vector; cutoff=1e-10)
    #firstMoments =  #[spinID][axis]
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
        push!(σs, sqrt.(eigenDecomp.values[nonZeros] / 2))
        push!(Vs, eigenDecomp.vectors[:, nonZeros])
    end
    return cTWAGaussianState(μs, σs, Vs)
end

function twaSample(state::cTWAGaussianState)
    percluster = Vector{Float64}[]
    for (μ,σ,V) in zip(state.μs, state.σs, state.Vs)
        sample = randn(length(σ))
        push!(percluster, μ .+ V * (σ .* sample))
    end
    return reduce(vcat, percluster)
end
