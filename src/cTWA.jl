struct cTWAParameters{LT, QT} <: AbstractTWAParameters
    L::LT # linear couplings
    Q::QT # quadratic couplings
    clusterbasis::ClusterBasis
    structureconstants::Vector{Vector{SparseMatrixCSC{Float64, Int64}}} # clusterSize -> structure constants
    Lworkspace::LT
    function cTWAParameters(L::AbstractVector,Q::AbstractMatrix, cb::ClusterBasis)
        issparse(Q) || @info "Q is not sparse (type $(typeof(Q))). A sparse matrix is probably more performant."
        new{typeof(L),typeof(Q)}(L, Q, cb, _precompute_structureconstants(cb), similar(L))
    end
end

function twaUpdate!(ds, s, p::cTWAParameters, t)
    # L,Q = p.derivs
    # mul!(loc_deriv, Q, s)
    # loc_deriv += L
    loc_deriv = p.Lworkspace
    mul!(loc_deriv, p.Q, s)
    loc_deriv .*= 2
    loc_deriv .+= p.L
    cb = p.clusterbasis
    #for (clusterID, cluster) in enumerate(cb.clustering)
    @inbounds for clusterID in 1:length(cb.clustering)
        offset = cb.clusterOffsets[clusterID]
        lastIndex = clusterID==length(cb.clusterOffsets) ? length(cb) : cb.clusterOffsets[clusterID+1]

        structureConsts = p.structureconstants[length(cb.clustering[clusterID])]

        l = @view loc_deriv[offset+1:lastIndex]
        s2 = @view s[offset+1:lastIndex]
        # @show size(structureConsts[1]), size(l)
        for (i, stateIndex) in enumerate(offset+1:lastIndex)
            ds[stateIndex] = dot(l,structureConsts[i],s2)
        end
    end
    ds
end
