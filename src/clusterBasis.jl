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

_toclusterbasis(::Val{:Hopp}, coeff, cb) = _toclusterbasis(Val(:XX),coeff/2,cb)+_toclusterbasis(Val(:YY),coeff/2,cb)
