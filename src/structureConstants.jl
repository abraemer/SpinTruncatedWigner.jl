const LEVI_CIVITA = let temp = [zeros(3,3) for _ in 1:3]
    temp[1][2,3] = temp[2][3,1] = temp[3][1,2] = 1
    temp[1][3,2] = temp[2][1,3] = temp[3][2,1] = -1
    temp
end

const LEVI_CIVITA_ϵ = [0 3 -2; -3 0 1; 2 -1 0]

function _precompute_structureconstants(cb::ClusterBasis)
    maxcluster = maximum(length.(cb.clustering))
    consts = Vector{SparseMatrixCSC{Float64, Int64}}[]
    for k in 1:maxcluster
        push!(consts, sparse.(_full_structure_constants(k)))
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
