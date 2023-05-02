struct dTWAParameters{LT, QT} <: AbstractTWAParameters
    L::LT # linear couplings
    Q::QT # quadratic couplings
    Lworkspace::LT # workspace so twaUpdate! does not allocate
    # TODO what happens on GPU?
    dTWAParameters(L::AbstractVector,Q::AbstractMatrix) = new{typeof(L),typeof(Q)}(L,Q, similar(L))
end

function twaUpdate!(ds, s, p::dTWAParameters, t)
    # L,Q = p.derivs
    # mul!(loc_deriv, Q, s)
    # loc_deriv += L
    loc_deriv = 2p.Q*s
    loc_deriv += p.L
    @inbounds for site in 1:length(ds)÷3
        l = @view loc_deriv[3site-2:3site]
        s2 = @view s[3site-2:3site]
        ds[3site-2] =  2(s2[3]*l[2] - s[2]*l[3])
        ds[3site-1] =  2(s2[1]*l[3] - s[3]*l[1])
        ds[3site-0] =  2(s2[2]*l[1] - s[1]*l[2])
        # ds[3site-2] =  2dot(l, LEVI_CIVITA[1], s2)
        # ds[3site-1] =  2dot(l, LEVI_CIVITA[2], s2)
        # ds[3site-0] =  2dot(l, LEVI_CIVITA[3], s2)
    end
    ds
end


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
