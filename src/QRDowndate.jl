module QRDowndate
# copyright (c) 2017 Ralph Smith
# SPDX-License-Identifier: MIT-Expat
# License-Filename: LICENSE.md

export qrdowndate, qrdowndate!
using LinearAlgebra

# classical Gramm-Schmidt
function gs_step(U,::Type{Val{:CGS}},v)
    g = U' * v
    r = v - U * g
    β = norm(r,2)
    rmul!(r,1/β)
    r, g, β
end

# modified Gramm-Schmidt
function gs_step(U,::Type{Val{:MGS}},v)
    n = size(U,2)
    r = copy(v)
    g = zeros(eltype(U),n)
    for k=1:n
        # paper has a temp vector q; we just use r here
        ζ = dot(view(U,:,k),r)
        r = r - ζ * view(U,:,k)
        g[k] = ζ
    end
    β = norm(r,2)
    rmul!(r,1/β)
    r, g, β
end

# backwards modified Gramm-Schmidt
function gs_step(U,::Type{Val{:BMGS}},v)
    n = size(U,2)
    r = copy(v)
    g = zeros(eltype(U),n)
    for k=n:-1:1
        ζ = dot(view(U,:,k),r)
        r = r - ζ * view(U,:,k)
        g[k] = ζ
    end
    β = norm(r,2)
    rmul!(r,1/β)
    r, g, β
end

"""
Given `U[m,n]`, find unit vector `u0[m]` and vector `d[n+1]` such that
`U' * u0 = 0` and `hcat(u0,U) * d = e1`.

Algorithm from [^Barlow2005]

[Barlow2005]: Barlow et al., BIT 45, 259 (2005)
"""
function getorth(U::AbstractMatrix{T},M1,M2) where T
    m,n = size(U)
    @assert m >= n
    e1 = zeros(T,m); e1[1] = 1
    v1,g1,β1 = gs_step(U,M1,e1)
    βmin = sqrt(1/2)
    if β1 == 0
        β2 = α = zero(T)
        f = copy(g1)
        d = vcat(zeros(eltype(g1),1),g1)
        branch = 10
    else
        v2,g2,β2 = gs_step(U,M2,v1)
        if β1 >= βmin
            f = copy(g1)
            α = β1
            # β2 = zero(T) # early guess since paper fails to specify
            u0 = copy(v1)
            d = vcat([α],g1)
            branch = 20
        else
            f = g1 + β1 * g2
            α = β1 * β2
            if β2 >= βmin
                u0 = v2
                d = vcat([α],f)
            end
            branch = 30
        end
    end
    if β2 < βmin
        # FIXME: inelegant
        cval = Inf
        k = 0
        for j=1:m
            ek = zeros(m); ek[j] = 1
            tval = norm(U' * ek)
            if tval < cval
                k = j
                cval = tval
            end
        end

        ek = zeros(m); ek[k] = 1
        w1,k1,γ1 = gs_step(U,M1,ek)
        u0,k2,γ2 = gs_step(U,M2,w1)
        if α > 0
            d1 = vcat(zeros(T,1),f)
            Uhat = hcat(u0, U)
            # confusingly this is named $v_2$ in the paper,
            # but is NOT the same as the above v2.
            v2b = -U * f; v2b[1] += one(T)
            rmul!(v2,1/α)
            w2,d2,γ3 = gs_step(Uhat,M1,v2b)
            d = d1 + α * d2
            branch += 1
        else
            branch += 2
        end
    end

    return u0, d, branch
end

"""
    qrdowndate(Q,R) -> Qbar, Rbar

downdate the QR decomposition of a matrix (by the top row).

Given a QR decomposition `X=Q*R`, computes `Qbar, Rbar` such that
`Xbar = Qbar * Rbar` where `Xbar == X[2:end,:]`.
"""
function qrdowndate(U,R)
    Unew,Rnew = copy(U),copy(R)
    qrdowndate!(Unew,Rnew)
    Unew[2:end,:],Rnew
end

"""
    qrdowndate!(Q,R)

downdate the QR decomposition of a matrix.

Given a QR decomposition `X=Q*R`, computes `Qbar, Rbar` such that
`Xbar = Qbar * Rbar` where `Xbar == X[2:end,:]`, overwriting
the bottom of `Q` and `R`.
"""
function qrdowndate!(U::AbstractMatrix{T},R::AbstractMatrix{T}) where T
    m,n = size(U)
    (m > n) || throw(ArgumentError("U must be tall"))
    # check size of R?
    u0, d, b = getorth(U,Val{:MGS},Val{:BMGS})

    ρ = d[1]
    r = d[2:end]
    u = zeros(T,1,n)
    # find V st
    #     V is a product of Givens rotations
    #     V' * d = norm(d)*e1
    #     V' * vcat(zeros(1,n),R) = vcat(z',Rbar) for Up.Tri. Rhat
    # Ubar = hcat(u0,U) * V
    for k=n:-1:1
        G,ρ = givens(ρ,r[k],1,2)
        tmp = vcat(Transpose(u[k:n]),Transpose(R[k,k:n]))
        lmul!(G,tmp)
        u[k:n] = tmp[1,:]
        R[k,k:n] = tmp[2,:]
        tmp2 = hcat(u0,U[:,k])
        rmul!(tmp2,adjoint(G))
        u0 = tmp2[:,1]
        U[:,k] = tmp2[:,2]
    end
    # unceremoniously return indicator of branch used in orthogonalizer
    b
end


end
