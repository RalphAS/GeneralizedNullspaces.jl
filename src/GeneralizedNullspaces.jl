module GeneralizedNullspaces
# copyright (c) 2017-2019 Ralph Smith
# SPDX-License-Identifier: MIT-Expat
# License-Filename: LICENSE.md

export gnsd, gnsd!, GNSD

using LinearAlgebra

include("QRDowndate.jl")
using .QRDowndate

"""
    GNSD <: Factorization

A generalized nullspace decomposition of a matrix, typically obtained from
[`gnsd`](@ref). If `A` is a square matrix, then the GNSD is `A == V*B*V'`
where `V` is a unitary matrix and `B` is block upper triangular.
The diagonal blocks of `B` are square and null, with orders listed in
a vector `μ`, except for a possible last block which
will represent the range of `A^ν` if it is not empty, where `ν == length(μ)`.
The order `μ[j]` is the number of zero-diagonal Jordan blocks of order `≥ j`
in the Jordan decomposition of a matrix close to `A`.
The blocks on the first superdiagonal of `B` have full column rank.
Columns of `V` span the corresponding subspaces of `A`.
"""
struct GNSD{T,M<:AbstractMatrix} <: Factorization{T}
    B::M
    V::M
    μ::Vector{Int}
    GNSD{T,M}(B::AbstractMatrix{T},V::AbstractMatrix{T},μ::Vector{Int}) where {T,M} =
        new(B,V,μ)
end
GNSD(B::AbstractMatrix{T},V::AbstractMatrix{T},μ::Vector{Int}) where {T} = GNSD{T,typeof(B)}(B,V,μ)

const pverbosity = Ref{Int}(0)
function set_verbosity(k)
    pverbosity[] = k
end

const useviews = Ref{Bool}(true)

"""
    gnsd(A) -> F::GNSD

Compute the generalized nullspace decomposition of a matrix `A`.

Returns a [`GNSD`](@ref) object.
`B` and `V` can be obtained from the factorization `F` with `F[:B]` and
`F[:V]`, such that `A = V * B * V'`.

Iterating the decomposition produces the tuple `(B, V, ν, μ)`
where `ν` is the index of `A` and `μ` holds the diagonal block orders.
See [`GNSD`](@ref) for the structure of `B`.

Uses the QR/update algorithm from [^Guglielmi2015].

[^Guglielmi2015]: N.Guglielmi, M.Overton, & G.Stewart,
"An efficient algorithm for computing the generalized null space decomposition",
SIAM J. Matrix Anal. Appl. 36 (1), 38-54 (2015).
"""
gnsd(A::Matrix{T},tol=sqrt(eps(T))) where T = gnsd!(copy(A),tol)

"""
    gnsd!(A) -> F::GNSD

`gnsd!` is the same as [`gnsd`](@ref), but saves space by
overwriting the input `A` instead of creating a copy.
"""
function gnsd!(B::Matrix{T},tol=sqrt(eps(T))) where T
    verbosity = pverbosity[]
    m,n = size(B)
    F = qr(B)
    Q = Matrix(F.Q)
    R = F.R
    μ = Vector{Int}(undef,0) # sigh.
    # we will multiply rotations into V
    V = Matrix{T}(I,size(B))

    k = 1 # block number
    idxk = 1 # starting index of block k
    local rxnorm

    while true
        (verbosity > 0) && println("starting block $k")
        j = idxk-1 # pointer to working row
        while j<n
            j += 1
            # R22 = R[j:n,j:n]
            if useviews[]
                x,rxnorm = nullvector(view(R,j:n,j:n))
            else
                x,rxnorm = nullvector(R[j:n,j:n])
            end
            (verbosity > 0) && println("row j=$j rxnorm: $rxnorm")
            (rxnorm > tol) && break
            if j>1
                x = vcat(zeros(T,j-1),x)
            end

            # Transposition is reversed w.r.t. paper because
            # of Julia Givens convention.

            # Find unitary Vi s.t. Vi * x = e1 (up to factor)
            # (Vi stands for $V_{i,i+1}$ in the paper.)
            # Find unitary U s.t. R = U * R22 * Vi' where R22 is Up.Tri.,
            # Save R22 in R
            # Update Q <- Vi * Q * U'
            # Accumulate Vi into V
            # Update B <- Vi * B * Vi'

            for i=n:-1:j+1
                Vi,r = givens(x[i-1],x[i],i-1,i)
                x[i-1] = r
                x[i] = 0
                if useviews[]
                    rmul!(view(R,1:i,:),adjoint(Vi)) # slow
                else
                    R[1:i,:] .= rmul!(R[1:i,:],adjoint(Vi)) # allocates
                end
                rmul!(V,adjoint(Vi))
                rmul!(B,adjoint(Vi))
                lmul!(Vi,B)
                lmul!(Vi,Q)
                Ui,r = givens(R[i-1,i-1],R[i,i-1],i-1,i)
                R[i-1,i-1] = r
                R[i,i-1] = 0
                if useviews[]
                    lmul!(Ui,view(R,:,i:n)) # slow
                else
                    R[:,i:n] = lmul!(Ui,R[:,i:n]) # allocates
                end
                rmul!(view(Q,idxk:n,:),adjoint(Ui))

            end
            # Find unitary W s.t. Rtilde = W' * Rhat is Up.Tri.,
            #    w/ zero first row
            # Update Q <- Q * W
            for i=j+1:n
                W,r = givens(R[i,i],R[j,i],i,j)
                R[i,i] = r
                R[j,i] = 0
                if useviews[]
                    lmul!(W,view(R,:,i+1:n)) # slow
                else
                    R[:,i+1:n] = lmul!(W,R[:,i+1:n]) # slow
                end
                rmul!(view(Q,idxk:n,:),adjoint(W))
            end
        end

        if j==n && rxnorm < tol
            # done (nilpotent)
            push!(μ,j-idxk+1)
            break
        elseif j==idxk
            # done (nonsingular last block, not counted in μ)
            break
        else
            # downdate to prepare for next block
            push!(μ,j-idxk)
            for i=idxk:j-1
                if useviews[]
                    qrdowndate!(view(Q,i:n,j:n),view(R,j:n,j:n))
                else
                    Qxx = Q[i:n,j:n]
                    Rxx = R[j:n,j:n]
                    qrdowndate!(Qxx,Rxx)
                    Q[i:n,j:n] = Qxx
                    R[j:n,j:n] = Rxx
                end
            end
            # ready for next block
            idxk=j
            k += 1
        end
    end

    # Actually null out the (sub)diagonal blocks
    i=1 # first index of current block
    for j in eachindex(μ)
        B[i:n, i:i+μ[j]-1] .= 0
        i += μ[j]
    end
    GNSD(B,V,μ)
end

"""
    nullvector(R, tol=0) -> x, nRx

Compute an approximate null vector `x` for upper triangular matrix `R`.
Also returns the norm of `R * x`, which is a rough approximation to
the smallest singular value of `R`.
"""
function nullvector(R::AbstractMatrix{T}, tol=zero(real(T))) where T
    # For now, we allow R to be a view.
    all(x->isa(x, Base.OneTo), axes(R)) ||
        throw(ArgumentError("only implemented for 1-based indexing"))
    n = size(R,1)
    x = zeros(T,n)

    # Case 1: tiny/zero diagonal element
    # Do a partial upper triangular back-solve, but skip the initial division.

    # Caveat: If `tol` is too small, and it is met at some index, but
    # should have been met earlier, the backsolve is unstable.
    # Consider using a better default or rescaling.
    idxsmall = findfirst(x -> (abs(x) <= tol), diag(R))
    if (idxsmall != nothing) && (idxsmall > 0)
        b = zeros(T,idxsmall); b[end] = 1
        x[idxsmall] = 1
        @inbounds for i in idxsmall-1:-1:1
            b[i] -= R[i,idxsmall]
        end
        @inbounds for j in idxsmall-1:-1:1
            xj = x[j] = R[j,j] \ b[j]
            for i in j-1:-1:1
                b[i] -= R[i,j] * xj
            end
        end
        xnrm = norm(x)
        rmul!(x,1/xnrm)
        rxnorm = norm(R*x)
    else
        # Case 2:
        # The paper suggests a LINPACK norm-estimation scheme.
        # This is from GVL4, p.141, but loop is reversed to give a 1-norm.
        p = zeros(T,n)
        y = zeros(T,n)
        pp = zeros(T,n)
        pm = zeros(T,n)
        for k=1:n
            yp = (1-p[k])/R[k,k]
            ym = (-1-p[k])/R[k,k]
            pp[k:end] .= p[k:end] .+ yp .* R[k:end,k]
            pm[k:end] .= p[k:end] .+ ym .* R[k:end,k]
            if abs(yp)+norm(pp,1) >= abs(ym)+norm(pm,1)
                y[k] = yp
                p[k:end] = pp[k:end]
            else
                y[k] = ym
                p[k:end] = pm[k:end]
            end
        end
        normalize!(y)

        x = UpperTriangular(R) \ y
        # We often need a test for nullity later, viz. ‖R x‖;
        # return it from here to avoid an extra MxV.
        rxnorm = 1/norm(x)
        rmul!(x,rxnorm)
    end
    x,rxnorm
end

# iteration for destructuring into components
Base.iterate(S::GNSD) = (S.B, Val(:V))
Base.iterate(S::GNSD, ::Val{:V}) = (S.V, Val(:ν))
Base.iterate(S::GNSD, ::Val{:ν}) = (S.ν, Val(:μ))
Base.iterate(S::GNSD, ::Val{:μ}) = (S.μ, Val(:done))
Base.iterate(S::GNSD, ::Val{:done}) = nothing

function Base.getproperty(F::GNSD, d::Symbol)
    if d == :ν
        return length(F.μ)
    else
        return getfield(F, d)
    end
end

end # module
