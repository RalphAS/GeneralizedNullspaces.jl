# copyright (c) 2017 Ralph Smith
# SPDX-License-Identifier: MIT-Expat
# License-Filename: LICENSE.md

function gnsdcheck(A,B,V,ν,μ,tol=1e-8; quiet=false)
    n = size(B,1)
    @assert size(A) == (n,n)
    @assert size(B) == (n,n)
    @assert size(V) == (n,n)
    @assert ν == length(μ)

    ok = true
    t = vecnorm(A - V * B * V') / vecnorm(A)
    if t > tol
        ok = false
        warn("decomposition error: $t")
    elseif !quiet
        println("decomposition error: ",t)
    end
    t = vecnorm(eye(V) - V' * V)
    if t > 10n*eps()
        ok = false
        warn("unitarity error: $t")
    elseif !quiet
        println("unitarity error: ",t)
    end

    # compute start indices of blocks
    bs = cumsum(vcat([1],μ))
    (bs[end] == n+1) || push!(bs,n+1)
    nb = length(bs)-1

    for j=1:ν
        if vecnorm(B[bs[j]:n, bs[j]:bs[j+1]-1]) != 0
            warn("subdiag block $j is not null")
            ok=false
        end
    end
    for ib=1:nb
        Bii = B[bs[ib]:bs[ib+1]-1,bs[ib]:bs[ib+1]-1]
        if ib>ν
            F = svdfact(Bii)
            if any(F[:S].< tol)
                warn("diag block $ib (> ν) is singular")
                ok=false
            end
        else
            if vecnorm(Bii) != 0
                warn("nonzero diag block $ib")
                ok=false
            end
        end
        for jb=ib+1:nb
            Bij = B[bs[ib]:bs[ib+1]-1,bs[jb]:bs[jb+1]-1]
            F = svdfact(Bij)
            if any(F[:S].< tol)
                warn("block $ib,$jb does not have full column rank")
                ok=false
            end
        end
    end
    ok && (quiet || println("all tests pass."))
    ok
end

"""
    makegnb(ν,μx) -> B

construct a matrix with specified generalized nullspace structure

This makes the block-structured part `B`; apply a unitary transform to taste.
"""
function makegnb(ν,μx; cplx=false)
    nb = length(μx)
    (nb ∈ [ν,ν+1]) || throw(ArgumentError("must specify ν or ν+1 blocks"))
    for j=2:ν
        (μx[j-1] >= μx[j]) ||
            throw(ArgumentError("singular block sizes must be nonincreasing"))
    end
    bs = vcat([1],1 .+ cumsum(μx)) # starting indices of blocks
    n = bs[end] -1
    if cplx
        B = randn(n,n) + 1im*randn(n,n)
    else
        B = randn(n,n)
    end
    for i=1:ν
        B[bs[i]:n, bs[i]:bs[i+1]-1] = 0
    end
    scale!(B,1/norm(B))
    B
end
