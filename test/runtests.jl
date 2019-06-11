using GeneralizedNullspaces
using GenericSVD
using Test
using LinearAlgebra

include("testutils.jl")

quiet = true

# example from appendix B of the Guglielmi et al. paper
let tol=1e-10
    a = 233/896
    b = 248/896
    c = 171/896
    d = 29/896
    e = 15/896
    z = 69/448
    s = 2101/9632
    t = 295/19264
    u = 1403/28896

    A = [z    s      s      s      t     u     t     u     t     u;
         a    b      c      c      e     d     0     0     0     d;
         a    c      b      c      0     d     e     d     0     0;
         a    c      c      b      0     0     0     d     e     d;
         3/32 7/16   3/32   3/32   3/32  3/32  0     0     0     3/32;
         9/64 39/128 39/128 3/64   3/128 9/64  3/128 1/128 0     1/128;
         3/32 3/32   7/16   3/32   0     3/32  3/32  3/32  0     0;
         9/64 3/64   39/128 39/128 0     1/128 3/128 9/64  3/128 1/128;
         3/32 3/32   3/32   7/16   0     0     0     3/32  3/32  3/32;
         9/64 39/128 3/64   39/128 3/128 1/128 0     1/128 3/128 9/64]

    B, V, ν, μ = gnsd(A,tol)

    @test ν == 2
    @test μ == [3,1]
    @test gnsdcheck(A,B,V,ν,μ)
end

let tol=1e-9
    ν=1
    μx = [3,2]
    AA = makegnb(1,μx)
    n = size(AA,1)
    Q,R = qr(randn(n,n))
    A = Q' * AA * Q
    B,V,ν1,μ = gnsd(A,tol)
    @test ν == 1
    @test μ == μx[1:ν]
    @test gnsdcheck(A,B,V,ν,μ)
end

let tol=1e-9
    ν = 2
    μx = [3,2,1]
    AA = makegnb(ν,μx)
    n = size(AA,1)
    Q,R = qr(randn(n,n))
    A = Q' * AA * Q
    B,V,ν1,μ = gnsd(A,tol)
    quiet || println("input ν=$ν μ'=$μx")
    quiet || println("ν=$ν1 μ=$μ")
    @test ν == ν1
    @test gnsdcheck(A,B,V,ν1,μ; quiet=quiet)
end

for TT in (Float64,Complex{Float64})
    let tol=1e-9
        ν=2
        μx = [3,2,1]
        AA = makegnb(ν,μx)
        n = size(AA,1)
        Q,R = qr(randn(n,n))
        A = Matrix{TT}(Q' * AA * Q)
        B,V,ν1,μ = gnsd(A,tol)
        quiet || println("input ν=$ν μ'=$μx")
        quiet || println("ν=$ν1 μ=$μ")
        @test ν == ν1
        @test gnsdcheck(A,B,V,ν1,μ; quiet=quiet)
    end
end

let tol=1e-9
    ν=2
    μx=[3,2,1]
    AA = makegnb(ν,μx;cplx=true)
    n = size(AA,1)
    Q,R = qr(randn(n,n))
    A = Q' * AA * Q
    B,V,ν1,μ = gnsd(A,tol)
    quiet || println("input ν=$ν μ'=$μx")
    quiet || println("ν=$ν1 μ=$μ")
    @test ν == ν1
    @test gnsdcheck(A,B,V,ν1,μ; quiet=quiet)
end

let tol=1e-9
    ν=1
    μx = [3,2]
    AA = makegnb(1,μx)
    n = size(AA,1)
    Q,R = qr(randn(n,n))
    A = Q' * AA * Q
    Ab = Matrix{BigFloat}(A)
    B,V,ν1,μ = gnsd(Ab,tol)
    @test ν == 1
    @test μ == μx[1:ν]
    @test gnsdcheck(A,B,V,ν,μ)
end
