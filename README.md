# GeneralizedNullspaces

[![Build Status](https://travis-ci.org/RalphAS/GeneralizedNullspaces.jl.svg?branch=master)](https://travis-ci.org/RalphAS/GeneralizedNullspaces.jl)

[![codecov.io](http://codecov.io/github/RalphAS/GeneralizedNullspaces.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/GeneralizedNullspaces.jl?branch=master)

`GeneralizedNullspaces.jl` is a Julia implementation of the QR-updating
method of Guglielmi *et al.* (2015) for computing the generalized null space
decomposition (GNSD) of dense matrices.

The GNSD is `A = V * B * V'` where `V` is unitary and `B` has a block
staircase structure. (See documentation for the `GNSD` type.) A vector
`μ` is provided such that the first `μ[1]` columns of `V` are null
vectors of `A`, the next `μ[2]` columns are null vectors of `A^2`,
etc.

The API is similar to the SVD methods in the standard library. `GNSD`, `gnsd`,
`gnsdfact`, and `gnsdfact!` are exported.

Note that computations of this sort are delicate; one is actually
decomposing a particularly defective matrix close to `A`. The user is
responsible for providing a tolerance determining how closely nullity
is enforced.  (A default is provided, with reluctance.) See the
reference for guidance.

Note: although the code passes basic tests, more exhaustive checks
against stressing (ill-conditioned) inputs are desirable but not yet
included. The code is fairly generic, but testing on anything other than
`Float64` (real and complex) matrices has been perfunctory.

Reference:

* N. Guglielmi, M. Overton, and G.W. Stewart, SIAM J. Matrix Anal. Appl. 36, 38 (2015).
