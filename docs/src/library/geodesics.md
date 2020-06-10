# Geodesics

```@meta
CurrentModule = ManifoldDiffusions
```
A geodesic ``\\gamma:I\\to\\mathcal{M}`` with ``\\gamma_0=x\\in\\mathcal{M}`` and ``\\dot{\\gamma}_0=v\\in T_x\\mathcal{M}`` is modelled as a solution to the Hamiltonian system

```@math
\\mathrm{d}x = \\nabla_p H(x, p)\\mathrm{d}t
\\mathrm{d}p = -\\nabla_x H(x, p)\\mathrm{d}t  
```

```@docs
Geodesic
ExponentialMap
ParallelTransport
```
