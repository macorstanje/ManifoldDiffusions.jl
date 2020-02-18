# Manifolds
This repository contains code for simulations of diffusions and diffusion bridges on Riemannian manifolds

# Implementation of manifolds

Manifolds embedded in a Euclidean space are subtypes of the EmbeddedManifold type. Manifolds ℳ, such as the 2-sphere 𝕊², the 2-Torus 𝕋² and the paraboloid, are already implemented and accompanied with the following properties:

  - A smooth function <img src="https://render.githubusercontent.com/render/math?math=f(x,\mathcal{M})"> that takes arrays as input and outputs a function whose nullspace describes the manifold.
  - A matrix-valued function <img src="https://render.githubusercontent.com/render/math?math=P(x,\mathcal{M})"> that takes an array 𝓍 as input and outputs the projection matrix of the ambient space onto the tangent space at 𝓍.
  - A function <img src="https://render.githubusercontent.com/render/math?math=F(q,\mathcal{M})"> that transforms local coordinates <img src="https://render.githubusercontent.com/render/math?math=q"> to points in the ambient space.

Using these properties, one derives the Riemannian metric
<img src="https://render.githubusercontent.com/render/math?math=g(q,\mathcal{M}) = \mathrm{d}F(q,\mathcal{M})'\mathrm{d}F(q,\mathcal{M})"> and the cometric <img src="https://render.githubusercontent.com/render/math?math=g^x=g\^{-1}">. The Cristoffel symbols Γ and a Hamiltonian follow.

The unit sphere is equipped with the southern sterograpgical projection by default.

```@docs
  𝕊 = Sphere(1.0)

  # The point (0,0,-1) corresponds with local coorinates q = (0,0)
  q = [0.,0.]

  x = F(q, 𝕊) # Should be (0,0,-1)
  f(x, 𝕊)     # Should be 0

  # Calculate the Riemannian metric or Christoffel symbols via
  g(q, 𝕊)
  Γ(q, 𝕊)

  # Define an impulse p and determine the Hamiltonian at (q,p)
  p = [1.,0.]
  Hamiltonian(q, p, 𝕊)
  
```
