# Manifolds
This repository contains code for simulations of diffusions and diffusion bridges on Riemannian manifolds

# Implementation of manifolds

Manifolds embedded in a Euclidean space are subtypes of the EmbeddedManifold type. Manifolds ℳ, such as the 2-sphere 𝕊², the 2-Torus 𝕋² and the paraboloid, are already implemented and accompanied with the following properties:

  - A smooth function <img src="https://render.githubusercontent.com/render/math?math=f(x,\mathcal{M})"> that takes arrays as input and outputs a function whose nullspace describes the manifold.
  - A matrix-valued function <img src="https://render.githubusercontent.com/render/math?math=P(x,\mathcal{M})"> that takes an array 𝓍 as input and outputs the projection matrix of the ambient space onto the tangent space at 𝓍.
  - A function <img src="https://render.githubusercontent.com/render/math?math=F(q,\mathcal{M})"> that transforms local coordinates <img src="https://render.githubusercontent.com/render/math?math=q"> to points in the ambient space.

Using these properties, one derives the Riemannian metric
<img src="https://render.githubusercontent.com/render/math?math=g(q,\mathcal{M}) = \mathrm{d}F(q,\mathcal{M})'\mathrm{d}F(q,\mathcal{M})"> and the cometric <img src="https://render.githubusercontent.com/render/math?math=g^x=g\^{-1}">.
