# Manifolds
This repository contains code for simulations of diffusions and diffusion bridges on Riemannian manifolds

# Implementation of manifolds

Manifolds embedded in a Euclidean space are subtypes of the EmbeddedManifold type. Manifolds ℳ, such as the 2-sphere 𝕊², the 2-Torus 𝕋² and the paraboloid, are already implemented and accompanied with the following properties:

  - A smooth function f( , ℳ) that takes arrays as input and outputs a function whose nullspace describes the manifold.
  - A matrix-valued function P( , ℳ) that takes an array 𝓍 as input and outputs the projection matrix of the ambient space onto the tangent space at 𝓍.
  - A function F( , ℳ) that transforms local coordinates to points in the ambient space.

  Using these properties, one derives the Riemannian metric
  $$ g(𝓍, ℳ) = dF(𝓍, ℳ)' dF(𝓍, ℳ) $$
  and the cometric gˣ = g⁻¹.
