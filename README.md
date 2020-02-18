# Manifolds
This repository contains code for simulations of diffusions and diffusion bridges on Riemannian manifolds

# Implementation of manifolds

Manifolds embedded in a Euclidean space are subtypes of the EmbeddedManifold type. Manifolds ℳ, such as the 2-sphere 𝕊², the 2-Torus 𝕋² and the paraboloid, are already implemented and accompanied with the following properties:

  - A smooth function f( , ℳ) that takes arrays as input and outputs a function whose nullspace describes the manifold
  - A matrix-valued function P( , ℳ) that takes an array x as input and outputs the projection matrix of the ambient space onto the tangent space at x
