# Manifolds
This repository contains code for simulations of diffusions and diffusion bridges on Riemannian manifolds

## Implementation of manifolds

Manifolds embedded in a Euclidean space are subtypes of the EmbeddedManifold type. Manifolds ℳ, such as the 2-sphere 𝕊², the 2-Torus 𝕋² and the paraboloid, are already implemented and accompanied with the following properties:

  - A smooth function ```f(x, ℳ)``` that takes arrays as input and outputs a function whose nullspace describes the manifold.
  - A matrix-valued function ```P(x, ℳ)```that takes an array ```x``` as input and outputs the projection matrix of the ambient space onto the tangent space at ```x```.
  - A function ```F(q, ℳ)``` that transforms local coordinates```q``` to points in the ambient space.

Using these properties, one derives the Riemannian metric

```math
g(q,\mathcal{M}) = \mathrm{d}F(q,\mathcal{M})'\mathrm{d}F(q,\mathcal{M})
```

as well as the Cristoffel symbols Γ and a Hamiltonian.

### Example: Unit Sphere
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

## Geodesics and Parallel Transport
In Geodesics.jl, a sympletic integrator is implemented for the Hamiltonian system that describes geodesics. Given a discretized time  ```tt```, an initial point```q₀``` on the manifold ```ℳ``` and initial velocity ```v₀```. Calling the function  

```@docs
qq, vv = Geodesic(q₀, v₀, tt, ℳ)
```
returns both the trajectory on ℳ and the trajectory on the tangent bundle. This function is used in the function

```@docs
ExponentialMap(q₀, v₀, ℳ)
```

## Frames and the Frame bundle
The structure ```Frame``` is defined through a tuple ```(x,ν)```, where x is an array of size d and ν is a d×d-matrix that represents a basis for 𝑇ₓℳ. Given a Frame ```u```, we also define elements in 𝑇ᵤF(ℳ) through a triple ```(u, ẋ, ν̇)``` where  ```ẋ``` is a vector of size d representing a tangent vector to ℳ and ```ν̇``` is a matrix of size d×d. Elementary rules of calculation are defined for tangent frames and frames.

On the sphere, one can construct frames and tangent frames as follows

```@docs
  𝕊 = Sphere(1.0)
  q = [0.,0.]

  # A frame at q with standard basis (in the local chart)
  u = Frame(q, [1. 0. ; 0. 1.])

  # q is obtained by
  u.x
  # ν is obtained by
  u.ν

  # Equivalent to u.x, one can use the canonical projection map Π
  Π(u) # returns u.x
```

## Horizontal lift
The horizontal lift is, in local coordinates, given by

```math
H_i(u) = \nu_i^j\frac{\partial}{\partial x^j}-\nu_i^j\nu_m^l\frac{\partial}{\partial \nu_m^k}
```


This is implemented through the function

```@docs
Hor(i, u, ℳ)
```
that returns a TangentFrame to ```u::Frame```.

## Stochastic development
Using a Riemannian cometric,  as seen in e.g. Sommer and Svane, 2017, we derive similar dynamics for geodesics in the frame bundle and can use the exponential map to do forward simulations of the SDE

```math
\mathrm{d} U_t = H_i(U_t) \circ \mathrm{d} W_t
```
Given a starting frame ```u₀``` and a d-dimensional standard Brownian motion ```W::SamplePath```, we obtain a SamplePath with elements of type ```Frame``` using 

```@docs
StochasticDevelopment(W, u₀, ℳ) 
```

## Literature
Stefan Sommer and Anne Marie Svane: Modelling Anisotropic Covariance using Stochastic Development and Sub-Riemannian Frame Bundle Geometry. Journal of Geometric Mechanics (3), 2017, [doi.org/10.3934/jgm.2017015](https://doi.org/10.3934/jgm.2017015)
