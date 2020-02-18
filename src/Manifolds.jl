using Bridge
using Plots
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Einsum
using Plots

# Definitions of the manifolds additional properties
include("Definitions.jl")

# Definitions and calculation rules for Frames
include("Frames.jl")

# Integrators and other functions for finding geodesics and the exponential map
include("Geodesics.jl")

# Describes stochastic development on the Frame bundle
include("FrameBundles.jl")

# Tools for creating plots on manifolds
include("ManifoldPlots.jl")
