include("../scr/ManifoldDiffusions.jl")
using LegendrePolynomials


abstract type ManifoldProcess{T} <: ContinuousTimeProcess{T} end
struct ManifoldBM{T} <: ManifoldProcess{T}
    ℳ::Manifold
    ManifoldBM{T}(ℳ) where{T} = new{T}(ℳ)
end
