abstract type FrameBundleProcess end
include("Frames.jl")

# The frame bundle over a manifold ℳ
struct FrameBundle{TM} <: EmbeddedManifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:EmbeddedManifold} = new{TM}(ℳ)
end

"""
    Riemannian structure on the Frame bundle
"""

# Riemannian cometric on the Frame bundle
Σ(u::Frame, v::T, w::T) where {T<:AbstractArray} = dot(inv(u.ν)*v , inv(u.ν)*w)
function g(X::TangentFrame, Y::TangentFrame)
        if X.u != Y.u
            error("Vectors are in different tangent spaces")
        end
    return Σ(X.u, Πˣ(X), Πˣ(Y))
end

# Test; should be 1_{i=i}
# ℳ = Sphere(1.0)
# u = Frame([rand(),rand()], [rand() 0. ; 0. rand()])
# i,j = 1,2
# g(Hor(i, u, ℳ), Hor(j, u, ℳ))

# Christoffel Symbols
function Γ(u::Frame, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u.x)
    ∂g = reshape(ForwardDiff.jacobian(x -> g(x, Fℳ)), d+d^2, d+d^2, d+d^2)
    g⁻¹ = gˣ(u, Fℳ)
    @einsum out[i,j,k] := .5*g⁻¹[i,l]*(∂g[k,l,i] + ∂g[l,j,k] - ∂g[j,k,l])
    return out
end

# Hamiltonian
function Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    if p.u != u
        error("p is not tangent to u")
    end
    return .5*g(p,p)
end

# Hamiltonian as functions of two vectors of size d+d^2
function Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM}) where {Tx, Tp<:AbstractArray, TM}
    N = length(x)
    d = Int64((sqrt(1+4*N)-1)/2)
    u = Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d))
    P = TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d))
    return Hamiltonian(u, P, Fℳ)
end

"""
    Geodesic flow on the frame bundle
"""
include("Geodesics.jl")

function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u.x)
    U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
    V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d)) , xx)
    vv = map(p->TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

# Code to test it
#
# tt = collect(0:0.01:1.0)
# u₀ = Frame([0.,0.], [1. 0. ; 0. 1.])
# v₀ = TangentFrame(u₀, [1.,0.] , [2. 0. ; 0. 2.])
#
# 𝕊 = Sphere(1.0)
# F𝕊 = FrameBundle(𝕊)
#
# uu, vv = Geodesic(u₀, v₀, tt, F𝕊)
# XX = map(y -> F(Π(y), 𝕊), uu)
#
# using Plots
# include("Sphereplots.jl"); plotly()
# SpherePlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3), 𝕊)
