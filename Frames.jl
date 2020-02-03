include("Definitions.jl")
"""
    Elements of F(ℳ) consist of a position x and a GL(d, ℝ)-matrix ν that
    represents a basis for 𝑇ₓℳ
"""

struct Frame{Tx, Tν}
    x::Tx
    ν::Tν
    function Frame(x::Tx, ν::Tν) where {Tx, Tν <: AbstractArray}
        # if rank(ν) != length(x)
        #     error("A is not of full rank")
        # end
        new{Tx, Tν}(x, ν)
    end
end

# A tangent vector (ẋ, ν̇) ∈ 𝑇ᵤF(ℳ)
struct TangentFrame{Tx,Tν}
    u::Frame
    ẋ::Tx
    ν̇::Tν
    function TangentFrame(u, ẋ::Tx, ν̇::Tν) where {Tx, Tν <: AbstractArray}
        new{Tx,Tν}(u, ẋ, ν̇)
    end
end

"""
    Some generic functions for calculations on F(ℳ)
"""

# Theoretically, these do not exist, used for numerical calculations
Base.:+(u::Frame{Tx, Tν}, v::Frame{Tx, Tν}) where {Tx, Tν} = Frame(u.x + v.x , u.ν .+ v.ν)
Base.:-(u::Frame{Tx, Tν}, v::Frame{Tx, Tν}) where {Tx, Tν} = Frame(u.x - v.x , u.ν .- v.ν)
Base.:-(u::Frame{Tx, Tν}) where {Tx, Tν} = Frame(-u.x , -u.ν)

Base.:+(u::Frame{Tx, Tν}, y::Tx) where {Tx, Tν} = Frame(u.x + y, u.ν)
Base.zero(u::Frame{Tx, Tν}) where {Tx, Tν} = Frame(zero(u.x), one(u.ν))

Base.:*(u::Frame{Tx, Tν}, y::Tx) where {Tx,Tν} = Frame(y.*u.x, y.*u.ν)

function Base.:+(X::TangentFrame{Tx, Tν}, Y::TangentFrame{Tx,Tν}) where {Tx,Tν}
    # if X.u != Y.u
    #     error("Vectors are in different tangent spaces")
    # end
    return TangentFrame(X.u, X.ẋ + Y.ẋ, X.ν̇ + Y.ν̇)
end

function Base.:-(X::TangentFrame{Tx, Tν}, Y::TangentFrame{Tx,Tν}) where {Tx,Tν}
    if X.u != Y.u
        error("Vectors are in different tangent spaces")
    end
    return TangentFrame(X.u, X.ẋ - Y.ẋ, X.ν̇ - Y.ν̇)
end

# this function should be the exponential map on F(ℳ)
function Base.:+(u::Frame{Tx, Tν}, X::TangentFrame{Tx, Tν}) where {Tx,Tν}
    # if X.u != u
    #     error("X is not tangent to u")
    # end
    return Frame(u.x + X.ẋ , u.ν + X.ν̇)
end

function Base.:*(X::TangentFrame{Tx, Tν}, y::Float64) where {Tx, Tν}
    TangentFrame(X.u , X.ẋ.*y , X.ν̇.*y)
end

function Base.:*(y::Float64, X::TangentFrame{Tx, Tν}) where {Tx, Tν}
    TangentFrame(X.u , X.ẋ.*y , X.ν̇.*y)
end

# Canonical projection
Π(u::Frame{Tx, Tν}) where {Tx,Tν} = u.x

# Pushforward map of the canonocal projection
Πˣ(X::TangentFrame{Tx, Tν}) where {Tx, Tν} = X.ẋ

# The group action of a frame on ℝᵈ
FrameAction(u::Frame{Tx, Tν}, e::T) where {Tx,Tν,T<:AbstractArray} = u.ν*e

# Horizontal lift of the orthogonal projection
Pˣ(u::Frame, ℳ::T) where {T<:EmbeddedManifold} = TangentFrame(u, u.x, P(u.x, ℳ))

"""
    Horizontal vector fields
"""

# Horizontal vector (a tangent frame) corresponding to the i'th unit vector
function Hor(i::Int64, u::Frame, ℳ::TM) where {TM<:EmbeddedManifold}
    x, ν = u.x, u.ν
    _Γ = Γ(u.x, ℳ)
    @einsum dν[i,k,m] := -ν[i,j]*ν[l,m]*_Γ[k,j,l]
    return TangentFrame(u, ν[:,i], dν[i,:,:])
end

# Horizontal vector field
# Hor(i::Int64, u::Frame, ℳ::T) where {T<:EmbeddedManifold} = TangentFrame(u, u.x, Pˣ(u, ℳ)[:,i])

Hor(1,u₀,𝕊)
"""
    Stochastic development

    Simulate the process {Ut} on F(ℳ) given by the SDE
        dUt = H(Ut)∘dWt
"""

function StochasticDevelopment(dW, u::Frame)
    x, ν = u.x, u.ν
    # NOT FINISHED
end
"""
    Now let us create a stochastic process on the frame bundle of the 2-sphere 𝕊²
"""

# Functions for solving SDEs on the frame bundle
using Bridge
include("FrameBundles.jl")

struct SphereDiffusion <: FrameBundleProcess
    𝕊::Sphere

    function SphereDiffusion(𝕊::Sphere)
        new(𝕊)
    end
end

Hor(i, u, ℙ::SphereDiffusion) = Hor(i, u, ℙ.𝕊)
Bridge.constdiff(::SphereDiffusion) = false

𝕊 = Sphere(1.0)
ℙ = SphereDiffusion(𝕊)

x₀ = [0.,0]
u₀ = Frame(x₀, [1. 0.5 ; 0.5 1.])

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{2}}())
U = solve(StratonovichEuler(), u₀, W, ℙ)
X  = map(y -> F(y.x,𝕊), U.yy)

plot([extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

using Plots
include("Sphereplots.jl")
plotly()
SpherePlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕊)

function SimulatePoints(n, u₀, ℙ::SphereDiffusion)
    out = Frame[]
    it = 0
    while length(out) < n
        W = sample(0.:dt:T, Wiener{ℝ{3}}())
        U = solve(StratonovichEuler(),u₀, W, ℙ)
        push!(out, U.yy[end])
    end
    return out
end

@time ξ = SimulatePoints(1000, u₀, ℙ)

SphereScatterPlot(extractcomp(Π.(ξ),1), extractcomp(Π.(ξ),2), extractcomp(Π.(ξ),3), x₀, 𝕊 )
