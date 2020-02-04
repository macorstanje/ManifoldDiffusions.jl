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
    _Γ = Γ(x, ℳ)
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

function IntegrateStep(dW, u::Frame, ℳ)
    x, ν = u.x, u.ν
    uᴱ = u + sum([Hor(i, u,ℳ)*dW[i] for i in 1:length(dW)])
    y = u + sum([(Hor(i,uᴱ,ℳ) + Hor(i, u,ℳ))*dW[i]*0.5 for i in 1:length(dW)])
    return y
end

using Bridge

StochasticDevelopment(W, u₀, ℳ) = let X = Bridge.samplepath(W.tt, zero(u₀)); StochasticDevelopment!(X, W, u₀,ℳ); X end
function StochasticDevelopment!(Y, W, u₀, ℳ)
    tt = W.tt
    ww = W.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:length(tt)-1
        dw = ww[k+1] - ww[k]
        yy[..,k] = y
        y = IntegrateStep(dw, y, ℳ)
    end
    yy[..,length(tt)] = y
    Y
end

"""
    Now let us create a stochastic process on the frame bundle of the 2-sphere 𝕊²
"""

𝕊 = Sphere(1.0)

x₀ = [0.,0]
u₀ = Frame(x₀, [1. 0; 0 1.])

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕊)
X  = map(y -> F(Π(y), 𝕊), U.yy)

using Plots
plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("Sphereplots.jl"); plotly()
SpherePlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕊)

function SimulatePoints(n, u₀, ℙ::SphereDiffusion)
    out = Frame[]
    while length(out) < n
        W = sample(0.:dt:T, Wiener{ℝ{2}}())
        U = StochasticDevelopment(W, u₀, ℙ.𝕊)
        push!(out, U.yy[end])
    end
    return out
end

@time Ξ = SimulatePoints(1000, u₀, ℙ)

ξ = map(y->F(Π(y), 𝕊), Ξ)
SphereScatterPlot(extractcomp(ξ ,1), extractcomp(ξ,2), extractcomp(ξ,3), F(x₀,𝕊), 𝕊 )

"""
    Now let us create a stochastic process on the frame bundle of the paraboloid
"""

ℙ = Paraboloid(1.0, 1.0)

x₀ = [1.0,1.0]
u₀ = Frame(x₀, [1. 0. ; 0. 1.])

W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, ℙ)
X  = map(y -> F(Π(y), ℙ), U.yy)

plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("ParaboloidPlots.jl")
ParaboloidPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), ℙ)
