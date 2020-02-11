include("Definitions.jl")
include("Frames.jl")
# The frame bundle over a manifold ℳ
struct FrameBundle{TM} <: EmbeddedManifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:EmbeddedManifold} = new{TM}(ℳ)
end

include("Geodesics.jl")

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
    Stochastic development

    Simulate the process {Ut} on F(ℳ) given by the SDE
        dUt = H(Ut)∘dWt
"""

function IntegrateStep(dW, u::Frame, ℳ)
    x, ν = u.x, u.ν
    uᴱ = ExponentialMap(u, sum([Hor(i, u,ℳ)*dW[i] for i in eachindex(dW)]), FrameBundle(ℳ))
    y = ExponentialMap(u, sum([(Hor(i,uᴱ,ℳ) + Hor(i, u,ℳ))*dW[i]*0.5 for i in eachindex(dW)]), FrameBundle(ℳ))
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

# UNCOMMENT TO TRY SIMULATING PATHS

𝕊 = Sphere(1.0)

x₀ = [0.,0]
u₀ = Frame(x₀, [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)])

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

# Torus
𝕋 = Torus(4.0,1.0)
x₀ = [0.,0]
u₀ = Frame(x₀, [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)])

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋)
X  = map(y -> F(Π(y), 𝕋), U.yy)

using Plots
plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("Torusplots.jl"); plotly()
TorusPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕋)


function SimulatePoints(n, u₀, ℳ::TM) where {TM <: EmbeddedManifold}
    out = Frame[]
    while length(out) < n
        W = sample(0.:dt:T, Wiener{ℝ{2}}())
        U = StochasticDevelopment(W, u₀, ℳ)
        push!(out, U.yy[end])
    end
    return out
end

@time Ξ = SimulatePoints(10, u₀, 𝕊)

ξ = map(y->F(Π(y), 𝕊), Ξ)
SphereScatterPlot(extractcomp(ξ ,1), extractcomp(ξ,2), extractcomp(ξ,3), F(x₀,𝕊), 𝕊 )

# """
#     Now let us create a stochastic process on the frame bundle of the paraboloid
# """
#
# ℙ = Paraboloid(2.0, 1.0)
#
# x₀ = [1.0,1.0]
# u₀ = Frame(x₀, [1. 0. ; 0. 2.])
#
# W = sample(0:dt:T, Wiener{ℝ{2}}())
# U = StochasticDevelopment(W, u₀, ℙ)
# X  = map(y -> F(Π(y), ℙ), U.yy)
#
# plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])
#
# include("ParaboloidPlots.jl")
# ParaboloidPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), ℙ)
