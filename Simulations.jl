include("Definitions.jl")
include("Frames.jl")
include("Geodesics.jl")
include("FrameBundles.jl")

using Plots

"""
    On the Ellipse
"""
𝔼 = Ellipse(1.0,2.0)

# Parallel transport
tt = collect(0:0.01:0.5)
ct = π.*tt
γ = map(x -> F( x , 𝔼) , ct)
total = map(x -> F(π*x, 𝔼), collect(0:0.01:2))

plot(extractcomp(total,1), extractcomp(total, 2), label = "Ellipse")
plot!(extractcomp(γ,1), extractcomp(γ, 2) , label = "γ")

dct = zeros(length(ct)) .+ π
ξ₀ = 1.0

ξξ = ParallelTransport(ct, dct, ξ₀, tt, 𝔼)

vv = map(n-> ForwardDiff.derivative(x->F(x,𝔼), n), ct).*ξξ

# Visualization of the parallel vector field
function line(q, p) # tangent vector p to a point q
    t = collect(0:0.5:1)
    [q[1].+p[1].*t  q[2].+p[2].*t]
end

fig = plot(extractcomp(total,1), extractcomp(total, 2), label = "Ellipse")
plot!(fig, extractcomp(γ,1), extractcomp(γ, 2) , label = "γ")
for i in 0:1:10
    plot!(fig, line(γ[5*i+1],vv[5*i+1])[:,1], line(γ[5*i+1],vv[5*i+1])[:,2])
end
fig


c₀ = 0.
u₀ = Frame(c₀, 1.0)

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{1}}())
U = StochasticDevelopment(W, u₀, 𝔼)
X  = map(y -> F(Π(y), 𝔼), U.yy)

plot(U.tt, [extractcomp(X,1), extractcomp(X,2)])

plot(extractcomp(total,1), extractcomp(total, 2), label = "Ellipse")
plot!(extractcomp(X,1), extractcomp(X,2))
"""
    On the Sphere
"""

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

plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("Sphereplots.jl"); plotly()
SpherePlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕊)

"""
    On the Torus
"""

𝕋 = Torus(4.0,1.0)
x₀ = [0.,0]
u₀ = Frame(x₀, [1. 0.; 0. 1.])

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋)
X  = map(y -> F(Π(y), 𝕋), U.yy)

plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("Torusplots.jl"); plotly()
TorusPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕋)

"""
    On the paraboloid
"""

ℙ = Paraboloid(1.0, 1.0)

x₀ = [1.0,1.0]
u₀ = Frame(x₀, [1. 0. ; 0. 2.])

W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, ℙ)
X  = map(y -> F(Π(y), ℙ), U.yy)

plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

include("ParaboloidPlots.jl")
ParaboloidPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), ℙ)
