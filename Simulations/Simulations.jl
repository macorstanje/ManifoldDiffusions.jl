include("../src/Manifolds.jl")

"""
    On the Ellipse
"""

𝔼 = Ellipse(2.0,1.0)

# Visualization of the parallel vector field
function line(q, p) # tangent vector p to a point q
    t = collect(0:0.5:1)
    [q[1].+p[1].*t  q[2].+p[2].*t]
end

# Stochastic Horizontal Development
c₀ = 0.
u₀ = Frame(c₀, -1.0)

W = sample(0:dt:T, Wiener{ℝ{1}}())
U = StochasticDevelopment(W, u₀, 𝔼)
X  = map(y -> F(Π(y), 𝔼), U.yy)

plotly()

# Plot of the process (x(t), ν(t)) local coordinates
plot(U.tt, [Π.(U.yy), map(u->u.ν, U.yy)])

# Plot of x, y-values of the 2-dimensional process
plot(U.tt, [extractcomp(X,1), extractcomp(X,2)])

# plot of the process in 2D on the Ellipse
plot(extractcomp(total,1), extractcomp(total, 2), linestyle = :dot, label = "Ellipse")
plot!(extractcomp(X,1), extractcomp(X,2))

sum(diff(Π.(U.yy)).^2)

ξξ = map(u->u.ν, U.yy)
vv = map(n-> ForwardDiff.derivative(x->F(x,𝔼), n), Π.(U.yy)).*ξξ

# Plot of the process + vectors representing the frames.
plotly()
total = map(x -> F(π*x, 𝔼), collect(0:0.01:2))
fig = plot(extractcomp(total,1), extractcomp(total, 2), linestyle = :dot,  label = "Ellipse")
plot!(fig, extractcomp(X,1), extractcomp(X, 2) , label = "X")
for i in 0:100:length(tt)
    plot!(fig, line(X[i+1],vv[i+1])[:,1], line(X[i+1],vv[i+1])[:,2], label = "t = $(U.tt[i+1])")
end
fig


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
