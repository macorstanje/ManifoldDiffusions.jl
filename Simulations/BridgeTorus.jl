include("../src/Manifolds.jl")

"""
    On the circle 𝕊, the transition density of Brownian motion is given by

    1/√(4πt) ∑_{k∈ℤ²} exp(-(y-2kπ)²/4t)
"""

# Construct unit Torus
𝕋 = Torus(3.0 , 1.0)

# heat kernel ℙ(X_T ∈ dz | X_t = y) = p(t, y ; T, z)dz
function HeatKernel(t, y , T, z, K , 𝕋)
    out = sum([exp(-norm(y-z-[2*k*π, 2*l*π])^2/(4*(T-t))) for k in -K:K, l in -K:K])
    return out/(4*π*(T-t))
end

"""
    Consider a diffusion bridge on the circle starting at (1,0), ending at v = (-1,0)
    We need a guiding term
     V(t, u) = ∑ Hᵢ(u)u⁻¹∇log p(t, πu ; T, v)
"""

T = 1.0
dt = 0.001
v = [3π/2, π/2]
F(v, 𝕋)


function Vᵒ(t, u, 𝕋)
    ∇logp = ForwardDiff.gradient(x -> log(HeatKernel(t, x, T, v, 100, 𝕋)), u.x)
#    return sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logp)[i] for i in eachindex(∇logp)])
    return sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logp)[i] for i in eachindex(∇logp)])
end

u₀ = Frame([π/2, 0] , [1. 0. ; 0.  1/3] , 𝕋)

vv = ForwardDiff.jacobian(x->F(x,𝕋), Π(u₀))*u₀.ν

W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋; drift = true)
X  = map(y -> F(Π(y), 𝕋), U.yy)

WW = [W]
UU = [U]
XX = [X]
for i in 1:6
    W = sample(0:dt:T, Wiener{ℝ{2}}())
    StochasticDevelopment!(U, W, u₀, 𝕋; drift = true)
    X  = map(y -> F(Π(y), 𝕋), U.yy)
    push!(WW, W)
    push!(UU, U)
    push!(XX, X)
end

plotly()
fig = TorusPlot(extractcomp(XX[1],1), extractcomp(XX[1],2), extractcomp(XX[1],3), 𝕋)
for i in 1:6
    TorusPlot!(fig, extractcomp(XX[i],1), extractcomp(XX[i],2), extractcomp(XX[i],3), 𝕋)
end
Plots.plot!([F(u₀.x, 𝕋)[1]], [F(u₀.x, 𝕋)[2]], [F(u₀.x, 𝕋)[3]],
            seriestype = :scatter,
            color= :red,
            legend = true,
            markersize = 2.5,
            label = "Start")
Plots.plot!([F(v, 𝕋)[1]], [F(v, 𝕋)[2]], [F(v, 𝕋)[3]],
            seriestype = :scatter,
            legend = true,
            color = :blue,
            markersize = 2.5,
            label = "End")
display(fig)
