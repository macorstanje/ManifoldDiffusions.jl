include("../src/Manifolds.jl")

"""
    On the circle 𝕊, the transition density of Brownian motion is given by

    1/√(4πt) ∑_{k∈ℤ²} exp(-(y-2kπ)²/4t)
"""

# Construct unit Torus
𝕋 = Torus(3.0 , 1.0)

# heat kernel ℙ(Xt ∈ dy | X_s = x) = p(s, x, t, y)dy
function HeatKernel(s, x , t, y, K , 𝕋)
    out = sum([exp(-norm(x-y-[2*k*π, 2*l*π])^2/(4*(t-s))) for k in -K:K, l in -K:K])
    return out/(4*π*(t-s))
end


"""
    Consider a diffusion bridge on the circle starting at (1,0), ending at v = (-1,0)
    We need a guiding term
     V(t, u) = ∑ Hᵢ(u)u⁻¹∇log p(t, πu ; T, v)
"""

T = 1.0
dt = 0.001
v = [π, -π/2]
F(v, 𝕋)
function V(t, u, ℂ)
    ∇logp = ForwardDiff.gradient(x -> log(HeatKernel(t, x, T, v, 100, 𝕋)), u.x)
    return sum([Hor(i, u, ℂ)*(inv(u.ν)*∇logp)[i] for i in 1:2])
end

u₀ = Frame([π/3, π] , [1.0 0. ; 0.  1. ] , 𝕋)

W = sample(0:dt:T, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋, drift = true)
X  = map(y -> F(Π(y), 𝕋), U.yy)

plotly()
plot(U.tt, [extractcomp(X,1), extractcomp(X,2), extractcomp(X,3)])

TorusPlot(extractcomp(X,1), extractcomp(X,2), extractcomp(X,3), 𝕋)
