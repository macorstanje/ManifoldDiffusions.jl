include("../src/Manifolds.jl")

"""
    On the circle 𝕊, the transition density of Brownian motion is given by

    1/√(4πt) ∑ exp(-(y-2kπ)²/4t)
"""

# Construct unit circl3e
ℂ = Ellipse(1.0 , 1.0)

# heat kernel ℙ(Xt ∈ dy | X_s = x) = p(s, x, t, y)dy
function HeatKernel(s, x , t, y, K , ℂ)
    out = sum([exp(-(x-y-2*k*π)^2/(4*(t-s))) for k in -K:K])
    return out/sqrt(4*π*(t-s))
end


"""
    Consider a diffusion bridge on the circle starting at (1,0), ending at v = (-1,0)
    We need a guiding term
     V(t, u) = ∑ Hᵢ(u)u⁻¹∇log p(t, u ; T, v)
"""

T = 1.0
dt = 0.001
v = π

 function V(t, u, ℂ)
     ∇logp = ForwardDiff.derivative(x -> log(HeatKernel(t, x, T, v, 100, ℂ)), u.x)
     return Hor(1, u, ℂ)*inv(u.ν)*∇logp
 end

u₀ = Frame(0.0 , 1.0 , ℂ)

W = sample(0:dt:T, Wiener{ℝ{1}}())
U = StochasticDevelopment(W, u₀, ℂ, drift = true)
X  = map(y -> F(Π(y), ℂ), U.yy)

plotly()
plot(U.tt, [extractcomp(X,1), extractcomp(X,2)])
plot(U.tt, map(u->u.x ,U.yy))
total = map(x -> F(π*x, ℂ), collect(0:0.01:2))
# plot of the process in 2D on the Ellipse
plot(extractcomp(total,1), extractcomp(total, 2), linestyle = :dot, label = "Ellipse")
plot!(extractcomp(X,1), extractcomp(X,2))
