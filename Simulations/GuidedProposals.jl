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
    Guided proposal for diffusion bridge with drift. We consider
        dUt = V(Ut)dt + Hᵢ(Ut)∘dWtⁱ
"""

T = 1.0
dt = 0.001
u₀ = Frame([π/2, 0] , [1. 0. ; 0.  1/3] , 𝕋)
v = [3π/2, π/2]

function ĥ(t, y, K, 𝕋)
    HeatKernel(t, y, T, v, K, 𝕋)/HeatKernel(0, Π(u₀), T, v, K, 𝕋)
end

# Setting a vector field on the Torus
V(y, 𝕋) = [y[1]+π, y[2]+π]

# Lift of V
V⁺(u, 𝕋) = TangentFrame(u, V(Π(u), 𝕋) , u.ν)


# Set up the drift for the guided proposal Uᵒ
function Vᵒ(t, u, 𝕋)
    ∇logh = ForwardDiff.gradient(y -> log(ĥ(t, y, 100, 𝕋)), u.x)
#    return sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logp)[i] for i in eachindex(∇logp)])
    return V⁺(u, 𝕋) + sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logh)[i] for i in eachindex(∇logh)])
end

"""

We have
    dℙ⁺/dℙ⁰(Uᵒ) ∝ exp{-∫₀ᵗ V⁺ĥ(s, U_s)/ĥ(s, U_s) ds }

"""
function llikelihood!(U::SamplePath, W::SamplePath, 𝕋)
    tt = U.tt
    uu = U.yy
    ww = W.yy

    som::Float64 = 0.
    u::typeof(u₀) = u₀
    for k in 1:length(tt)-1
        ds = tt[k+1] - tt[k]
        s = tt[k]

        dw = ww[k+1] - ww[k]
        uu[..,k] = u

        # Forward simulation of the process
        ∇logh = ForwardDiff.gradient(y -> log(ĥ(s, y, 100, 𝕋)), u.x)
        vᵒ = V⁺(u, 𝕋) + sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logh)[i] for i in eachindex(∇logh)])
        u = IntegrateStep(dw, u, 𝕋) + vᵒ*ds

        # Extra likelihood term
        som += dot(V(u.x, 𝕋), ∇logh)*ds
    end
    uu[.., length(tt)] = u
    som
end



"""
    Take MCMC steps to update the driving BMs
"""

using ProgressMeter

function MCMC(iterations)
    W = sample(0:dt:T, Wiener{ℝ{2}}())
    U = StochasticDevelopment(W, u₀, 𝕋; drift = false)
    Uᵒ = deepcopy(U)
    ll = llikelihood!(Uᵒ, W, 𝕋)

    Xᵒ  = map(y -> F(Π(y), 𝕋), Uᵒ.yy)

    UUᵒ = [Uᵒ]
    XXᵒ = [Xᵒ]

    acc = 0
    ρ = .5
    ll_array = [ll]
    p = Progress(iterations, 1, "Computing initial pass...", 50)
    for iter in 1:iterations
        W₂ = sample(0:dt:T, Wiener{ℝ{2}}())
        Wᵒ = copy(W)
        Wᵒ.yy .= ρ*W.yy + sqrt(1-ρ^2)*W₂.yy

        # Simulate a proposal and compute the log-likelihood
        llᵒ = llikelihood!(Uᵒ, Wᵒ, 𝕋)
        push!(ll_array, llᵒ)

        if log(rand()) <= llᵒ - ll
            push!(UUᵒ, Uᵒ)
            push!(XXᵒ, map(y -> F(Π(y), 𝕋), Uᵒ.yy) )
            ll = llᵒ
            acc += 1
        end
        next!(p)
    end
    return UUᵒ, XXᵒ, ll_array, acc
end


UUᵒ, XXᵒ, ll, acc = MCMC(50)


plotly()
fig = TorusPlot(extractcomp(XXᵒ[1],1), extractcomp(XXᵒ[1],2), extractcomp(XXᵒ[1],3), 𝕋)
for i in max(acc-5, 0):acc
    TorusPlot!(fig, extractcomp(XXᵒ[i],1), extractcomp(XXᵒ[i],2), extractcomp(XXᵒ[i],3), 𝕋)
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
