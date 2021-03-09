include("../src/ManifoldDiffusions.jl")
using ManifoldDiffusions
using ProgressMeter
using LinearAlgebra
using ForwardDiff
using Bridge
using StaticArrays
const ℝ{N} = SVector{N, Float64}

"""
    On the circle 𝕊, the transition density of Brownian motion is given by

    1/√(4πt) ∑_{k∈ℤ²} exp(-(y-2kπ)²/4t)
"""

# Construct unit Torus
𝕋 = Torus(3.0 , 1.0)
K = 20

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
TimeChange(T) = (x) ->  x * (2-x/T)
tt = TimeChange(T).(0.:dt:T)

u₀ = Frame([π/2, 0] , [1. 0. ; 0.  1/3] , 𝕋)
v = [3π/2, π]

h₀ = HeatKernel(0, Π(u₀), T, v, K, 𝕋)

# Setting a vector field on the Torus
V(y, θ, 𝕋) = [0. , θ*π ]
# Lift of V
V⁺(u, θ, 𝕋) = TangentFrame(u, V(Π(u), θ, 𝕋) , u.ν)

# Three dimensional representation of V
# ForwardDiff.jacobian(x->F(x,𝕋), u₀.x)*V(u₀.x, θ,  𝕋)



# Simulate U forward with $θ=0.5
Vᵒ(t, u, 𝕋) = V⁺(u, 0.5, 𝕋)

W = sample(tt, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋; drift=true)

# pick 10 times in [0,1]
n = 10 # amount of observations
indices = sample(2:1:length(U.tt)-1, n-2, replace=false, ordered=true)
pushfirst!(indices, 1)
push!(indices, length(U.tt))
τ = U.tt[indices]
# Select observations
ξ = map(u -> Π(u), U.yy[indices])
Ξ = map(y -> F(y, 𝕋), ξ)

X = map(y -> F(Π(y), 𝕋), U.yy)

plotly()
TorusPlot(extractcomp(X, 1), extractcomp(X, 2), extractcomp(X, 3), 𝕋)
plot!(extractcomp(Ξ, 1), extractcomp(Ξ, 2), extractcomp(Ξ, 3), seriestype = :scatter, markersize = 2.0)

"""
For computational ease, we first compute
p̂[j] = p̂(t_{j-1}, ξ_{j-1} ; t_{j} , ξ_{j})
"""
p = [HeatKernel(τ[1], ξ[1], τ[2], ξ[2], K, 𝕋)]
for j in 2:length(indices)-1
    push!(p, HeatKernel(τ[j], ξ[j], τ[j+1], ξ[j+1], K, 𝕋))
end

function ρ̂(t, y, 𝕋)
    k = findmin(abs.(τ.-t))[2]
    if τ[k] <= t
        k += 1
    end
    if k == length(τ)
        return HeatKernel(t, y, τ[end], ξ[end], K, 𝕋)
    else
        return HeatKernel(t, y, τ[k], ξ[k], K, 𝕋)*prod([p[j] for j in k:length(p)])
    end
end

function ĥ(t, y, 𝕋)
    ρ̂(t, y, 𝕋)/h₀
end

# Set up the drift for the guided proposal Uᵒ
function Vᵒ(t, u, 𝕋)
    ∇logh = ForwardDiff.gradient(y -> log(ĥ(t, y, 𝕋)), u.x)
#    return sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logp)[i] for i in eachindex(∇logp)])
    return V⁺(u, θ, 𝕋) + sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logh)[i] for i in eachindex(∇logh)])
end

θ=0.5


W = sample(tt, Wiener{ℝ{2}}())
StochasticDevelopment!(Uᵒ, W, u₀, 𝕋; drift = true)

Xᵒ = map(y -> F(Π(y), 𝕋), Uᵒ.yy)
plotly()
TorusPlot(extractcomp(X, 1), extractcomp(X, 2), extractcomp(X, 3), 𝕋)
plot!(extractcomp(Xᵒ, 1), extractcomp(Xᵒ, 2), extractcomp(Xᵒ, 3), linewidth = 2.0)
plot!(extractcomp(Ξ, 1), extractcomp(Ξ, 2), extractcomp(Ξ, 3), seriestype = :scatter, markersize = 2.0)

"""

We have
    dℙ⁺/dℙ⁰(Uᵒ) ∝ exp{-∫₀ᵗ V⁺ĥ(s, U_s)/ĥ(s, U_s) ds }

"""

function loglikelihood(U::SamplePath, W::SamplePath, θ, 𝕋)
    tt = U.tt
    uu = U.yy
    ww = W.yy

    som::Float64 = 0.
    for k in 1:length(tt)-1
        ds = tt[k+1] - tt[k]
        s = tt[k]

        u = U.yy[k]

        ∇logh = ForwardDiff.gradient(y -> log(ĥ(s, y, 𝕋)), u.x)

        # Extra likelihood term
        som += dot(V(u.x,θ, 𝕋), ∇logh)*ds
    end
    som
end

function loglikelihood!(U::SamplePath, W::SamplePath, u₀, θ, 𝕋)
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
        ∇logh = ForwardDiff.gradient(y -> log(ĥ(s, y, 𝕋)), u.x)
        vᵒ = V⁺(u, θ, 𝕋) + sum([Hor(i, u, 𝕋)*(inv(u.ν)*∇logh)[i] for i in eachindex(∇logh)])
        u = IntegrateStep(dw, u, 𝕋) + vᵒ*ds

        # Extra likelihood term
        som += dot(V(u.x,θ, 𝕋), ∇logh)*ds
    end
    uu[.., length(tt)] = u
    som
end

function loglikelihood(W::SamplePath, u₀, θ, 𝕋)
    let U = Bridge.samplepath(W.tt, zero(u₀)); ll=loglikelihood!(U, W, u₀, θ, 𝕋); U,ll end
end

loglikelihood(W, u₀, 0.5, 𝕋)

function UpdateBridges!(W, U, ρ, θ, τ)
    acc = zeros(length(τ)-1)
    u₀ = U.yy[1]
    for i  in 1:length(τ)-1
        # indices of τ[i] and  τ[i+1] in the array W.tt
        i⁻ = findall(x -> x == τ[i], W.tt)[1]
        i⁺ = findall(x -> x == τ[i+1], W.tt)[1]
        W₂ = sample(W.tt[i⁻:1:i⁺], Wiener{ℝ{2}}(), W.yy[i⁻])
        Wᵒ = copy(W₂)
        Wᵒ.yy .= ρ*W.yy[i⁻:1:i⁺] + sqrt(1-ρ^2)W₂.yy
        Uᵒ, llᵒ = loglikelihood(Wᵒ, U.yy[i⁻], θ, 𝕋)

        ll = loglikelihood(U[i⁻:1:i⁺], W[i⁻:1:i⁺], θ, 𝕋)
        if log(rand()) <= llᵒ - ll
            W.yy[i⁻:1:i⁺] .= Wᵒ.yy
            U.yy[i⁻:1:i⁺] .= Uᵒ.yy
            ll = llᵒ
            acc[i] += 1
        end
    end
    acc
end

W = sample(tt, Wiener{ℝ{2}}())
U = StochasticDevelopment(W, u₀, 𝕋;drift=true)

X = map(y -> F(Π(y), 𝕋), U.yy)

plotly()
TorusPlot(extractcomp(X, 1), extractcomp(X, 2), extractcomp(X, 3), 𝕋)
plot!(extractcomp(Ξ, 1), extractcomp(Ξ, 2), extractcomp(Ξ, 3), seriestype = :scatter, markersize = 2.0)

Uᵒ = copy(U)
UpdateBridges!(W, Uᵒ, .5, .5, τ)

Xᵒ = map(y -> F(Π(y), 𝕋), Uᵒ.yy)
plotly()
TorusPlot(extractcomp(X, 1), extractcomp(X, 2), extractcomp(X, 3), 𝕋)
plot!(extractcomp(Xᵒ, 1), extractcomp(Xᵒ, 2), extractcomp(Xᵒ, 3), linewidth = 2.0)
plot!(extractcomp(Ξ, 1), extractcomp(Ξ, 2), extractcomp(Ξ, 3), seriestype = :scatter, markersize = 2.0)


"""
    Take MCMC steps to update the driving BMs
"""

function adaptstepsize!(δ, n, accinfo)
    adaptskip = 10
    if mod(n,adaptskip)==0
        η(n) = min(0.1, 10/sqrt(n))

        targetaccept = 0.5

        recent_mean = ( accinfo[end] - accinfo[end-adaptskip+1] )/adaptskip
        if recent_mean > targetaccept
            δ *= exp(η(n))
        else
            δ *= exp(-η(n))
        end
    end
end


function MCMC(iterations, ε)
    W = sample(0:dt:τ[end], Wiener{ℝ{2}}())
    U = StochasticDevelopment(W, u₀, 𝕋; drift = false)
    Uᵒ = deepcopy(U)
    θ = 2*rand()-1
    ll = llikelihood!(Uᵒ, W, θ,  𝕋)

    X  = map(y -> F(Π(y), 𝕋), Uᵒ.yy)

    θθ = [θ]
    UU = [Uᵒ]
    XX = [X]

    acc = zeros(length(τ)-1)
    acc_θ = [0]
    ρ = .5
    ll_array = [ll]
    p = Progress(iterations, 1, "Percentage completed ...", 50)
    for iter in 1:iterations

        # Update antidevelopment
        # W₂ = sample(0:dt:τ[end], Wiener{ℝ{2}}())
        # Wᵒ = copy(W)
        # Wᵒ.yy .= ρ*W.yy + sqrt(1-ρ^2)*W₂.yy
        #
        # # Simulate a proposal and compute the log-likelihood
        # llᵒ = llikelihood!(Uᵒ, Wᵒ, θ, 𝕋)
        # if log(rand()) <= llᵒ - ll
        #     U = Uᵒ
        #     X = map(y -> F(Π(y), 𝕋), Uᵒ.yy)
        #     ll = llᵒ
        #     acc += 1
        # end

        acc += UpdateBridges!(W, U, ρ, θ, τ)

        # Update paremter
        θᵒ = θ + ε*(2*rand()-1)
        llᵒ = loglikelihood!(Uᵒ, W, u₀, θᵒ, 𝕋)
        if log(rand()) <= llᵒ - ll
            U = Uᵒ
            X = map(y -> F(Π(y), 𝕋), Uᵒ.yy)
            θ = θᵒ
            ll = llᵒ
            push!(acc_θ, acc_θ[end] + 1)
        else
            push!(acc_θ, acc_θ[end])
        end
        push!(UU, U)
        push!(XX, X)
        push!(θθ, θ)
        next!(p)
        adaptstepsize!(ε, iter, acc_θ)
    end
    return UU, XX, θθ, acc, acc_θ
end


UU, XX, θθ, acc, acc_θ = MCMC(200, 0.1)
acc_θ
plotly()
Plots.plot(θθ)

fig = TorusPlot(extractcomp(XX[1],1), extractcomp(XX[1],2), extractcomp(XX[1],3), 𝕋)
for i in max(acc_θ[end]-5+1, 1):acc_θ[end]+1
    plot!(fig, extractcomp(XX[i],1), extractcomp(XX[i],2), extractcomp(XX[i],3), linewidth = 2.0)
end
# Plots.plot!([F(u₀.x, 𝕋)[1]], [F(u₀.x, 𝕋)[2]], [F(u₀.x, 𝕋)[3]],
#             seriestype = :scatter,
#             color= :red,
#             legend = true,
#             markersize = 2.5,
#             label = "Start")
Plots.plot!(extractcomp(Ξ, 1), extractcomp(Ξ, 2), extractcomp(Ξ, 3),
            seriestype = :scatter,
            legend = true,
            color = :blue,
            markersize = 2.0,
            label = "Observations")
display(fig)
