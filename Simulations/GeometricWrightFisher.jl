using LinearAlgebra
include("../src/ManifoldDiffusions.jl")
using Main.ManifoldDiffusions
using StaticArrays
using Einsum
using ForwardDiff
using Bridge
using Plots

extractcomp(v, i) = map(x->x[i], v)
const ℝ{N} = SVector{N, Float64}

struct Simplex{T<:Int64} <: Manifold
    n::T
    function Simplex(n::T) where {T<:Int64}
        if n<=0
            error("Dimension must be a positive integer")
        end
        new{T}(n)
    end
end

inSimplex(x) = all([x[i]>0 && x[i]<1 for i in eachindex(x)]) && sum(x) == 1
inH(x) = (sum(x)==0)

𝒞(p) = p./sum([p[i] for i in eachindex(p)])

pertubate(p, q) = 𝒞(p.*q)
power(α, p) = 𝒞(p.^α)

# Aitchison inner product
function innerA(p,q)
    n = length(p)
    return sum([log(p[i]/p[j])*log(q[i]/q[j])/(2*n) for i in eachindex(p) for j in eachindex(p)])
end
innerH(p,q) = dot(p,q)

# Center log-ration transform Δ → H
function clr(p)
    g = prod(p)^(1/length(p))
    return log.(p./g)
end

# Aitchison distance
function dA²(p,q)
    n = length(p)
    return sum([(log(p[i]/p[j]) - log(q[i]/q[j]))^2 for i in eachindex(p) for j in eachindex(p)])/(2*n)
end
#
sfm(x) = 𝒞(exp.(x))

"""
    Riemannian geometry
"""

# g(x,u,v) = sum([sum(x)*u[i]*v[i]/x[i] for i in eachindex(u)])
# Riemannian metric
function g(x::T, Δ::Simplex) where {T<:Union{AbstractArray, Real}}
    if Δ.n == 1
        return 1.0
    else
     return diagm([sum(x)/x[i] for i in eachindex(x)])
end
end

function gˣ(p::T, Δ::Simplex) where {T<:Union{AbstractArray, Real}}
    diagm(p) - p*transpose(p)
end

function Main.ManifoldDiffusions.Γ(x::T, Δ::Simplex) where {T<:Union{AbstractArray, Real}}
    if Δ.n == 1
        ∂g = ForwardDiff.derivative(y -> g(y,Δ), x)
        g⁻¹ = 1/g(x, Δ)
        return .5*g⁻¹*∂g
    end
    if Δ.n > 0
        ∂g = reshape(ForwardDiff.jacobian(y -> g(y,Δ), x), Δ.n, Δ.n, Δ.n)
        g⁻¹ = gˣ(x, Δ)
        @einsum out[i,j,k] := .5*g⁻¹[i,l]*(∂g[k,l,i] + ∂g[l,j,k] - ∂g[j,k,l])
        return out
    end
end

# function Hamiltonian(x, p, Δ::Simplex)
#     .5*p'*gˣ(x, Δ)*p
# end
#
# function Integrate(H, tt, x₀, p₀, Δ::Simplex)
#     N = length(tt)
#     x, p = x₀, p₀
#     xx, pp = [x], [p]
#     for i in 1:(N-1)
#         dt = tt[i+1]-tt[i]
#         dp = ForwardDiff.gradient(x -> -H(x, p, Δ), x).*dt
#         p += .5*dp
#         dx = ForwardDiff.gradient(p -> H(x, p, Δ), p).*dt
#         x += dx
#         dp = ForwardDiff.gradient(x -> -H(x, p, Δ), x).*dt
#         p += .5*dp
#         push!(xx, x)
#         push!(pp, p)
#     end
#     return xx,pp
# end
#
# function Geodesic(x₀, v₀, tt, Δ::Simplex)
#     xx, vv = Integrate(Hamiltonian, tt, x₀, v₀, Δ)
#     return xx, vv
# end
#
# function ExponentialMap(x₀, v₀ , tt, Δ::Simplex)
#     tt = collect(0,0.01,1.0)
#     xx, vv = Geodesic(x₀,v₀,tt,Δ)
#     return xx[end]
# end
#
# function ParallelTransport(γ, γ̇, V₀, tt, Δ::Simplex)
#     N = length(tt)
#     V = V₀
#     VV = [V₀]
#     for n in 1:(N-1)
#         dt = tt[n+1] - tt[n]
#         _Γ = Γ(γ[n], Δ)
#         _γ̇ = γ̇[n]
#         if length(V₀)>1
#             @einsum dV[i] := - _Γ[i,j,k]*V[j]*_γ̇[k]*dt
#         else
#             dV = -_Γ*V*_γ̇*dt
#         end
#         V += dV
#         push!(VV, V)
#     end
#     return VV
# end
#
# function Hor(i::Int64, u::Frame, Δ::Simplex)
#     x, ν = u.x, u.ν
#     _Γ = Γ(x, Δ)
#     if length(x)>1
#      @einsum dν[i,k,m] := -ν[j,i]*ν[l,m]*_Γ[k,j,l]
#      return TangentFrame(u, ν[:,i], dν[i,:,:])
#     else
#      return TangentFrame(u, ν, -ν^2*_Γ)
#     end
# end


Δ = Simplex(1)
tt = collect(0:0.01:10)
W = sample(tt, Wiener())

x₀ = 1/2
u₀ = Frame(x₀, 1/4 , Δ)
X = StochasticDevelopment(W, u₀ , Δ; drift=false)

plotly()
K=10
XX = [X]
plt1 = plot(X.tt, map(x -> clr(Π(x)), X.yy), legend = false)
for i in 2:K
    sample!(W, Wiener())
    push!(XX, StochasticDevelopment(W, u₀ , Δ; drift=false))
    plot!(plt1, tt, map(x -> clr(Π(x)), XX[i].yy), legend=false)
end
display(plt1)



v =
Vᵒ(t,y, Δ::Simplex) =
