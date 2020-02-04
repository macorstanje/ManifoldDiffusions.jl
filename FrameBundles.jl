abstract type FrameBundleProcess end
include("Frames.jl")

# The frame bundle over a manifold ℳ
struct FrameBundle{TM} <: EmbeddedManifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:EmbeddedManifold} = new{TM}(ℳ)
end

"""
    Riemannian structure on the Frame bundle
"""

# returns a matrix of size d+d² × d+d²
# function g(u::Frame, Fℳ::FrameBundle{TM}) where {TM}
#     ν , ℳ, d = u.ν, Fℳ.ℳ, length(u.x)
#     δ = Matrix{eltype(u.x)}(I,d, d)
#     @einsum W⁻¹[i,j] := δ[α,β]*ν[α,i]*ν[β, j]
#     _Γ = Γ(u.x, ℳ)
#     @einsum Gamma[j, i, α] := _Γ[i,j,k]*ν[α, k]
#     Ga = reshape(Gamma, d^2, d)
#     return [W⁻¹ -W⁻¹*Ga' ; -Ga*W⁻¹ Ga*W⁻¹*Ga']
# end

# Action of g on tangent frames, conform Anisotropic covariance paper by somer et al.
# function g(X::TangentFrame, Y::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
#     if X.u != Y.u
#         error("Vectors are in different tangent spaces")
#     end
#     d = length(X.ẋ)
#     qX = vcat(X.ẋ, vec(reshape(X.ν̇, d^2, 1)))
#     qY = vcat(Y.ẋ, vec(reshape(Y.ν̇, d^2, 1)))
#     return qX'*g(X.u, Fℳ)*qY
# end
Σ(u::Frame, v::T, w::T) where {T<:AbstractArray} = dot(inv(u.ν)*v , inv(u.ν)*w)
function g(X::TangentFrame, Y::TangentFrame)
        if X.u != Y.u
            error("Vectors are in different tangent spaces")
        end
    return Σ(X.u, Πˣ(X), Πˣ(Y))
end

u = Frame([rand(),rand()], [rand() 0. ; 0. rand()])
i,j = 1,2
g(Hor(i, u, ℳ), Hor(j, u, ℳ))

function gˣ(u::Frame, Fℳ::FrameBundle{TM}) where {TM}
    return inv(g(u, Fℳ))
end

# Christoffel Symbols
function Γ(u::Frame, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u.x)
    ∂g = reshape(ForwardDiff.jacobian(x -> g(x, Fℳ)), d+d^2, d+d^2, d+d^2)
    g⁻¹ = gˣ(u, Fℳ)
    @einsum out[i,j,k] := .5*g⁻¹[i,l]*(∂g[k,l,i] + ∂g[l,j,k] - ∂g[j,k,l])
    return out
end

# Hamiltonian
function Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    P = vcat(p.ẋ, vec(reshape(p.ν̇, d^2, 1)))
    return .5*P'*gˣ(u, Fℳ)*P
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
    Geodesic flow on the frame bundle
"""
include("Geodesics.jl")

function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u.x)
    U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
    V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d)) , xx)
    vv = map(p->TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

tt = collect(0:0.01:1.0)
u₀ = Frame([0.,0.], [1. 0. ; 0. 1.])
v₀ = TangentFrame(u₀, [1.,0.] , [2. 0. ; 0. 2.])

𝕊 = Sphere(1.0)
F𝕊 = FrameBundle(𝕊)

Geodesic(u₀, v₀, tt, F𝕊)

g(u₀, F𝕊)
