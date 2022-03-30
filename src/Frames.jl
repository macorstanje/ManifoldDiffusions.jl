"""
    Frame{Tq, Tν, TM}

Elements of ``\\mathrm{F}(\\mathcal{M})`` consist of a position `q::Tq` on `ℳ<:EmbeddedManifold` 
and a ``\\mathrm{GL}(d, \\mathbb{R})``-matrix `ν::Tν` that consists of column vectors that form
a basis for T_xℳ. `q` and `ν` are understood to be in local coordinates in chart `n`.

# Example: A frame on the south pole on the 2-unit-sphere

```julia-repl
julia> 𝕊 = Sphere(2, 1.0)
julia> u = Frame([0. , 0.], [1. 0. ; 0.  1.], 1, 𝕊)
julia> u.q # returns [0. , 0.]
julia> u.ν # returns [1. 0. ; 0. 1.]
```
"""
struct Frame{Tq, Tν, TM}
    q::Tq
    ν::Tν
    n::Int64
    ℳ::TM

    function Frame(q::Tq, ν::Tν, n::Int64, ℳ::TM) where {Tq, Tν <: Union{AbstractArray, Real}, TM<:Manifold}
        new{Tq, Tν, TM}(q, ν, n, ℳ)
    end
end

"""
    getx(u::Frame{Tq,Tν,TM}) where {Tq,Tν,TM}

Get the projection on the manifold. 
"""
getx(u::Frame{Tq,Tν,TM}) where {Tq,Tν,TM} = ϕ⁻¹(u.q, u.n ,u.ℳ)

"""
    TangentFrame{Tx, Tν}

A tangent vector ``(\\dot{x}, \\dot{\\nu}) \\in T_u\\mathrm{F}(\\mathcal{M})``.
This object consists of the frame `u::Frame` that it is tangent to and the
velocities `q̇` and `ν̇`.

# Example:
```julia-repl
julia> 𝕊 = Sphere(1.0)
julia> u = Frame([0. , 0.], [1. 0. ; 0.  1.], 𝕊)
julia> V = TangentFrame(u, [1. 0.] , [-0.1 0. ; -0.5 1.])
```
"""
struct TangentFrame{Tq,Tν}
    u::Frame
    q̇::Tq
    ν̇::Tν

    function TangentFrame(u, q̇::Tq, ν̇::Tν) where {Tq, Tν <: Union{AbstractArray, Real}}
        new{Tq,Tν}(u, q̇, ν̇)
    end
end
Base.zero(u::Frame{Tq, Tν,TM}) where {Tq,Tν,TM} = Frame(zero(u.q), one(u.ν), u.n, u.ℳ)

# Vector space operations on 𝑇ᵤF(ℳ)
function Base.:+(X::TangentFrame{Tq, Tν}, Y::TangentFrame{Tq,Tν}) where {Tq,Tν}
    return TangentFrame(X.u, X.q̇ + Y.q̇, X.ν̇ + Y.ν̇)
end

function Base.:-(X::TangentFrame{Tq, Tν}, Y::TangentFrame{Tq,Tν}) where {Tq,Tν}
    return TangentFrame(X.u, X.q̇ - Y.q̇, X.ν̇ - Y.ν̇)
end

# this function should be the exponential map on F(ℳ)
function Base.:+(u::Frame{Tx, Tν, TM}, X::TangentFrame{Txx, Tν}) where {Tx,Txx,Tν, TM}
    return Frame(u.q+X.q̇, u.ν+X.ν̇, u.n, u.ℳ)
end

function Base.:*(X::TangentFrame{Tx, Tν}, y::Ty) where {Tx, Tν} where {Ty<:Real}
    TangentFrame(X.u , X.q̇.*y , X.ν̇.*y)
end
Base.:*(y::Ty, X::TangentFrame{Tx, Tν}) where {Tx, Tν} where {Ty<:Real}= X*y

"""
    Π(u::Frame)

Canonical projection ``Π: \\mathrm{F}(\\mathcal{M}) \\to \\mathcal{M}`` that
maps ``(x,\\nu)`` to ``x``.
"""
Π(u::Frame{Tx, Tν, TM}) where {Tx,Tν, TM} = getx(u)

"""
    Πˣ(X::TangentFrame)

Pushforward map of the canonocal projection ``Π^*: T\\mathrm{F}(\\mathcal{M})
\\to T\\mathcal{M}`` that maps ``(\\dot{x}, \\dot{\\nu})`` to ``\\dot{x}``
"""
Πˣ(X::TangentFrame{Tx, Tν}) where {Tx, Tν} = Dϕ⁻¹(u.q, u.n,u.ℳ)*X.ẋ

# The group action of a frame on ℝᵈ
FrameAction(u::Frame{Tx, Tν, TM}, e::T) where {Tx,Tν,T<:Union{AbstractArray, Real}, TM} = TangentVector(getx(u), u.ν*e, u.ℳ)

# Horizontal lift of the orthogonal projection
Pˣ(u::Frame, ℳ::T) where {T<:Manifold} = TangentFrame(u, u.q, P(u.q, ℳ))

"""
    Hor(i::Int64, u::Frame)

Returns the horizontal vector ``H_i(u)`` in ``T_u\\mathrm{F}(\\mathcal{M})`` as
an element of type `TangentFrame`.
"""
function Hor(i::Int64, u::Frame)
    q, ν, n, ℳ = u.q, u.ν, u.n, u.ℳ
    _Γ = Γ(q, n, ℳ)
    if length(q)>1
        @einsum dν[i,j,m] := -0.5*_Γ[i,k,l]*ν[k,m]*ν[l,j]
        return TangentFrame(u, ν[:,i], dν[:,:,i])
    else
        return TangentFrame(u, ν, -ν^2*_Γ)
    end
end

"""
    FrameBundle

The object `FrameBundle(ℳ)` represents the frame bundle over a manifold ``\\mathcal{M}``.
"""
struct FrameBundle{TM} <: Manifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:Manifold} = new{TM}(ℳ)
end
Dimension(FM::FrameBundle) = Dimension(FM.ℳ)+Dimension(FM.ℳ)^2

Σ(u::Frame, v::T, w::T) where {T<:AbstractArray} = dot(inv(u.ν)*v , inv(u.ν)*w)

"""
    g(X::TangentFrame, Y::TangentFrame)

Adds a Riemannian structure to the Frame bundle by introducing a cometric
"""
function g♯(X::TangentFrame, Y::TangentFrame)
        if X.u != Y.u
            error("Vectors are in different tangent spaces")
        end
    return Σ(X.u, X.q̇, Y.q̇)
end

"""
    Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM})

Returns the Hamiltonian that results from the cometric `g`.
"""
function Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    if p.u != u
        error("p is not tangent to u")
    end
    return .5*g♯(p,p)
end

"""
    Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM})

Different representation of the Hamiltonian as functions of two vectors of size
``d+d^2``
"""
function Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM}) where {Tx, Tp<:AbstractArray, TM}
    N = length(x)
    d = Int64((sqrt(1+4*N)-1)/2)
    u = Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d), Fℳ.ℳ)
    P = TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d))
    return Hamiltonian(u, P, Fℳ)
end

"""
    Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM})

Returns a geodesic on `Fℳ` starting at `u₀` with initial velocity `v₀` and
evaluated at a discretized time interval `tt`.
"""
function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = Dimension(Fℳ.ℳ)
    if d==1
        U₀ = [u₀.x, u₀.ν]
        V₀ = [v₀.ẋ, v₀.ν̇]
    else
        U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
        V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    end
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d), Fℳ.ℳ) , xx)
    vv = map(p->TangentFrame(u₀, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

"""
    ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM})

The exponential map on `Fℳ` starting from `u₀` with initial velocity `v₀`.
"""
function ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    tt = collect(0:0.01:1)
    uu, vv = Geodesic(u₀, v₀, tt, Fℳ)
    if length(u₀.x)>1
        return uu[end]
    else
        return Frame(uu[end].x[1], uu[end].ν[1], Fℳ.ℳ)
    end
end