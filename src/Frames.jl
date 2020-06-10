"""
    Frame{Tx, Tν, TM}

Elements of ``\\mathrm{F}(\\mathcal{M})`` consist of a position `x::Tx` on `ℳ<:EmbeddedManifold` and a
``\\mathrm{GL}(d, \\mathbb{R})``-matrix `ν::Tν` that consists of column vectors that form
a basis for T_xℳ. All input is assumed to be in local coordinates that coincide
with `F( ,ℳ)`.

# Example: A frame on the south pole on the sphere

```julia-repl
julia> 𝕊 = Sphere(1.0)
julia> u = Frame([0. , 0.], [1. 0. ; 0.  1.], 𝕊)
julia> u.x # returns [0. , 0.]
julia> u.ν # returns [1. 0. ; 0. 1.]
```
"""
struct Frame{Tx, Tν, TM}
    x::Tx
    ν::Tν
    ℳ::TM
    function Frame(x::Tx, ν::Tν, ℳ::TM) where {Tx, Tν <: Union{AbstractArray, Real}, TM<:EmbeddedManifold}
        # if rank(ν) != length(x)
        #     error("A is not of full rank")
        # end
        new{Tx, Tν, TM}(x, ν, ℳ)
    end
end

"""
    TangentFrame{Tx, Tν}

A tangent vector ``(\\dot{x}, \\dot{\\nu}) \\in T_u\\mathrm{F}(\\mathcal{M})``.
This object consists of the frame `u::Frame` that it is tangent to and the
velocities `ẋ` and `ν̇`.

# Example:
```julia-repl
julia> 𝕊 = Sphere(1.0)
julia> u = Frame([0. , 0.], [1. 0. ; 0.  1.], 𝕊)
julia> V = TangentFrame(u, [1. 0.] , [-0.1 0. ; -0.5 1.])
```
"""
struct TangentFrame{Tx,Tν}
    u::Frame
    ẋ::Tx
    ν̇::Tν
    function TangentFrame(u, ẋ::Tx, ν̇::Tν) where {Tx, Tν <: Union{AbstractArray, Real}}
        new{Tx,Tν}(u, ẋ, ν̇)
    end
end

Base.zero(u::Frame{Tx, Tν}) where {Tx, Tν} = Frame(zero(u.x), one(u.ν), u.ℳ)

# Vector space operations on 𝑇ᵤF(ℳ)

function Base.:+(X::TangentFrame{Tx, Tν}, Y::TangentFrame{Tx,Tν}) where {Tx,Tν}
    # if X.u != Y.u
    #     error("Vectors are in different tangent spaces")
    # end
    return TangentFrame(X.u, X.ẋ + Y.ẋ, X.ν̇ + Y.ν̇)
end

function Base.:-(X::TangentFrame{Tx, Tν}, Y::TangentFrame{Tx,Tν}) where {Tx,Tν}
    if X.u != Y.u
        error("Vectors are in different tangent spaces")
    end
    return TangentFrame(X.u, X.ẋ - Y.ẋ, X.ν̇ - Y.ν̇)
end

# this function should be the exponential map on F(ℳ)
function Base.:+(u::Frame{Tx, Tν, TM}, X::TangentFrame{Tx, Tν}) where {Tx,Tν, TM}
    # if X.u != u
    #     error("X is not tangent to u")
    # end
    return Frame(u.x + X.ẋ , u.ν + X.ν̇, u.ℳ)
end

function Base.:*(X::TangentFrame{Tx, Tν}, y::Float64) where {Tx, Tν}
    TangentFrame(X.u , X.ẋ.*y , X.ν̇.*y)
end

function Base.:*(y::Float64, X::TangentFrame{Tx, Tν}) where {Tx, Tν}
    TangentFrame(X.u , X.ẋ.*y , X.ν̇.*y)
end

"""
    Π(u::Frame)

Canonical projection ``Π: \\mathrm{F}(\\mathcal{M}) \\to \\mathcal{M}`` that
maps ``(x,\\nu)`` to ``x``.
"""
Π(u::Frame{Tx, Tν, TM}) where {Tx,Tν, TM} = u.x

"""
    Πˣ(X::TangentFrame)

Pushforward map of the canonocal projection ``Π^*: T\\mathrm{F}(\\mathcal{M})
\\to T\\mathcal{M}`` that maps ``(\\dot{x}, \\dot{\\nu})`` to ``\\dot{x}``
"""
Πˣ(X::TangentFrame{Tx, Tν}) where {Tx, Tν} = X.ẋ

# The group action of a frame on ℝᵈ
FrameAction(u::Frame{Tx, Tν, TM}, e::T) where {Tx,Tν,T<:Union{AbstractArray, Real}, TM} = TangentVector(u.x, u.ν*e, u.ℳ)

# Horizontal lift of the orthogonal projection
Pˣ(u::Frame, ℳ::T) where {T<:EmbeddedManifold} = TangentFrame(u, u.x, P(u.x, ℳ))

"""
    Hor(i::Int64, u::Frame, ℳ::TM) where {TM<:EmbeddedManifold}

Returns the horizontal vector ``H_i(u)`` in ``T_u\\mathrm{F}(\\mathcal{M})`` as
an element of type `TangentFrame`.
"""
function Hor(i::Int64, u::Frame, ℳ::TM) where {TM<:EmbeddedManifold}
    x, ν = u.x, u.ν
    _Γ = Γ(x, ℳ)
    if length(x)>1
        @einsum dν[i,k,m] := -ν[j,i]*ν[l,m]*_Γ[k,j,l]
        return TangentFrame(u, ν[:,i], dν[i,:,:])
    else
        return TangentFrame(u, ν, -ν^2*_Γ)
    end
end
