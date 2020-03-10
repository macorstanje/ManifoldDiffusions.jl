"""
    Elements of F(ℳ) consist of a position x and a GL(d, ℝ)-matrix ν that
    represents a basis for 𝑇ₓℳ
"""

# A frame, represented by a matrix ν at an element x of a manifold ℳ
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

# A tangent vector (ẋ, ν̇) ∈ 𝑇ᵤF(ℳ)
struct TangentFrame{Tx,Tν}
    u::Frame
    ẋ::Tx
    ν̇::Tν
    function TangentFrame(u, ẋ::Tx, ν̇::Tν) where {Tx, Tν <: Union{AbstractArray, Real}}
        new{Tx,Tν}(u, ẋ, ν̇)
    end
end

"""
    Some generic functions for calculations on TF(ℳ)
"""

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


# Canonical projection Π: F(ℳ) → ℳ
Π(u::Frame{Tx, Tν, TM}) where {Tx,Tν, TM} = u.x

# Pushforward map of the canonocal projection Πˣ: TF(ℳ) → Tℳ
Πˣ(X::TangentFrame{Tx, Tν}) where {Tx, Tν} = X.ẋ

# The group action of a frame on ℝᵈ
FrameAction(u::Frame{Tx, Tν, TM}, e::T) where {Tx,Tν,T<:Union{AbstractArray, Real}, TM} = TangentVector(u.x, u.ν*e, u.ℳ)

# Horizontal lift of the orthogonal projection
Pˣ(u::Frame, ℳ::T) where {T<:EmbeddedManifold} = TangentFrame(u, u.x, P(u.x, ℳ))

"""
    Horizontal vector fields
"""

# Horizontal vector (a tangent frame) corresponding to the i'th unit vector
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
