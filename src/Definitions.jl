"""
    Manifold

Abstract (super-)type under which all speficic manifolds fall
"""
abstract type Manifold end

const ℝ{N} = SVector{N, Float64}
const IndexedTime = Tuple{Int64,Float64}
outer(x) = x*x'
outer(x,y) = x*y'
extractcomp(v, i) = map(x->x[i], v)

"""
    EmbeddedManifold <: Manifold

`EmbeddedManifold` is a manifold embedded in ``\\mathbb{R}^N``
"""
abstract type EmbeddedManifold <: Manifold end

"""
    Dimension(ℳ::TM) where {TM<:EmbeddedManifold}

Returns the dimension of the manifold 'ℳ'
"""
function Dimension(ℳ::TM) where {TM<:Manifold}
end

"""
    AmbiantDimension(ℳ::TM) where {TM<:EmbeddedManifold}

Returns the dimension of the ambient space of the manifold 'ℳ'
"""
function AmbientDimension(ℳ::TM) where {TM<:EmbeddedManifold}
end

"""
    TangentVector{T, TM}

Elements of ``T_x\\mathcal{M}`` and equipped with vector space operations. Given `x`
and `ẋ` in ``\\mathbb{R}^N`` and an `EmbeddedManifold` `ℳ` with ambient dimension `N`,
use `X=TangentVector(x,ẋ,ℳ)` to create a tangent vector. 

Since ``T_x\\mathcal{M}`` is a vector space, `TangentVector`s allow for addition and 
scalar multiplication. 
"""
struct TangentVector{T<:AbstractArray,TM<:EmbeddedManifold}
    x::T
    ẋ::T
    ℳ::TM
    function TangentVector(v::T, x::T, ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}
        new{T,TM}(x, v, ℳ)
    end
end

# Vector space operations on 𝑇ₓℳ
function Base.:+(X::TangentVector{T,TM}, Y::TangentVector{T,TM}) where {T,TM}
    if X.x != Y.x || X.ℳ != Y.ℳ
        error("X and Y are not in the same tangent space")
    end
    return TangentVector(X.x, X.ẋ+Y.ẋ, X.ℳ)
end

function Base.:-(X::TangentVector{T,TM}, Y::TangentVector{T,TM}) where {T,TM}
    if X.x != Y.x || X.ℳ != Y.ℳ
        error("X and Y are not in the same tangent space")
    end
    return TangentVector(X.x, X.ẋ-Y.ẋ, X.ℳ)
end

function Base.:*(X::TangentVector{T, TM}, α::Tα) where {Tα<:Real,T,TM}
    return TangentVector(X.x, α.*X.ẋ, X.ℳ)
end
Base.:*(α::Tα, X::TangentVector{T, TM}) where {Tα<:Real,T,TM} = X*α

"""
    RegularSumbanifold <: EmbeddedManifold
    
Supertype for regular submanifolds.
"""
abstract type RegularSubmanifold <: EmbeddedManifold end

"""
    f(x::T, ℳ::TM) where {T<:AbstractArray, TM<:RegularSubmanifold}

Function that describes the embedding of 'ℳ' in the ambient space. 'f' is so
that ''ℳ = f^{-1}(\\{ 0 \\})''
"""
function f(x::T, ℳ::TM) where {T<:AbstractArray, TM<:RegularSubmanifold}
end

"""
    P(x::T, ℳ::TM) where {T<:AbstractArray, TM<:RegularSubmanifold}

The projection matrix ``\\mathbb{R}^N \\to T_x\\mathcal{M}`` given
by ``I-n(x)n(x)^T``, where ``n(x)=∇f(x)/|∇f(x)|``.
"""
function P(x::T, ℳ::TM) where {T<:AbstractArray, TM<:RegularSubmanifold}
end

 """   
    nCharts(ℳ::TM) where {TM<:Manifold}

Returns the amount of charts the manifold 'ℳ' has
"""
function nCharts(ℳ::TM) where {TM<:Manifold}
end

"""
    inChart(x::T , ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}

Yields a vector of Booleans where entry `i` is true if the point `x` in the
ambient space lies in chart ``U_i``
"""
function inChart end
# function inChart(x::T, ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}
# end
"""
    inChartRange(q::T , ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}

Yields a vector of Booleans where entry `i` is true if the point `q` in ℝᵈ lies in the
range of chart ``U_i``
"""
function inChartRange end

"""
    ϕ(x::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}

Defines the map ``U_i \\to \\mathbb{R}^d`` from global to local coordinates
"""
function ϕ(x::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}
end

"""
    ϕ⁻¹(q::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}

Defines the inverse map ``\\mathbb{R}^d \\to U_i`` of 'ϕ' from local to global
coordinates.
"""
function ϕ⁻¹(q::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}
end

"""
    Dϕ(x::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}

Defines the Jacobian matrix of `ϕ`
"""
function Dϕ(x::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}
    if length(x) == 1
        J = ForwardDiff.derivative((p) -> ϕ(p,i, ℳ), x)
    else
        J = ForwardDiff.jacobian((p) -> ϕ(p,i, ℳ), x)
    end
    return J
end

"""
    Dϕ⁻¹(q::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}

Defines the Jacobian matrix of `ϕ⁻¹`
"""
function Dϕ⁻¹(q::T, i::Int64, ℳ::TM) where {T<:AbstractArray, TM<:Manifold}
    if length(q) == 1
        J = ForwardDiff.derivative((p) -> ϕ⁻¹(p,i, ℳ), q)
    else
        J = ForwardDiff.jacobian((p) -> ϕ⁻¹(p,i, ℳ), q)
    end
    return J
end

"""
    g(q::T, i::Int64, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}

If `ℳ<:EmbeddedManifold` is given in local coordinates
``\\phi^{-1}:\\mathbb{R}^d \\to \\mathbb{R}^N`` in chart ``U_i``, we obtain a
Riemannian metric. `g(q,i, ℳ)` returns the matrix
``D\\phi^{-1}^T D\\phi^{-1}``,where ``D\\phi^{-1}``
denotes the Jacobian matrix for ``\\phi^{-1}`` in `q<:Union{AbstractArray, Real}`.
"""
function g(q::T, i::Int64, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    # q = ϕ(x,i,ℳ)
    J = Dϕ⁻¹(q,i,ℳ)
    return J'*J
end

"""
    g♯(q::T, i::Int64,ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}

Riemannian cometric, the inverse matrix of `g(q,i,ℳ)`.
"""
function g♯(q::T, i::Int64,ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    return inv(g(q, i,ℳ))
end

"""
    Γ(q::T, n::Int64, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}

Christoffel symbols ``Γ^i_{jk}`` for the Levi-Civita connection in chart `n`.

Given `q` in chart `n`, `Γ(q, n, ℳ)` returns a matrix of size ``d\\times d\\times d`` where the
element `[i,j,k]` corresponds to ``Γ^i_{jk}`` and `d=Dimension(ℳ)`.
"""
function Γ(q::T, n::Int64, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    d = length(q)
    @assert d==Dimension(ℳ) "q is not of the same dimension as ℳ"
    if d == 1
        ∂g = ForwardDiff.derivative(y -> g(y,n,ℳ), q)
        g⁻¹ = 1/g(q,n,ℳ)
        return .5*g⁻¹*∂g
    else
        ∂g = reshape(ForwardDiff.jacobian(y -> g(y,n,ℳ), q), d, d, d)
        g⁻¹ = g♯(q,n,ℳ)
        @einsum out[i,j,k] := .5*g⁻¹[i,l]*(∂g[k,l,i] + ∂g[l,j,k] - ∂g[j,k,l])
        return out
    end
end

"""
    Hamiltonian(x::Tx, p::Tp, n::Int64, ℳ::TM) where {Tx, Tp <: Union{AbstractArray, Real}, TM <: EmbeddedManifold}

Returns the Hamiltonian induced by the Riemannian metric for a tangent vector `p` to `ℳ` at `x`
"""
function Hamiltonian(x::Tx, X::TangentVector, ℳ::TM) where {Tx<: Union{AbstractArray, Real}, TM <: Manifold}
    n = rand(findall(==(1), inChart(x,ℳ)))
    q = ϕ(x, n, ℳ) ; p = Dϕ(x,n,ℳ)*X.ẋ
    return .5*p'*g♯(q, n, ℳ)*p
 end
