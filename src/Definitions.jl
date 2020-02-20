
abstract type Manifold end
abstract type SDEForm end

struct Ito <: SDEForm end
struct Stratonovich <: SDEForm end

const ℝ{N} = SVector{N, Float64}
const IndexedTime = Tuple{Int64,Float64}
outer(x) = x*x'
outer(x,y) = x*y'
extractcomp(v, i) = map(x->x[i], v)

"""
    EmbeddedManifold creates a manifold ``M = f^{-1}({0})`` of dimension d=N-n
    where ``f`` should be a smooth function ``ℝ^N → ℝ^n``
"""

abstract type EmbeddedManifold <: Manifold end

"""
    Elements of 𝑇ₓℳ and some operations
"""

# a vector v ∈ 𝑇ₓℳ
struct TangentVector{T,TM}
    x::T
    v::T
    ℳ::TM
    function TangentVector(v::T, x::T, ℳ::TM) where {T<:AbstractArray, TM<:EmbeddedManifold}
        new{T,TM}(x, v,ℳ)
    end
end

# Vector space operations on 𝑇ₓℳ
function Base.:+(X::TangentVector{T,TM}, Y::TangentVector{T,TM}) where {T,TM}
    if X.x != Y.x || X.ℳ != Y.ℳ
        error("X and Y are not in the same tangent space")
    end
    return TangentVector(X.x, X.v+Y.v, X.ℳ)
end

function Base.:-(X::TangentVector{T,TM}, Y::TangentVector{T,TM}) where {T,TM}
    if X.x != Y.x || X.ℳ != Y.ℳ
        error("X and Y are not in the same tangent space")
    end
    return TangentVector(X.x, X.v-Y.v, X.ℳ)
end

function Base.:*(X::TangentVector{T, TM}, α::Tα) where {Tα<:Real,T,TM}
    return TangentVector(X.x, α.*X.v, X.ℳ)
end
Base.:*(α::Tα, X::TangentVector{T, TM}) where {Tα<:Real,T,TM} = X*α

"""
Settings for an ellipse 𝔼 as subset of ℝ²
"""

struct Ellipse{T<:Real} <: EmbeddedManifold
    a::T
    b::T

    function Ellipse(a::T, b::T) where {T<:Real}
        if a<=0 || b<=0
            error("a and b must be positive")
        end
        new{T}(a,b)
    end
end

function f(q::T, 𝔼::Ellipse) where {T<:AbstractArray}
    (q[1]/𝔼.a)^2 + (q[2]/𝔼.b)^2 - 1.0
end

function P(q::T, 𝔼::Ellipse) where {T<:AbstractArray}
    x, y, a, b = q[1], q[2], 𝔼.a, 𝔼.b
    ∇f = 2.0.*[x/(a^2) , y/(b^2)]
    n = ∇f./norm(∇f)
    return Matrix{eltype(n)}(I,2,2) - n*n'
end

function F(θ::T, 𝔼::Ellipse) where {T<:Real}
    [𝔼.a*cos.(θ) , 𝔼.b*sin.(θ)]
end

"""
    We introduce some manifolds embedded in ℝ³, given by f⁻¹({0}) and
    parameterized by ℝ² ⊇ (u,v) ↦ F(u,v) ⊆ ℝ³
"""

"""
    Settings for the sphere 𝕊²
"""

struct Sphere{T<:Real} <: EmbeddedManifold
    R::T

    function Sphere(R::T) where {T<:Real}
        if R<=0
            error("R must be positive")
        end
        new{T}(R)
    end
end

function f(q::T, 𝕊::Sphere) where {T<:AbstractArray}
    q[1]^2+q[2]^2+q[3]^2-𝕊.R^2
end

# Projection matrix
function P(q::T, 𝕊::Sphere) where {T<:AbstractArray}
    R, x, y, z = 𝕊.R, q[1], q[2], q[3]
    return [4*R^2-x^2 -x*y -x*z ; -x*y 4*R^2-y^2 -y*z ; -x*z -y*z 4*R^2-z^2]./(4*R^2)
end

# Stereographical projection
function F(q::T, 𝕊::Sphere) where {T<:AbstractArray}
    R, u, v = 𝕊.R, q[1], q[2]
    return [ 2*u/(u^2+v^2+1) , 2*v/(u^2+v^2+1) , (u^2+v^2-1)/(u^2+v^2+1)  ]
end

"""
    Settings for the Torus 𝕋².
"""

struct Torus{T<:Real} <: EmbeddedManifold
    R::T
    r::T

    function Torus(R::T, r::T) where {T<:Real}
        if R<r
            error("R must be larger than or equal to r")
        end
        new{T}(R,r)
    end
end

function f(x::T, 𝕋::Torus) where {T<:AbstractArray}
    R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
    (x^2 + y^2 + z^2 + R^2 - r^2)^2 - 4.0*R^2*(x^2 + y^2)
end

# Projection matrix
function P(x::T, 𝕋::Torus) where {T<:AbstractArray}
    R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
    ∇f = [  4*x*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*x,
            4*y*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*y,
            4*z*(x^2+y^2+z^2+R^2-r^2)]# ForwardDiff.gradient((y)->f(y, 𝕋), x)
    n = ∇f./norm(∇f)
    return Matrix{eltype(n)}(I,3,3) .- n*n'
end

function F(x::T, 𝕋::Torus) where {T<:AbstractArray}
    R, r, u, v = 𝕋.R, 𝕋.r, x[1], x[2]
    return [(R+r*cos(u))*cos(v) , (R+r*cos(u))*sin(v) , r*sin(u)]
end


"""
    Settings for the Paraboloid ℙ²
"""

struct Paraboloid{T<:Real} <: EmbeddedManifold
    a::T
    b::T

    function Paraboloid(a::T, b::T) where {T<:Real}
        if a == 0 || b == 0
            error("parameters cannot be 0")
        end
        new{T}(a, b)
    end
end

function f(x::T, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, x, y, z = ℙ.a, ℙ.b, x[1], x[2], x[3]
    return (x/a)^2 + (y/b)^2 - z
end

function P(x::T, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, x, y, z = ℙ.a, ℙ.b, x[1], x[2], x[3]
    ∇f = [2*x/a, 2*y/b , -1]
    n = ∇f./norm(∇f)
    return Matrix{eltype(n)}(I,3,3) .- n*n'
end

function F(q::T, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, u, v = ℙ.a, ℙ.b, q[1], q[2]
    return [u, v, (u/a)^2+(v/b)^2]
end
"""
    If a manifold is given as result of a function F:ℝᵈ → ℝᴺ, we obtain a
    Riemannian metric and Christoffel symbols for the Levi-Civita connection
"""

# Riemannian metric in terms of a parameterization F
function g(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    if length(q) == 1
        J = ForwardDiff.derivative((p) -> F(p, ℳ), q)
    else
        J = ForwardDiff.jacobian((p) -> F(p, ℳ), q)
    end
    return J'*J
    # [4/(q[1]^2+q[2]^2+1)^2 0 ; 0 4/(q[1]^2+q[2]^2+1)^2]
end

# Returns the cometric
function gˣ(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    return inv(g(q, ℳ))
    # [(q[1]^2+q[2]^2+1)^2/4 0 ; 0 (q[1]^2+q[2]^2+1)^2/4]
end

# Christoffel symbols Γ^i_{jk}
function Γ(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    d = length(q)
    if d == 1
        ∂g = ForwardDiff.derivative(x -> g(x,ℳ), q)
        g⁻¹ = 1/g(q, ℳ)
        return .5*g⁻¹*∂g
    else
        ∂g = reshape(ForwardDiff.jacobian(x -> g(x,ℳ), q), d, d, d)
        g⁻¹ = gˣ(q, ℳ)
        @einsum out[i,j,k] := .5*g⁻¹[i,l]*(∂g[k,l,i] + ∂g[l,j,k] - ∂g[j,k,l])
        return out
    end
end


# Hamiltonian
function Hamiltonian(x::Tx, p::Tp, ℳ::TM) where {Tx, Tp <: Union{AbstractArray, Real}, TM <: EmbeddedManifold}
    .5*p'*gˣ(x, ℳ)*p
 end
