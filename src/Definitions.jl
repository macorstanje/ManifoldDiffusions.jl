"""
    Manifold

Abstract (super-)type under which all speficic manifolds fall
"""
abstract type Manifold end

# abstract type SDEForm end
# struct Ito <: SDEForm end
# struct Stratonovich <: SDEForm end

const ℝ{N} = SVector{N, Float64}
const IndexedTime = Tuple{Int64,Float64}
outer(x) = x*x'
outer(x,y) = x*y'
extractcomp(v, i) = map(x->x[i], v)

"""
    EmbeddedManifold <: Manifold

EmbeddedManifold creates a manifold ``\\mathcal{M} = f^{-1}(\\{0\\})`` of
dimension ``d=N-n`` where ``f`` should be a smooth function
``\\mathbb{R}^N \\to \\mathbb{R}^n``. An EmbeddedManifold `ℳ` can be equipped
equipped with functions `f( , ℳ)`, `P( , ℳ)` and `F( , ℳ)`.
Here `f` is such that `f(q, ℳ)=0` when ``q\\in\\mathcal{M}``.

`P(q, ℳ)` is the projection matrix ``\\mathbb{R}^N \\to T_q\\mathcal{M}`` given
by ``I-n(q)n(q)^T``, where ``n(q)=∇f(q)/|∇f(q)|``.

`F(q, ℳ)` is the transformation from local coordinates `q` to global coordinates
in ``\\mathbb{R}^N``.
"""
abstract type EmbeddedManifold <: Manifold end

"""
    TangentVector{T, TM}

Elements of ``T_x\\mathcal{M}`` and equipped with vector space operations.
"""
struct TangentVector{T,TM}
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
    Ellipse{T<:Real} <: EmbeddedManifold

Settings for an ellipse as subset of ``\\mathbb{R}^2``. Elements satisfy
``(x/a)^2 + (y/b)^2 = 1``.

For an object `𝔼 = Ellipse(a, b)`, one has

- `` f(q, \\mathcal{𝔼}) = \\left(\\frac{q_1}{a}\\right)^2 + \\left(\\frac{q_2}{b}\\right)^2 - 1 ``
- `` F(q, 𝔼) = \\begin{pmatrix} a\\cos q & b \\sin q\\end{pmatrix}``

# Example: Generate a unit circle
```julia-repl
julia> 𝔼 = Ellipse(1.0, 1.0)
```
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
    Sphere{T<:Real} <: EmbeddedManifold

Settings for the sphere ``\\mathbb{S}^2``. Call `Sphere(R)` to generate a sphere
with radius `R<:Real`. Elements satisfy ``x^2+y^2+z^2=R^2``. The local coordinates
are modelled via a stereograpgical projection.

For a Sphere `𝕊 = Sphere(R)`, one has

- ``f(q, 𝕊) = q_1^2+q_2^2-R^2``
- ``F(q, 𝕊) = \\begin{pmatrix} \\frac{2q_1}{q_1^2+q_2^2+1} & \\frac{2q_2}{q_1^2+q_2^2+1} & \\frac{q_1^2+q_2^2-1}{q_1^2+q_2^2+1} \\end{pmatrix}``

# Example: Generate a unit sphere
```julia-repl
julia> 𝕊 = Sphere(1.0)
```
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
    Torus{T<:Real} <: EmbeddedManifold

Settings for the torus ``\\mathbb{T}^2`` with inner radius ``r`` and outer radius
``R``. Call `Torus(R,r)` to generate a torus with inner radius `r<:Real` and outer radius `R<:Real`.
Elements satisfy ``(x^2+y^2+z^2+R^2-r^2)^2=4R^2(x^2+y^2)``.

For a Torus `𝕋 = Torus(R, r)`, one has

- ``f(q, 𝕋) = (q_1^2+q_2^2+q_3^2+R^2-r^2)^2-4R^2(q_1^2+q_2^2)``
- ``F(q, 𝕋) = \\begin{pmatrix} (R+r\\cos q_1)\\cos q_2 & (R+r\\cos q_1)\\sin q_2 & r\\sin q_1 \\end{pmatrix} ``

# Example: Generate a torus with ``R=3`` and ``r=1``
```julia-repl
julia> 𝕋 = Torus(3.0, 1.0)
```
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
    Paraboloid{T<:Real} <: EmbeddedManifold

Settings for the Paraboloid. Call `Paraboloid(a,b)` to generate a paraboloid
with parameters `a<:Real` and outer radius `b<:Real`.
Elements satisfy ``(x/a)^2+(y/b)^2 = z``.

For a paraboloid `ℙ = Paraboloid(a, b)`, one has

- ``f(q, ℙ) = \\left(\\frac{q_1}{a}\\right)^2 + \\left(\\frac{q_2}{b}\\right)^2-q_3 ``
- ``F(q, ℙ) = \\begin{pmatrix} q_1 & q_2 & \\left(\\frac{q_1}{a}\\right)^2 + \\left(\\frac{q_2}{b}\\right)^2 \\end{pmatrix} ``

# Example: Generate a torus with ``a=0`` and ``b=1``
```julia-repl
julia> ℙ = Parabolod(3.0, 1.0)
```
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
    g(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}

If `ℳ<:EmbeddedManifold` is given in local coordinates ``F:\\mathbb{R}^d \\to \\mathbb{R}^N``
, we obtain a Riemannian metric. `g(q, ℳ)` returns the matrix ``\\mathrm{d}F^T\\mathrm{d}F``,
where ``\\mathrm{d}F`` denotes the Jacobian matrix for ``F`` in `q<:Union{AbstractArray, Real}`.
"""
function g(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    if length(q) == 1
        J = ForwardDiff.derivative((p) -> F(p, ℳ), q)
    else
        J = ForwardDiff.jacobian((p) -> F(p, ℳ), q)
    end
    return J'*J
end

# Returns the cometric
function gˣ(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}
    return inv(g(q, ℳ))
end


"""
    Γ(q::T, ℳ::TM) where {T<:Union{AbstractArray, Real}, TM<:EmbeddedManifold}

If `ℳ<:EmbeddedManifold` is given in local coordinates ``F:\\mathbb{R}^d \\to \\mathbb{R}^N``
, we obtain Christoffel symbols ``Γ^i_{jk}`` for the Levi-Civita connection.

In local coordinates `q`, `Γ(q, ℳ)` returns a matrix of size ``d\\times d\\times d`` where the
element `[i,j,k]` corresponds to ``Γ^i_{jk}``.
"""
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

"""
    Hamiltonian(x::Tx, p::Tp, ℳ::TM) where {Tx, Tp <: Union{AbstractArray, Real}, TM <: EmbeddedManifold}

Returns the Hamiltonian induced by the Riemannian metric for a tangent vector `p` to `ℳ` at `x`
"""
function Hamiltonian(x::Tx, p::Tp, ℳ::TM) where {Tx, Tp <: Union{AbstractArray, Real}, TM <: EmbeddedManifold}
    .5*p'*gˣ(x, ℳ)*p
 end
