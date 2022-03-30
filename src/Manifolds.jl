"""
    Ellipse{T<:Real} <: EmbeddedManifold

Settings for an ellipse as subset of ``\\mathbb{R}^2``. Elements satisfy
``(x/a)^2 + (y/b)^2 = 1``.

For an object `𝔼 = Ellipse(a, b)`, one has

- `` f(x, \\mathcal{𝔼}) = \\left(\\frac{x_1}{a}\\right)^2 + \\left(\\frac{x_2}{b}\\right)^2 - 1 ``
- `` ϕ⁻¹(q, 1, 𝔼) = \\begin{pmatrix} a\\cos q & b \\sin q\\end{pmatrix}``

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

Dimension(𝔼::Ellipse{T}) where {T} = 1
AmbientDimension(𝔼::Ellipse{T}) where {T} = 2

function f(x::T, 𝔼::Ellipse) where {T<:AbstractArray}
    (x[1]/𝔼.a)^2 + (x[2]/𝔼.b)^2 - 1.0
end

function P(x::T, 𝔼::Ellipse) where {T<:AbstractArray}
    ∇f = 2.0.*[x[1]/(𝔼.a^2) , x[2]/(𝔼.b^2)]
    n = ∇f./norm(∇f)
    return Matrix{eltype(n)}(I,2,2) - n*n'
end

nCharts(𝔼::Ellipse{T}) where {T} = 1
inChart(x::T, 𝔼::Ellipse{S}) where {T<:AbstractArray} where {S} = true

function ϕ(x::T, n::Int64, 𝔼::TM) where {T<:AbstractArray, TM<:Ellipse}
    if n>nCharts(𝔼)
        error("This chart does not exist")
    end
    return abs(x[2]/𝔼.b)==1 ? sign(x[2])*pi/2 : arcsin(x[2]/𝔼.b)
end


function ϕ⁻¹(q::T, n::Int64, ::Ellipse) where {T<:Real}
    if n>nCharts(𝔼)
        error("This chart does not exist")
    end
    [𝔼.a*cos.(q) , 𝔼.b*sin.(q)]
end

"""
    Sphere{T<:Real} <: EmbeddedManifold

Settings for the n-sphere ``\\mathbb{S}^n``. Call `Sphere(n)` to generate a sphere
with radius 1. Elements satisfy ``|x|=1``. The local coordinates
are modelled via stereograpgical projections.

For a 2-Sphere `𝕊 = Sphere(2, R)`, one has

- ``f(q, 𝕊) = x_1^2+x_2^2+x_3^2-R^2``
- ``ϕ⁻¹(y,1, 𝕊) = \\begin{pmatrix} \\frac{2Ry_1}{y_1^2+y_2^2+1} & \\frac{2Ry_2}{y_1^2+y_2^2+1} & R\\frac{y_1^2+y_2^2-1}{y_1^2+y_2^2+1} \\end{pmatrix}``
- ``ϕ⁻¹(y,2, 𝕊) = \\begin{pmatrix} \\frac{2Ry_1}{y_1^2+y_2^2+1} & \\frac{2Ry_2}{y_1^2+y_2^2+1} & -R\\frac{y_1^2+y_2^2-1}{y_1^2+y_2^2+1} \\end{pmatrix}``


# Example: Generate a unit d-sphere
```julia-repl
julia> 𝕊 = Sphere(d, 1.0)
```
"""
struct Sphere{T<:Real} <: EmbeddedManifold
    n::Int64
    R::T

    function Sphere(n::Int64, R::T) where {T<:Real}
        return R>0 ? new{T}(n,R) : error("Radius must be positive")
    end
end

Dimension(𝕊::Sphere) = 𝕊.n
AmbientDimension(𝕊::Sphere) = 𝕊.n+1

function f(x::T, 𝕊::Sphere) where {T<:AbstractArray}
    dot(x,x)-𝕊.R^2
end

# Projection matrix
function P(x::T, 𝕊::Sphere) where {T<:AbstractArray}
    N = AmbientDimension(𝕊)
    return Matrix{eltype(q)}(I, N, N) - x*transpose(x)./𝕊.R^2
end

nCharts(𝕊::Sphere) = 2
function inChart(x::T, 𝕊::Sphere) where {T<:Union{Real,AbstractArray}}
    # North pole : 𝕊∖{(0,..,0,R)}, South pole : : 𝕊∖{(0,..,0,-R)}
    d = Dimension(𝕊)
    return [!isapprox(x, push!(zeros(d),𝕊.R)) , !isapprox(x, push!(zeros(d),-𝕊.R))]
end

function inChartRange(q::T, 𝕊::Sphere) where {T<:Union{Real,AbstractArray}}
    return [true for i in 1:nCharts(𝕊)]
end

# Stereographical projection
function ϕ(x::T, i::Int64, 𝕊::TM) where {T<:AbstractArray,TM<:Sphere}
    @assert i <= nCharts(𝕊) "Chart number can only be 1 or 2"
    sgn = sign(i-3/2) # -1 if i=1 and +1 if i=2
    @assert inChart(x, 𝕊)[i] "$q is not in chart $i"
    out = x[1:end-1]./(𝕊.R+sgn*x[end])
    return Dimension(𝕊) == 1 ? out[1] : out
end

function ϕ⁻¹(q::T, i::Int64, 𝕊::TM) where {T<:AbstractArray, TM<:Sphere}
    @assert i <= nCharts(𝕊) "Chart number can only be 1 or 2"
    s2 = dot(q, q)
    return push!(q[1:end].*2.0.*𝕊.R ./(s2+1) , -sign(i-3/2)*𝕊.R*(s2-1)/(s2+1))
end

# For the case where 𝕊 is a circle
function ϕ⁻¹(q::T, i::Int64, 𝕊::TM) where {T<:Real, TM<:Sphere}
    @assert Dimension(𝕊) == 1 "Dimension(𝕊) must be the same as the size of q"
    @assert i <= nCharts(𝕊) "Chart number can only be 1 or 2"
    return [2*q*𝕊.R/(q^2+1) , -sign(i-3/2)*𝕊.R*(q^2-1)/(q^2+1)]
end


"""
    Torus{T<:Real} <: EmbeddedManifold


Torus ``\\mathbb{T}^n``, considered as a cartesian product of spheres is 
constructed from ``d`` 1-`Sphere`s. Order spheres by radius. It is embedded in 
``\\mathbb{R}^{2n}`` and equipped with the charts inherited from the spheres.

Example: The 2-Torus with inner radius r and outer radius R is called via

```julia-repl
julia> 𝕋 = Torus([Sphere(1,R), Sphere(1,r)])
```
"""
struct Torus <: EmbeddedManifold
    Spheres

    function Torus(Spheres)
        eltype(Spheres)<:Sphere ? new(Spheres) : error("Second argument must be an array of spheres")
    end
end

# struct Torus{T<:Real} <: EmbeddedManifold
#     R::T
#     r::T
#
#     function Torus(R::T, r::T) where {T<:Real}
#         if R<r
#             error("R must be larger than or equal to r")
#         end
#         new{T}(R,r)
#     end
# end

Dimension(𝕋::Torus) = length(𝕋.Spheres)
AmbientDimension(𝕋::Torus) = 2*Dimension(𝕋)

# function f(x::T, 𝕋::Torus) where {T<:AbstractArray}
#     R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
#     (x^2 + y^2 + z^2 + R^2 - r^2)^2 - 4.0*R^2*(x^2 + y^2)
# end

# Projection matrix
# function P(x::T, 𝕋::Torus) where {T<:AbstractArray}
#     R, r, x, y, z = 𝕋.R, 𝕋.r, x[1], x[2], x[3]
#     ∇f = [  4*x*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*x,
#             4*y*(x^2+y^2+z^2+R^2-r^2) - 8*R^2*y,
#             4*z*(x^2+y^2+z^2+R^2-r^2)]# ForwardDiff.gradient((y)->f(y, 𝕋), x)
#     n = ∇f./norm(∇f)
#     return Matrix{eltype(n)}(I,3,3) .- n*n'
# end

nCharts(𝕋::Torus) = 2*Dimension(𝕋)
function inChart(x::T,𝕋::Torus) where {T<:AbstractArray}
    S = 𝕋.Spheres
    out = Array{Bool,1}()
    for n in 1:nCharts(𝕋)
        # Map chart n for 𝕋 to charts (i,j, ...) for spheres  bijectively 
        indices = digits(n-1, base = 2, pad = Dimension(𝕋))+Int.(ones(Dimension(𝕋)))
        push!(out, all([inChart(x[2l-1:2l], S[l])[indices[l]] for l in 1:length(S)]))
    end
    return out
end

function inChartRange(q::T, 𝕋::Torus) where {T<:AbstractArray}
    S = 𝕋.Spheres
    out = Array{Bool,1}()
    for n in 1:nCharts(𝕋)
        # Map chart n for 𝕋 to charts (i,j, ...) for spheres  bijectively 
        indices = digits(n-1, base = 2, pad = Dimension(𝕋))+Int.(ones(Dimension(𝕋)))
        push!(out, all([inChartRange(q[l], S[l])[indices[l]] for l in 1:length(S)]))
    end
    return out
end

function ϕ(x::T, n::Int64, 𝕋::Torus) where {T<:AbstractArray}
    S = 𝕋.Spheres
    # Map chart n for 𝕋 to charts (i,j, ...) for spheres  bijectively 
    indices = digits(n-1, base = 2, pad = Dimension(𝕋))+Int.(ones(Dimension(𝕋)))
    @assert inChart(x, 𝕋)[n] "x=$x is not in chart $n = $indices"
    return [ϕ(x[2*l-1:2*l], indices[l], S[l]) for l in 1:Dimension(𝕋)]
end

function ϕ⁻¹(x::T, n::Int64, 𝕋::Torus) where {T<:AbstractArray}
    S = 𝕋.Spheres
    # Map chart n for 𝕋 to charts (i,j, ...) for spheres  bijectively 
    indices = digits(n-1, base = 2, pad = Dimension(𝕋))+Int.(ones(Dimension(𝕋)))

    out = ϕ⁻¹(x[1], indices[1], S[1])
    for l in 2:length(x)
        out = vcat(out, ϕ⁻¹(x[l], indices[l], S[l]))
    end
    return out

    # R, r, u, v = 𝕋.R, 𝕋.r, x[1], x[2]
    # return [(R+r*cos(u))*cos(v) , (R+r*cos(u))*sin(v) , r*sin(u)]
end

""" 
    Dim4toDim3(x::T, 𝕋::Torus) where {T<:AbstractArray}

Function specifically designed for 2-torus. Here, we model it embedded in ``ℝ^4``, but
for visualization purposes it can be embedded in ``ℝ^3`` by this map ``ℝ^4 \\to ℝ^3``
"""
function Dim4toDim3(x::T, 𝕋::Torus) where {T<:AbstractArray}
    @assert length(x)==4 "This map only applies to arrays of length 4"
    x,y,z,w = x[1],x[2],x[3],x[4]
    R = 𝕋.Spheres[1].R
    return [x+w*x/R, y+w*y/R, z]
end

"""
    Paraboloid{T<:Real} <: EmbeddedManifold

Settings for the Paraboloid embedded in ``ℝ^3```. Call `Paraboloid(a,b)` to generate 
a paraboloidwith parameters `a<:Real` and outer radius `b<:Real`.
Elements satisfy ``(x/a)^2+(y/b)^2 = z``.

For a paraboloid `ℙ = Paraboloid(a, b)`, one has

- ``f(q, ℙ) = \\left(\\frac{q_1}{a}\\right)^2 + \\left(\\frac{q_2}{b}\\right)^2-q_3 ``
- ``ϕ⁻¹(q,1, ℙ) = \\begin{pmatrix} q_1 & q_2 & \\left(\\frac{q_1}{a}\\right)^2 + \\left(\\frac{q_2}{b}\\right)^2 \\end{pmatrix} ``

# Example: Generate a torus with ``a=0`` and ``b=1``
```julia-repl
julia> ℙ = Parabolod(3.0, 1.0)
```

NOTE: Not all functions for the paraboloid have been implemented as of yet. Still to do: 
- `nCharts`
- `inChart`
- `inChartRange`
- `ϕ`
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

Dimension(ℙ::Paraboloid) = 2
AmbientDimension(ℙ::Paraboloid) = 3

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

function ϕ⁻¹(q::T, n, ℙ::Paraboloid) where {T<:AbstractArray}
    a, b, u, v = ℙ.a, ℙ.b, q[1], q[2]
    return [u, v, (u/a)^2+(v/b)^2]
end

