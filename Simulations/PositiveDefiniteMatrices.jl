include("../src/ManifoldDiffusions.jl")

using Main.ManifoldDiffusions
using LinearAlgebra

struct SP{T<:Real} <: EmbeddedManifold
    n::T

    function SP(n::T) where {T<:Integer}
        if n <= 0
            error("n must be a positive integer")
        end
        new{T}(n)
    end
end

struct S{T<:Real}  <: EmbeddedManifold
    n::T

    function S(n::T) where {T<:Integer}
        if n <= 0
            error("n must be a positive integer")
        end
        new{T}(n)
    end
end

Dimension(𝒮::SP{T}) where {T} = 𝒮.n*(𝒮.n+1)/2
Embedded_Dimension(𝒮::SP{T}) where {T} = 𝒮.n*𝒮.n

function isS(M, 𝒮::S{T}) where {T}
    if size(M) != (𝒮.n, 𝒮.n)
        return DomainError(size(M) , "M is not an n×n matrix")
    end
    if !isapprox(norm(M-transpose(M)),0.0)
        return DomainError(norm(M-transpose(M)), "Matrix is not symmetric")
    end
    return nothing
end

function isSP(M, 𝒮::SP{T}) where {T}
    isS(M, S(𝒮.n))
    if !all(eigvals(M) .> 0)
        return DomainError(eigvals(M), "Matrix is not positive definite")
    end
    return nothing
end




#Frechet inner product
function gF(P , S₁, S₂ , 𝒮::SP{N}) where {N}
    isS(S₁, S(𝒮.n)); isS(S₂, S(𝒮.n)) ; isSP(P, SP(𝒮.n))
    return tr(transpose(S₁)*S₂)
end

function gLE(S , S₁, S₂ , 𝒮::SP{N}) where {N}
    isS(S₁, S(𝒮.n)); isS(S₂, S(𝒮.n)) ; isSP(S , SP(𝒮.n))
    # UNFINISHED
end
