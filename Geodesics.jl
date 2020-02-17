
"""
    generate geodesics γ: I -> ℳ satisfying γ₀ = x ∈ ℳ and γ̇₀ = v ∈ 𝑇ₓℳ

    View geodesics as Hamiltonian flows and solve the equations
    dx = ∇_p H(x, p) dt
    dp = - ∇_x H(x, p) dt
"""

# integrate over a discretized time interval tt, discards impulse vectors
function Integrate(H, tt, x₀::Tx, p₀::Tp, ℳ::TM) where {Tx, Tp <:AbstractArray, TM <: EmbeddedManifold}
    N = length(tt)
    x, p = x₀, p₀
    xx, pp = [x], [p]
    for i in 1:(N-1)
        dt = tt[i+1]-tt[i]
        dp = ForwardDiff.gradient(x -> -H(x, p, ℳ), x).*dt
        p += .5*dp
        dx = ForwardDiff.gradient(p -> H(x, p, ℳ), p).*dt
        x += dx
        dp = ForwardDiff.gradient(x -> -H(x, p, ℳ), x).*dt
        p += .5*dp
        push!(xx, x)
        push!(pp, p)
    end
    return xx,pp
end

function Geodesic(x₀::Tx, v₀::Tv, tt, ℳ::TM) where {Tx, Tv <: AbstractArray, TM<:EmbeddedManifold}
    xx, vv = Integrate(Hamiltonian, tt, x₀, v₀, ℳ)
    return xx, vv
end

function ExponentialMap(x₀::Tx, v₀::Tv, ℳ::TM) where {Tx, Tv <: AbstractArray, TM<:EmbeddedManifold}
    tt = collect(0:0.01:1)
    xx, vv = Geodesic(x₀, v₀, tt, ℳ)
    return xx[end]
end


"""
    Parallel transport along a curve γ:
        v̇^i(t) + Γ^i_{jk}(γ(t))v^j γ̇^k = 0 for all i
"""

# returns Vt, the parallel transport of initial vector V₀, where γ and γ̇ are known on tt
function ParallelTransport(γ, γ̇, V₀, tt, ℳ)
    N = length(tt)
    V = V₀
    VV = [V₀]
    for n in 1:(N-1)
        dt = tt[n+1] - tt[n]
        _Γ = Γ(γ[n], ℳ)
        _γ̇ = γ̇[n]
        if length(V₀)>1
            @einsum dV[i] := - _Γ[i,j,k]*V[j]*_γ̇[k]*dt
        else
            dV = -_Γ*V*_γ̇*dt
        end
        V += dV
        push!(VV, V)
    end
    return VV
end
