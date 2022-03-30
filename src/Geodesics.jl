
# """
#     generate geodesics γ: I -> ℳ satisfying γ₀ = x ∈ ℳ and γ̇₀ = v ∈ 𝑇ₓℳ

#     View geodesics as Hamiltonian flows and solve the equations
#     dx = ∇_p H(x, p) dt
#     dp = - ∇_x H(x, p) dt
# """

# # integrate Hamiltonian equations in chart n over a discretized time interval tt,
# # discards impulse vectors
# function Integrate(H, tt, x₀::Tx, p₀::Tp, ℳ::TM) where {Tx, Tp <:AbstractArray, TM <: Manifold}
#     N = length(tt)
#     x, p = x₀, p₀
#     xx, pp = [x], [p]
#     for i in 1:(N-1)
#         dt = tt[i+1]-tt[i]
#         n = rand(findall(==(1), inChart(x, ℳ)))
#         q = ϕ(x,n,ℳ)
#         dp = ForwardDiff.gradient(q -> -H(q, p, n, ℳ), q).*dt
#         p += .5*dp
#         dq = ForwardDiff.gradient(p -> H(q, p, n, ℳ), p).*dt
#         q += dq
#         dp = ForwardDiff.gradient(q -> -H(q, p, n, ℳ), q).*dt
#         p += .5*dp
#         x = ϕ⁻¹(q, n, ℳ)
#         push!(xx, x)
#         push!(pp, p)
#     end
#     return xx,pp
# end

"""
    Geodesic(x₀::Tx, v₀::TangentVector, tt, ℳ::TM) where {Tx<: AbstractArray, TM<:EmbeddedManifold}

Returns the values of the geodesic on `ℳ` starting at `x₀` with initial velicity
v₀ on a discretized time interval `tt`. All input is in local coordinates within chart `n`.
"""
function Geodesic(x₀::Tx, X::TangentVector, tt, ℳ::TM) where {Tx <: AbstractArray, TM<:Manifold}
    # xx, vv = Integrate(Hamiltonian, tt, x₀, v₀, ℳ)
    x = x₀ 
    n = rand(findall(==(1), inChart(x, ℳ)))
    v = Dϕ(x,n,ℳ)*X.ẋ
    xx, vv = [x], [v]
    for i in 1:length(tt)-1
        dt = tt[i+1] - tt[i]
        n = rand(findall(==(1), inChart(x, ℳ)))
        q = ϕ(x,n,ℳ)
        dq = v*dt
        q += 0.5*dq
        _Γ = Γ(q,n,ℳ)
        @einsum dv[k] := v[l]*v[j]*_Γ[k,l,j]
        v += dv
        q += 0.5*dq
        x = ϕ⁻¹(q,n,ℳ)
        push!(xx,x) ; push!(vv,v)
    end
    return xx, vv
end

"""
    ExponentialMap(x₀::Tx, v₀::Tv, ℳ::TM) where {Tx, Tv <: AbstractArray, TM<:EmbeddedManifold}

Returns as new element of `ℳ` that results from ``Exp_{x_0}v_0``, where the
point `x₀` on `ℳ` and initial velocity `v₀` are given in local coordinates within chart `n`.
"""
function ExponentialMap(x₀::Tx, X::TangentVector, ℳ::TM) where {Tx<:AbstractArray, TM<:Manifold}
    tt = collect(0:0.01:1)
    xx, vv = Geodesic(x₀, X, tt, ℳ)
    return xx[end]
end


"""
    ParallelTransport(γ, γ̇, n, V₀, tt, ℳ)

returns the parallel transport of an initial  vector V₀, tangent to ℳ at ``\\gamma_0``,
along a curve `γ` within chart `n`. It is assumed γ and γ̇ are known on a discretized time interval `tt`
"""
function ParallelTransport(γ, γ̇, n, V₀, tt, ℳ::TM) where {TM<:Manifold}
    N = length(tt)
    V = V₀
    VV = [V₀]
    for l in 1:(N-1)
        dt = tt[l+1] - tt[l]
        _Γ = Γ(γ[l],n,  ℳ)
        _γ̇ = γ̇[l]
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
