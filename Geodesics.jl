
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
    Geodesic flow and the exponential map on the Frame bundle
"""

function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u₀.x)
    U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
    V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d)) , xx)
    vv = map(p->TangentFrame(u₀, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

function ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    tt = collect(0:0.01:1)
    uu, vv = Geodesic(u₀, v₀, tt, Fℳ)
    return uu[end]
end

# UNCOMMENT TO SIMULATE GEODESICS

# include("Definitions.jl")
# tt = collect(0:0.001:1)
# x₀ = [0.,0.] # Corresponds with [0,0,-1]
# v₀ = ForwardDiff.jacobian(p->F(p, 𝕊), [0.,0.])*[1, -1]
# q, p = Integrate(Hamiltonian, tt, [0.,0.], [2.,-2.], 𝕊)
#
# Plots.plot([extractcomp(q,1), extractcomp(q,2)])
# x = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 1)
# y = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 2)
# z = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 3)
# Plots.plot([x,y,z])
# SpherePlot(x,y,z,𝕊)
# zero([0.,0.])
# [x[end-1], y[end-1], z[end-1]]


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
        @einsum dV[i] := - _Γ[i,j,k]*V[j]*_γ̇[k]
        V += dV
        push!(VV, V)
    end
    return VV
end
