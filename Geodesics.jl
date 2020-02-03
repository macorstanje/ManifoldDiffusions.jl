include("Definitions.jl")


"""
    generate geodesics γ: I -> ℳ satisfying γ₀ = x ∈ ℳ and γ̇₀ = v ∈ 𝑇ₓℳ
"""

using Bridge
include("SpherePlots.jl") ; plotly()

x₀ = F([0., 0.], 𝕊)
v₀ = [1,-1,0]

# Check result with the exponential map
cos(norm(v₀)) .* x₀ .+ (sin(norm(v₀))/norm(v₀)).*v₀


"""
    View geodesics as Hamiltonian flows and solve the equations
    dx = ∇_p H(x, p) dt
    dp = - ∇_x H(x, p) dt
"""

# integrate over a discretized time interval tt, discards impulse vectors
function Integrate(H, tt, x₀, p₀, ℳ)
    N = length(tt)
    x, p = x₀, p₀
    xx, pp = [x], [p]
    dx, dp = zero(x), zero(p)
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

tt = collect(0:0.001:1)
x₀ = [0.,0.] # Corresponds with [0,0,-1]
v₀ = ForwardDiff.jacobian(p->F(p, 𝕊), [0.,0.])*[1, -1]
q, p = Integrate(Hamiltonian, tt, [0.,0.], [2.,-2.], 𝕊)

Plots.plot([extractcomp(q,1), extractcomp(q,2)])
x = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 1)
y = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 2)
z = extractcomp([F(q[i], 𝕊) for i in 1:length(q)], 3)
Plots.plot([x,y,z])
SpherePlot(x,y,z,𝕊)
zero([0.,0.])
[x[end-1], y[end-1], z[end-1]]


"""
    Parallel transport along a curve γ:
        v̇^i(t) + Γ^i_{jk}(γ(t))v^j γ̇^k = 0 for all i
"""

# returns Vt, the parallel transport of initial vector V₀, where γ and γ̇ are known on tt
function ParallelTransport(γ, γ̇, V₀, V₀, tt, ℳ)
    N = length(tt)
    V = V₀
    VV = [V₀]
    for n in 1:(N-1)
        dt = tt[n+1] - tt[n]
        _Γ = Γ(γ[n], ℳ)
        _γ̇ = γ[n]
        @einsum out[i] := V[i] -_Γ[i,j,k]*V[j]*γ̇[k]
        push!(VV, V)
    end
    return VV
end
