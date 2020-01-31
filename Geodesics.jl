include("Definitions.jl")

using DifferentialEquations

"""
    generate geodesics γ: I -> ℳ satisfying γ₀ = x ∈ ℳ and γ̇₀ = v ∈ 𝑇ₓℳ
"""

# Evaluates the geodesic at time t
function Geodesic(x, v, t, ℳ)
    function GeodesicEquation(du,u,p,t)
        _Γ = Γ(u, ℳ)
        @einsum out[i] := -_Γ[i,j,k]*du[j]*du[k]
        out
    end

    tspan = (0.,t)
    prob = SecondOrderODEProblem(GeodesicEquation, v, x, tspan)
    sol = DifferentialEquations.solve(prob, Vern7())
    return sol
end

sol = Geodesic([1.,1.], [1.,0.] , 1.0 , 𝕊)

u = extractcomp(sol.u,3)
v = extractcomp(sol.u,4)
x = extractcomp([F([u[i], v[i]], 𝕊) for i in 1:length(u)], 1)
y = extractcomp([F([u[i], v[i]], 𝕊) for i in 1:length(u)], 2)
z = extractcomp([F([u[i], v[i]], 𝕊) for i in 1:length(u)], 3)
SpherePlot(x,y,z,𝕊)
