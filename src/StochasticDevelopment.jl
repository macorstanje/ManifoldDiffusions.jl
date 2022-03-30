"""
    SDESolver

Abstract (super-)type for solving methods for stochastic differential equations 
in Stratonovich form.
"""
abstract type SDESolver end

"""
    Heun

Euler-Heun integration scheme for SDE's in Stratonovich form.
"""
struct Heun <: SDESolver
end

function IntegrateStep!(::Heun, u::Frame, dZ)
    q, ν, n, ℳ = u.q, u.ν, u.n, u.ℳ

    ū::typeof(u) = u + sum([Hor(i,u)*dZ[i] for i in eachindex(dZ)])
    u= u + sum([0.5(Hor(i,ū)+Hor(i,u))*dZ[i] for i in eachindex(dZ)])
    return u
    # _Γ = Γ(q, n, ℳ)
    # @einsum qᴱ[i] := q[i] + ν[i,j].*dZ[j]
    # @einsum νᴱ[i,j] := ν[i,j] - _Γ[i,k,l]*ν[k,m]*ν[l,j]*dZ[m]
    # @einsum qNew[i] := q[i] + 0.5*(νᴱ[i,j]+ν[i,j])*dZ[j]
    # _Γᴱ = Γ(qᴱ, n, ℳ)
    # @einsum νNew[i,j] := ν[i,j] - 0.5*(_Γᴱ[i,k,l]*νᴱ[k,m]*νᴱ[l,j]+ _Γ[i,k,l]*ν[k,m]*ν[l,j])*dZ[m]
    # return Frame(qNew, νNew, n, ℳ)
end


"""
    StochasticDevelopment!(method::SDESolver, Y::SamplePath, Z::SamplePath, u₀::Frame)

Simulate the process ``\\{U_t\\}_t`` on ``\\mathrm{F}(\\mathcal{M})`` starting at
`u₀` that solves the SDE ``\\mathrm{d}U_t = H_i(U_t) \\circ \\mathrm{d}Z_t^i``
This function writes the process in `Fℳ` in place of `Y`
"""
function StochasticDevelopment!(method::SDESolver, Y::SamplePath, Z::SamplePath, u₀::Frame)
    ℳ = u₀.ℳ
    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:N-1
        yy[..,k]=y
        dz = zz[k+1] - zz[k]
        x = ϕ⁻¹(y.q, y.n, ℳ)
        i = rand(findall(==(1), inChart(x, ℳ)))
        y_temp = i != y.n ? Frame(ϕ(x,i,ℳ),Dϕ(x, i, ℳ)*Dϕ⁻¹(y.q,y.n,ℳ)*y.ν, i, ℳ) : y
        y_temp = IntegrateStep!(method, y_temp, dz)
        @assert inChartRange(y_temp.q, ℳ)[i] "Left range of chart $i"
        y = y_temp
    end
    yy[..,N] = y
    Y
end

function StochasticDevelopment(method::SDESolver, Z, u₀)
    let X = Bridge.samplepath(Z.tt, zero(u₀)); StochasticDevelopment!(method, X, Z, u₀); X end
end