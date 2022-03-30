include("../ManifoldDiffusions.jl/src/ManifoldDiffusions.jl")
using StaticArrays, LinearAlgebra, Plots
using Bridge
using Main.ManifoldDiffusions
using Einsum
#include("../GuidedProposals/GuidedProposals.jl")
const ℝ{N} = SVector{N, Float64}
T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)

L = SMatrix{3,3}(1.0I)
Σdiagel = 10^(-3)
Σ = SMatrix{3,3}(Σdiagel*I)

extractcomp(v, i) = map(x->x[i], v)


T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)

L = SMatrix{3,3}(1.0I)
Σdiagel = 10^(-3)
Σ = SMatrix{3,3}(Σdiagel*I)

extractcomp(v, i) = map(x->x[i], v)


S = Sphere(2, 1.0)
x₀ = [0.0,-1.0,0.0]
ν₁ = ManifoldDiffusions.Dϕ(x₀, 1, S)*[1.,0.,0.]
ν₂ = ManifoldDiffusions.Dϕ(x₀, 1, S)*[0.,0.,1.]
u₀ = Frame(ϕ(x₀,1,S), [1.0 0.0 ; 0.0 -1.0],1, S)
W = sample(tt, Wiener{ℝ{2}}())
UU = StochasticDevelopment(Heun(), W, u₀)

# charts = map(u -> u.n, UU.yy)
# plot(UU.tt, charts)
XX = map(u->ϕ⁻¹(u.q, u.n ,u.ℳ), UU.yy)
plotly()
SpherePlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3), S)

struct SphereAngularBM <: ContinuousTimeProcess{ℝ{2}}
end

struct SphereStereoBM1 <: ContinuousTimeProcess{ℝ{2}} 
end

struct SphereStereoBM2 <: ContinuousTimeProcess{ℝ{2}}
end


function F₁(x)
    u, v = x[1], x[2]
    return [2*u/(u^2+v^2+1), 2*v/(u^2+v^2+1), (u^2+v^2-1)/(u^2+v^2+1)]
end

function F₂(x)
    u, v = x[1], x[2]
    return [sin(u)*cos(v), sin(u)*sin(v), cos(u)]
end

function Bridge.b(t,x,P::SphereStereoBM1) 
    gx = ManifoldDiffusions.g♯(x,1,S)
    _Γ = ManifoldDiffusions.Γ(x,1,S)
    @einsum out[i] := -0.5*gx[j,k]*_Γ[i,j,k]
    return out
    # u,v = x[1],x[2]
    # return [(u^2+v^2+1)*(u-v)/4, 0.0]
    #[ 0.0 , 1/(2*tan(x[2]))]
end
   
#Bridge.σ(t,x,P::SphereAngularBM) = [1/abs(sin(x[2])) 0.0 ; 0.0 1.0]
function Bridge.σ(t,x,P::SphereStereoBM1)
    gx = ManifoldDiffusions.g♯(x,1,S)
    return cholesky(gx).U
    # return [2/(u^2+v^2+1) 0.0 ; 0.0 2/(u^2+v^2+1)]
end

function Bridge.b(t,x,P::SphereStereoBM2)
    u,v = x[1],x[2]
    # return [(u^2+v^2+1)*(u-v)/4, 0.0]
    return [0.,0.]
end
   
function Bridge.σ(t,x,P::SphereStereoBM2)
    u,v = x[1],x[2]
    return [(u^2+v^2+1)/2 0.0 ; 0.0 (u^2+v^2+1)/2]
end

S = Sphere(2, 1.0)
P1 = SphereStereoBM1()
P2 = SphereStereoBM2()
# W = sample(tt, Wiener{ℝ{2}}())

YS1 = Bridge.solve(Euler(), [0.0, -1.0], W, P1)
YS2 = Bridge.solve(Euler(), [0.0, -1.0], W, P2)
XS1 = map(y->F₁(y), YS1.yy)
XS2 = map(y->F₁(y), YS2.yy)
XS11 = extractcomp(XS1, 1) ; XS12 = extractcomp(XS1,2) ; XS13 = extractcomp(XS1,3)
XS21 = extractcomp(XS1, 1) ; XS22 = extractcomp(XS1,2) ; XS23 = extractcomp(XS1,3)

# = cos.(YA1).*sin.(YA2) ; XA2 = sin.(YA1).*sin.(YA2) ; XA3 = cos.(YA2)
plotly()
SpherePlot(XS11, XS12,XS13, S)
#SpherePlot(XS21, XS22,XS23, S)



struct FrameBM <: ContinuousTimeProcess{ℝ{6}}
end

Bridge.b(t,u,P::FrameBM) = zero(u)
function Bridge.σ(t,u,P::FrameBM)
    d = Int64(0.5*sqrt(1+4*length(u))-0.5)
    q = u[1:d]
    ν = reshape(u[d+1:end],d,d)
    _Γ = ManifoldDiffusions.Γ(q,1,S)
    dq = ν
    @einsum dν[i,j,m] := -0.5*_Γ[i,k,l]*ν[k,m]*ν[l,j]
    return vcat(dq, dν[:,1,:], dν[:,2,:])
end

u₀ = [0.0, -1.0, 1. ,0., 0. ,-1.]
UU2 = Bridge.solve(StochasticHeun(), u₀, W, FrameBM())

XF = map(x -> ManifoldDiffusions.ϕ⁻¹(x[1:2],1,S), UU2.yy[1:end-1])
XF1 = extractcomp(XF, 1) ; XF2 = extractcomp(XF,2) ; XF3 = extractcomp(XF,3)
SpherePlot(XF1,XF2,XF3,S)










function integratestep(dZ,x,n)
    d = Int64(0.5*sqrt(1+4*length(x))-0.5)
    q = x[1:d]
    ν = reshape(x[d+1:end],d,d)
    _Γ = ManifoldDiffusions.Γ(q,1,S)
    @einsum qᴱ[i] := q[i] + ν[i,j]*dZ[j] 
    @einsum νᴱ[i,j] := ν[i,j] - _Γ[i,k,l]*ν[k,m]*ν[l,j]*dZ[m]
    _Γᴱ = ManifoldDiffusions.Γ(qᴱ, n, S)
    @einsum qNew[i] := q[i] + 0.5*(νᴱ[i,j]+ν[i,j])*dZ[j]
    @einsum νNew[i,j] := ν[i,j] - 0.5*(_Γᴱ[i,k,l]*νᴱ[k,m]*νᴱ[l,j]+ _Γ[i,k,l]*ν[k,m]*ν[l,j])*dZ[m]
    return vcat(qNew, νNew[:,1], νNew[:,2])
end

u₀ = [0.0, -1.0, 1. ,0., 0. ,-1.]
uu = [u₀]
ww = W.yy ; tt = W.tt
for k in 1:length(tt)-1
    dw = ww[k+1]-ww[k]
    push!(uu, integratestep(dw, uu[k],1))
end
U = Bridge.samplepath(W.tt, uu)


x₀ = [1.0,0.0,0.0]
u₀ = Frame(x₀, [0.0 1.0 ; 1.0 0.0], S)
UU = StochasticDevelopment(W, u₀, S)

XX = map(u->u.x, UU.yy)
plotly()
SpherePlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3), S)

plot(tt, XA1.^2 .+XA2.^2 .+XA3.^2)


d=2
q = u₀[1:d]
ν = reshape(u₀[d+1:end],d,d)
_Γ = ManifoldDiffusions.Γ(q,1,S)
dw = ww[2]-ww[1]
@einsum qᴱ[i] := q[i] + ν[i,j]*dw[j]
@einsum νᴱ[i,j] := ν[i,j] - _Γ[i,k,l]*ν[k,m]*ν[l,j]*dw[m]
_Γᴱ = ManifoldDiffusions.Γ(qᴱ, 1, S)
@einsum qNew[i] := q[i] + 0.5*(νᴱ[i,j]+ν[i,j])*dw[j]
@einsum νNew[i,j] := ν[i,j] - 0.5*(_Γᴱ[i,k,l]*νᴱ[k,m]*νᴱ[l,j]+ _Γ[i,k,l]*ν[k,m]*ν[l,j])*dw[m]

ν₁ = ManifoldDiffusions.Dϕ(x₀, 1, S)*[1.,0.,0.]
ν₂ = ManifoldDiffusions.Dϕ(x₀, 1, S)*[0.,0.,1.]

ManifoldDiffusions.Dϕ⁻¹(q₀, 1, S)*[0.,-1.]
"""
    The object SphereDiffusion(σ, 𝕊) can be used to generate a diffusion
    on the sphere 𝕊. We will focus on the diffusion equation
        `` dX_t = Σ^½ P(X_t)∘dW_t ``
    where σ ∈ ℝ
"""

struct SphereDiffusion{T} <: ContinuousTimeProcess{ℝ{3}}
    Σ::T
    𝕊::Sphere

    function SphereDiffusion(σ::T, 𝕊::Sphere) where {T<:Real}
        if σ == 0
            error("σ cannot be 0")
        end
        new{T}(σ, 𝕊)
    end
end

Bridge.b(t, x, ℙ::SphereDiffusion{T}) where {T} = zeros(3)
Bridge.σ(t, x, ℙ::SphereDiffusion{T}) where {T} = ℙ.Σ*P(x, 𝕊)
Bridge.constdiff(::SphereDiffusion{T}) where {T} = false

"""
    Example: Constructing a Brownian motion on a sphere of radius 1
"""

𝕊 = Sphere(1.0)
ℙ = SphereDiffusion(1.0, 𝕊)

x₀ = [0.,0.,1.]
W = sample(0:dt:T, Wiener{ℝ{3}}())
X = solve(StratonovichEuler(), x₀, W, ℙ)

plotly()
SpherePlot(X, 𝕊)

"""
    Insert the settings for the auxiliary process tildeX
        and set partial bridges for each data point
"""
struct SphereDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    ξ
    σ
    B
end

Bridge.B(t, ℙt::SphereDiffusionAux) = ℙt.B
Bridge.β(t, ℙt::SphereDiffusionAux) = zeros(3)
Bridge.σ(t, ℙt::SphereDiffusionAux) = ℙt.σ
Bridge.b(t, x, ℙt::SphereDiffusionAux) = Bridge.B(t, ℙt)*x + Bridge.β(t,ℙt)
Bridge.a(t, ℙt::SphereDiffusionAux) = Bridge.σ(t, ℙt)*Bridge.σ(t, ℙt)'
Bridge.constdiff(::SphereDiffusionAux) = true

"""
    Now let us create a proposal diffusion bridge that hits ξ at time T
    we use the transition density of tildeX in the guided proposal

"""
ξ = [0.,1.,0.]
f(ξ, 𝕊) # This should be zero

ℙt = SphereDiffusionAux(ξ, P(ξ, 𝕋), [rand() rand() rand() ; rand() rand() rand() ; rand() rand() rand()])

"""
    Settings for the Guided proposal
"""
Φ(t, ℙt::SphereDiffusionAux) = exp(ℙt.B*t)
Φ(t, s, ℙt::SphereDiffusionAux) = exp(ℙt.B*(t-s)) # = Φ(t)Φ(s)⁻¹
Υ = Σ

Lt(t, ℙt::SphereDiffusionAux) = L*Φ(T, t, ℙt)
μt(t, ℙt::SphereDiffusionAux) = zeros(3)


M⁺ = zeros(typeof(Σ), length(tt))
M = copy(M⁺)
M⁺[end] = Υ
M[end] = inv(Υ)
for i in length(tt)-1:-1:1
    dt = tt[i+1] - tt[i]
    M⁺[i] = M⁺[i+1] + Lt(tt[i+1], ℙt)*Bridge.a(tt[i+1], ℙt)*Lt(tt[i+1], ℙt)'*dt + Υ
    M[i] = inv(M⁺[i])
end

H((i, t)::IndexedTime, x, ℙt::SphereDiffusionAux) = Lt(t, ℙt)'*M[i]*Lt(t, ℙt)
r((i, t)::IndexedTime, x, ℙt::SphereDiffusionAux) = Lt(t, ℙt)'*M[i]*(ℙt.ξ .- μt(t, ℙt) .- Lt(t, ℙt)*x)

struct GuidedProposalSphere <: ContinuousTimeProcess{ℝ{3}}
    ξ
    Target::SphereDiffusion
    Auxiliary::SphereDiffusionAux
end

function Bridge.b(t, x, ℙᵒ::GuidedProposalSphere)
    k = findmin(abs.(tt.-t))[2]
    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary
    a = Bridge.σ(t, x, ℙ)*Bridge.σ(t, x, ℙ)'
    return Bridge.b(t, x, ℙ) + a*r((k, tt[k]), x, ℙt)
end

Bridge.σ(t, x, ℙᵒ::GuidedProposalSphere) = Bridge.σ(t, x, ℙᵒ.Target)
Bridge.constdiff(::GuidedProposalSphere) = false

ℙᵒ = GuidedProposalSphere(ξ, ℙ, ℙt)
W = sample(0:dt:T, Wiener{ℝ{3}}())
Xᵒ = solve(StratonovichEuler(), [0.,0.,1.], W, ℙᵒ)

plotly()
plot([extractcomp(Xᵒ.yy, 1), extractcomp(Xᵒ.yy, 2), extractcomp(Xᵒ.yy, 3)])
SpherePlot(Xᵒ, 𝕊)
plot!([0.], [0.], [1.],
        legend = true,
        color = :red,
        seriestype = :scatter,
        markersize = 1.5,
        label = "start")
plot!([ξ[1]],  [ξ[2]],  [ξ[3]],
        color = :yellow,
        seriestype = :scatter,
        markersize = 1.5,
        label = "end")
