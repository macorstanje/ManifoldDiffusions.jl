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


S = Sphere(2, 1.0)
x₀ = [0.,0.,1.]; p₀ = [1.,-1.] #ForwardDiff.jacobian((p) -> ϕ⁻¹(p,2, S), ϕ(x₀,2,S))*[1.,-1.]

xx, vv = Geodesic(x₀, p₀, tt, S)
plotly()
SpherePlot(extractcomp(xx,1), extractcomp(xx,2), extractcomp(xx,3), S)

T = Torus([Sphere(1,3.0), Sphere(1,1.0)])
function Dim4toDim3(x::T, 𝕋::Torus) where {T<:AbstractArray}
    x,y,z,w = x[1],x[2],x[3],x[4]
    R = 𝕋.Spheres[1].R
    return [x+z*x/R, y+z*y/R, w]
end
x₀ = [3.0*cos(pi/2), 3.0*sin(pi/2) , 1.0*cos(pi), 1.0*sin(pi)]
xx, vv = Geodesic(x₀, p₀, tt, T)

x = x₀ ; y = p₀ ; xx = [x] ; yy = [y]
for i in 1:length(tt)-1
    dt = tt[i+1] - tt[i]
    n = rand(findall(==(1), inChart(x, S)))
    q = ϕ(x,n,S)
    dq = y*dt
    _Γ = Γ(ϕ(x,n,S),n,S)
    @einsum dy[k] := y[l]*y[j]*_Γ[k,l,j]
    y += dy
    q += dq
    x = ϕ⁻¹(q,n,S)
    push!(xx,x)
    push!(yy,y)
end
 

"""
    The object TorusDiffusion(σ, 𝕋) can be used to generate a diffusion
    on the Torus 𝕋. We will focus on the diffusion equation
        `` dX_t = Σ P(X_t)∘dW_t ``
    where Σ ∈ ℝ
"""

S1 = Sphere(1, 3.0) ; S2 = Sphere(1, 1.0)
𝕋 = Torus([S1,S2])
x₀ = [3.0*cos(pi/2), 3.0*sin(pi/2) , 1.0*cos(pi), 1.0*sin(pi)]


W = sample(tt, Wiener{ℝ{2}}())

u₀ = Frame(x₀ , [1.0 0.0 ; 0.0 1.0], 𝕋)
U = StochasticDevelopment(W, u₀, 𝕋)



XX = map(u->Dim4toDim3(u.x, 𝕋), U.yy)

plotly()
TorusPlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3),𝕋)


struct TorusAngularBM <: ContinuousTimeProcess{ℝ{2}}
    R::Float64
    r::Float64
end

function Bridge.b(t, x, ℙ::TorusAngularBM)
    u, v, R, r = x[1], x[2], ℙ.R, ℙ.r
    return [0.0 , -sin(v)/(2*r*(R+r*cos(v)))]
end

Bridge.σ(t, x, ℙ::TorusAngularBM) = [1/abs(ℙ.R+ℙ.r*cos(x[2])) 0.0 ; 0.0 1/ℙ.r]
function Trans(x, 𝕋::Torus)
    R, r = 𝕋.Spheres[1].R, 𝕋.Spheres[2].R
    u,v = x[1],x[2]
    return [(R+r*cos(v))*cos(u) , (R+r*cos(v))*sin(u), r*sin(v)]
end

Trans([pi/2,pi],𝕋)

𝕋2 = TorusAngularBM(3.0,1.0)
XX2 = Bridge.solve(Euler(), [pi/2, pi], W, 𝕋2)
XXX1 = (3.0.+cos.(extractcomp(XX2.yy,2))).*cos.(extractcomp(XX2.yy,1))
XXX2 = (3.0.+cos.(extractcomp(XX2.yy,2))).*sin.(extractcomp(XX2.yy,1))
XXX3 = sin.(extractcomp(XX2.yy,2))
TorusPlot(XXX1, XXX2, XXX3,𝕋)


plot(tt, XXX1)
plot!(tt, XXX2)
plot!(tt, XXX3)

plot(tt, extractcomp(XX,1))
plot!(tt, extractcomp(XX,2))
plot!(tt, extractcomp(XX,3))

    fig = plot(tt, XXX2, label = "Angular y")
    plot!(fig, tt, extractcomp(XX,2), label = "Framebundle y")
fig






𝕊 = Sphere(2, 1.0)
x₀ = [1.0,0.0,0.0]
u₀ = Frame(x₀, [1.0 0.0 ; 0.0 1.0], 𝕊)
W = sample(tt, Wiener{ℝ{2}}())
UU = StochasticDevelopment(W, u₀, 𝕊)

XX = map(u->u.x, UU.yy)
plotly()
SpherePlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3), 𝕊)

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
    u,v = x[1],x[2]
    return [(u^2+v^2+1)*(u-v)/4, 0.0]
    #[ 0.0 , 1/(2*tan(x[2]))]
end
   
#Bridge.σ(t,x,P::SphereAngularBM) = [1/abs(sin(x[2])) 0.0 ; 0.0 1.0]
function Bridge.σ(t,x,P::SphereStereoBM1)
    gx = ManifoldDiffusions.g♯(x,1,S)
    return cholesky(gx).U
    return [2/(u^2+v^2+1) 0.0 ; 0.0 2/(u^2+v^2+1)]
end

function Bridge.b(t,x,P::SphereStereoBM2)
    u,v = x[1],x[2]
    return [(u^2+v^2+1)*(u-v)/4, 0.0]
end
   
function Bridge.σ(t,x,P::SphereStereoBM2)
    u,v = x[1],x[2]
    return [2/(u^2+v^2+1) 0.0 ; 0.0 2/(u^2+v^2+1)]
end

S = Sphere(2, 1.0)
P1 = SphereStereoBM1()
P2 = SphereStereoBM2()
W = sample(tt, Wiener{ℝ{2}}())

YS1 = Bridge.solve(Euler(), [1.0, 0.0], W, P1)
#YS2 = Bridge.solve(Euler(), [1.0, 0.0], W, P2)
XS1 = map(y->F₁(y), YS1.yy)
#XS2 = map(y->F₁(y), YS2.yy)
XS11 = extractcomp(XS1, 1) ; XS12 = extractcomp(XS1,2) ; XS13 = extractcomp(XS1,3)
#XS21 = extractcomp(XS1, 1) ; XS22 = extractcomp(XS1,2) ; XS23 = extractcomp(XS1,3)

# = cos.(YA1).*sin.(YA2) ; XA2 = sin.(YA1).*sin.(YA2) ; XA3 = cos.(YA2)
plotly()
SpherePlot(XS11, XS12,XS13, S)
#SpherePlot(XS21, XS22,XS23, S)

x₀ = [1.0,0.0,0.0]
u₀ = Frame(x₀, [0.0 1.0 ; 1.0 0.0], S)
UU = StochasticDevelopment(W, u₀, S)

XX = map(u->u.x, UU.yy)
plotly()
SpherePlot(extractcomp(XX,1), extractcomp(XX,2), extractcomp(XX,3), S)

plot(tt, XA1.^2 .+XA2.^2 .+XA3.^2)








struct TorusDiffusion{T} <: ContinuousTimeProcess{ℝ{3}}
    Σ::T
    𝕋::Torus

    function TorusDiffusion(σ::T, 𝕋::Torus) where {T<:Real}
        if σ == 0
            error("σ cannot be 0")
        end
        new{T}(σ, 𝕋)
    end
end

Bridge.b(t, x, ℙ::TorusDiffusion{T}) where {T} = zeros(3)
Bridge.σ(t, x, ℙ::TorusDiffusion{T}) where {T} = ℙ.Σ*P(x, 𝕋)
Bridge.constdiff(::TorusDiffusion{T}) where {T} = false

"""
    Example: Constructing a Brownian motion on a Torus with
    inner radius r = ½ and outer radius R = 2
"""

𝕋 = Torus(2.0, 0.5)
ℙ = TorusDiffusion(1.0, 𝕋)

x₀ = [2.,0.,0.5]

function SimulatePoints(n, x₀, ℙ::TorusDiffusion)
    out = ℝ{3}[]
    it = 0
    while length(out) < n
        W = sample(0.:dt:T, Wiener{ℝ{3}}())
        X = solve(StratonovichEuler(),x₀, W, ℙ)
        if abs(f(X.yy[end], ℙ.𝕋)) <= 0.06
            push!(out, X.yy[end])
        end
        it += 1
    end
    return out, it
end

@time ξ, it = SimulatePoints(25, x₀, ℙ)

plotly()
TorusScatterPlot(extractcomp(ξ,1), extractcomp(ξ,2), extractcomp(ξ,3), x₀, 𝕋)
"""
    Insert the settings for the auxiliary process tildeX
        and set partial bridges for each data point

    Now let us create a proposal diffusion bridge that hits ξᵢ at time T
    we use the transition density of tildeX in the guided proposal

"""

# returns b(T, ξ), when the SDE dX_t = P(X_t)∘dW_t is in Ito form.
function bT(ξ)
    out = zeros(eltype(ξ), 3)
    for i = 1:3
        for k = 1:3
            Pr = (z) -> P(z, 𝕋)[i, k]
            grad = ForwardDiff.gradient(Pr, ξ)
            for j = 1:3
                out[i] += 0.5 * P(ξ, 𝕋)[j, k] * grad[j]
            end
        end
    end
    out
end

struct TorusDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    ξ
    σ
    B
end

Bridge.B(t, ℙt::TorusDiffusionAux) = ℙt.B
Bridge.β(t, ℙt::TorusDiffusionAux) = bT(ℙt.ξ) .- ℙt.B*ℙt.ξ
Bridge.σ(t, ℙt::TorusDiffusionAux) = ℙt.σ
Bridge.b(t, x, ℙt::TorusDiffusionAux) = Bridge.B(t, ℙt)*x + Bridge.β(t,ℙt)
Bridge.a(t, ℙt::TorusDiffusionAux) = Bridge.σ(t, ℙt)*Bridge.σ(t, ℙt)'
Bridge.constdiff(::TorusDiffusionAux) = true

"""
    Likelihood-based inference
"""
function RandomMatrix()
    [rand() rand() rand() ; rand() rand() rand() ; rand() rand() rand()]
end

ℙt = [TorusDiffusionAux(ξ[i], P(ξ[i], 𝕋), RandomMatrix()) for i in 1:length(ξ)]
ℙᵒ = [GuidedProposal(ξ[i], ℙ, ℙt[i], tt, Σ, L) for i in 1:length(ξ)]

# Likelihood on a grid of points
GridSize = 50
ϑ = [(0:GridSize-2) * 2 / (GridSize - 1); 2]
φ = [(0:GridSize-2) * 2 / (GridSize - 1); 2]
x = [(𝕋.R+𝕋.r*cospi(φ))*cospi(ϑ) for ϑ in ϑ, φ in φ]
y = [(𝕋.R+𝕋.r*cospi(φ))*sinpi(ϑ) for ϑ in ϑ, φ in φ]
z = [𝕋.r*sinpi(φ) for ϑ in ϑ, φ in φ]

points = vec(Point3f0.(x, y, z))

function LogLikelihoodGrid(points)
    n = length(points)
    p = Progress(n, 1, "Percentage completed: ", 50)
    ℓ_grid = Float64[]
    for pts in points
        pnt = Float64.([pts[1], pts[2], pts[3]])
        WW = [sample(0:dt:T, Wiener{ℝ{3}}()) for i in 1:length(ξ)]
        ll = sum([logp!(LeftRule(), Stratonovich(), WW[i], pnt, ℙᵒ[i]) for i in 1:length(ξ)])
        push!(ℓ_grid, ll)
        next!(p)
    end
    return ℓ_grid
end

ℓ_grid = LogLikelihoodGrid(points)
df = DataFrame(
    l = ℓ_grid,
    x = extractcomp(points, 1),
    y = extractcomp(points, 2),
    z = extractcomp(points, 3),
)

# Make a plot in R, save the dataframe using
outdir = "/Users/marc/Documents/Manifolds/DataFrames/"
CSV.write(outdir*"Torus.csv", df)

"""
    Function that converts local coordinates to ℝ³-valued vectors
    x = (R + rcos(v))cos(u)
    y = (R + rcos(v))sin(u)
    z = rsin(v)

    with the corresponding Riemannian metric G
"""

ϕ(x::T, 𝕋::Torus) where {T<:AbstractArray} = [
    (𝕋.R + 𝕋.r * cos(x[2])) * cos(x[1]),
    (𝕋.R + 𝕋.r * cos(x[2])) * sin(x[1]),
    𝕋.r * sin(x[2]),
]

G(v, 𝕋::Torus) = [(𝕋.R + 𝕋.r*cos(v))^2 0 ; 0 𝕋.r^2]
G⁻¹(v, 𝕋::Torus) = [1/(𝕋.R + 𝕋.r * cos(v))^2 0; 0 1 / 𝕋.r^2]

"""
    A MALA algorithm to draw samples from the likelihood

    We wish to sample for a log-likelihood of all data, defined through π
"""

# A loglikelihood given a set of bridges XX
function logπ(XX, x₀, ℙᵒ)
    if length(x₀) == 3
        return sum([logp(LeftRule(), XX[i], x₀, ℙᵒ[i]) for i in 1:length(ℙᵒ)])
    end
    if length(x₀) == 2
        return sum([logp(LeftRule(), XX[i], ϕ(x₀, 𝕋), ℙᵒ[i]) for i in 1:length(ℙᵒ)])
    end
end

# Simulate bridges from standard Brownian motions and return the loglikelihood
function logπ!(WW, x₀, ℙᵒ)
    if length(x₀) == 3
        return sum([logp!(LeftRule(), Stratonovich(), WW[i], x₀, ℙᵒ[i]) for i in 1:length(ℙᵒ)])
    end
    if length(x₀) == 2
        return sum([logp!(LeftRule(), Stratonovich(), WW[i], ϕ(x₀, 𝕋), ℙᵒ[i]) for i in 1:length(ℙᵒ)])
    end
end

logπ(XX, ℙ) = (x₀) -> logπ(XX, x₀, ℙᵒ)
logπ!(WW, ℙ) = (x₀) -> logπ!(WW, x₀, ℙᵒ)

function adaptmalastep!(δ, n, accinfo)
    adaptskip = 10
    if mod(n, adaptskip) == 0
        η(n) = min(0.1, 10 / sqrt(n))

        targetaccept = 0.5

        recent_mean = (accinfo[end] - accinfo[end-adaptskip+1]) / adaptskip
        if recent_mean > targetaccept
            δ *= exp(η(n))
        else
            δ *= exp(-η(n))
        end
    end
end



"""
    Apply Langevin adjusted updates:

    Descretize the Langevin equation
        dXt = (1/2)*∇_x log π (Χt) dt + P(Xt) ∘ dWt

    Euler discretization yields updates of the form

    xᵒ = x + h/2 ∇_x log π(x) + √(h)Z , where Z is N(0,1)-distributed

    Riemannian mala:

    We create a composition (u,v) ↦ ϕ(u,v) ↦ log π(ϕ(u,v))
    We then update using the scheme

    xᵒ = x + h/2 G⁻¹(u,v) ∇_(u,v) log π(ϕ(u,v)) + √( hG⁻¹(u,v) ) Z
"""

function MALA(ξ, ℙᵒ, ρ)
    n = length(ξ) # sample size
    acc = Int64[]
    push!(acc, 0)
    # random starting point
    u = 2 * π * rand()
    v = 2 * π * rand()
    uu, vv = [u], [v]
    x = ϕ([u, v], 𝕋)
    xx = [x]


    # Initial set of bridges to the data points
    TW = SamplePath{SArray{Tuple{3},Float64,1,3}}
    WW = TW[] # n standard brownian motions are stored in WW
    XX = TW[]  # n diffusion bridges between x and each of the ξᵢ are stored in XX
    ll = Float64[]
    for i = 1:n
        push!(WW, sample(0.0:dt:T, Wiener{ℝ{3}}()))
        push!(XX, deepcopy(WW[i]))
        push!(ll, logp!(LeftRule(), Stratonovich(), XX[i], x, ℙᵒ[i]))
    end

    # Αrray containing the gradients
    ∇ = ForwardDiff.gradient(logπ(XX, ℙᵒ), [u,v])
    ∇logπ = [∇]

    # Start iterating
    h = 0.001
    for iter = 1:200
        # Update the driving Brownian motions
        # W₂ = deepcopy(WW[1])
        # Wᵒ = deepcopy(WW[1])
        # for i = 1:n
        #     sample!(W₂, Wiener{ℝ{3}}())
        #     Wᵒ.yy .= ρ * WW[i].yy + sqrt(1 - ρ^2) * W₂.yy
        #     Xᵒ = deepcopy(Wᵒ)
        #     llᵒ = logp!(LeftRule(), Stratonovich(), Xᵒ, x, ℙᵒ[i]) # simultaneously overwrites Xᵒ by a GP(x, Wᵒ)
        #     if log(rand()) <= llᵒ - ll[i]
        #         XX[i].yy .= Xᵒ.yy
        #         WW[i].yy .= Wᵒ.yy
        #         ll[i] = llᵒ
        #     end
        # end

        # sample proposal for starting point
        ∇ = ForwardDiff.gradient(logπ(XX, ℙᵒ), [u,v])
        μ = [u, v] + 0.5 * h * G⁻¹(v, 𝕋) * ∇
        uᵒ, vᵒ = μ + sqrt(h) * rand(MvNormal([0, 0], G⁻¹(v, 𝕋)))
        xᵒ = ϕ([uᵒ, vᵒ], 𝕋)


        # Simulate bridges and calculate the log-likelihood for the proposed starting point
        llᵒ = zeros(n)
        XXᵒ = deepcopy(WW) # Set of diffusion bridges corresponding to the proposal
        for i = 1:n
            llᵒ[i] = logp!(LeftRule(), Stratonovich(), XXᵒ[i], xᵒ, ℙᵒ[i]) # simultaneously overwrites Xᵒᵒ[i] by a GP(xᵒ, W)
        end

        print("Iteration ", iter, ": ")
        print(
            "x = (",
            round(x[1]; digits = 2),
            " , ",
            round(x[2]; digits = 2),
            " , ",
            round(x[3]; digits = 2),
            ")",
        )
        print(" and ")
        print(
            "xᵒ = (",
            round(xᵒ[1]; digits = 2),
            " , ",
            round(xᵒ[2]; digits = 2),
            " , ",
            round(xᵒ[3]; digits = 2),
            ")",
        )
        print(" , ")
        print("sumll = ", sum(ll), " , sumllᵒ = ", sum(llᵒ))


        # The gradient to log π at the proposal
        ∇ᵒ = ForwardDiff.gradient(logπ(XXᵒ, ℙᵒ), [uᵒ,vᵒ])  # important: X not overwritten!

        # Proposal distribution, where q_xy = q(x|y)
        μᵒ = [uᵒ, vᵒ] + 0.5 * h * G⁻¹(vᵒ, 𝕋) * ∇ᵒ
        q_xᵒx = Distributions.logpdf(MvNormal(μ, h * G⁻¹(v, 𝕋)), [uᵒ, vᵒ])
        q_xxᵒ = Distributions.logpdf(MvNormal(μᵒ, h * G⁻¹(vᵒ, 𝕋)), [u, v])

        # Accept/reject the proposal
        logA = sum(llᵒ) - sum(ll) + q_xxᵒ - q_xᵒx
        print(" , logA: ", logA)
        if log(rand()) <= logA
            print(" ✓")
            push!(acc, acc[end] + 1)
            for i = 1:n
                XX[i] = XXᵒ[i]
                ll[i] = llᵒ[i]
            end
            x = xᵒ
            u = uᵒ
            v = vᵒ
            ∇ = ∇ᵒ
        else
            push!(acc, acc[end])
        end
        push!(xx, x)
        push!(uu, u)
        push!(vv, v)
        push!(∇logπ, ∇)
        println()

        # Adaptive MALA
        adaptmalastep!(h, iter, acc)
    end
    xx, ∇logπ, acc, uu, vv
end

x, ∇logπ, acc, u, v = MALA(ξ, ℙᵒ, .5)
