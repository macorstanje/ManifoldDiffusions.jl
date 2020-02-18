
# The frame bundle over a manifold ℳ
struct FrameBundle{TM} <: EmbeddedManifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:EmbeddedManifold} = new{TM}(ℳ)
end



"""
    Riemannian structure on the Frame bundle
"""

# Riemannian cometric on the Frame bundle
Σ(u::Frame, v::T, w::T) where {T<:AbstractArray} = dot(inv(u.ν)*v , inv(u.ν)*w)
function g(X::TangentFrame, Y::TangentFrame)
        if X.u != Y.u
            error("Vectors are in different tangent spaces")
        end
    return Σ(X.u, Πˣ(X), Πˣ(Y))
end

# Hamiltonian
function Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    if p.u != u
        error("p is not tangent to u")
    end
    return .5*g(p,p)
end

# Hamiltonian as functions of two vectors of size d+d^2
function Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM}) where {Tx, Tp<:AbstractArray, TM}
    N = length(x)
    d = Int64((sqrt(1+4*N)-1)/2)
    u = Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d))
    P = TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d))
    return Hamiltonian(u, P, Fℳ)
end

"""
    Geodesic flow and the exponential map on the Frame bundle
"""

function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u₀.x)
    if d>1
        U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
        V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    else
        U₀ = [u₀.x, u₀.ν]
        V₀ = [v₀.ẋ, v₀.ν̇]
    end
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d)) , xx)
    vv = map(p->TangentFrame(u₀, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

function ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    tt = collect(0:0.01:1)
    uu, vv = Geodesic(u₀, v₀, tt, Fℳ)
    if length(u₀.x)>1
        return uu[end]
    else
        return Frame(uu[end].x[1], uu[end].ν[1])
    end
end

"""
    Stochastic development

    Simulate the process {Ut} on F(ℳ) given by the SDE
        dUt = H(Ut)∘dWt
"""

function IntegrateStep(dW, u::Frame, ℳ)
    x, ν = u.x, u.ν
    uᴱ = ExponentialMap(u, sum([Hor(i, u,ℳ)*dW[i] for i in eachindex(dW)]), FrameBundle(ℳ))
    y = ExponentialMap(u, sum([(Hor(i,uᴱ,ℳ) + Hor(i, u,ℳ))*dW[i]*0.5 for i in eachindex(dW)]), FrameBundle(ℳ))
    return y
end


using Bridge
StochasticDevelopment(W, u₀, ℳ) = let X = Bridge.samplepath(W.tt, zero(u₀)); StochasticDevelopment!(X, W, u₀,ℳ); X end
function StochasticDevelopment!(Y, W, u₀, ℳ)
    tt = W.tt
    ww = W.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:length(tt)-1
        dw = ww[k+1] - ww[k]
        yy[..,k] = y
        y = IntegrateStep(dw, y, ℳ)
    end
    yy[..,length(tt)] = y
    Y
end

"""
    Now let us create a stochastic process on the frame bundle of various manifolds
"""
