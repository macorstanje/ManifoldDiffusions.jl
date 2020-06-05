"""
    FrameBundle

The object `FrameBundle(ℳ)` represents the frame bundle over a manifold ℳ.
"""
struct FrameBundle{TM} <: EmbeddedManifold
    ℳ::TM
    FrameBundle(ℳ::TM) where {TM<:EmbeddedManifold} = new{TM}(ℳ)
end

Σ(u::Frame, v::T, w::T) where {T<:AbstractArray} = dot(inv(u.ν)*v , inv(u.ν)*w)
"""
    g(X::TangentFrame, Y::TangentFrame)

Adds a Riemannian structure to the Frame bundle by introducing a cometric
"""
function g(X::TangentFrame, Y::TangentFrame)
        if X.u != Y.u
            error("Vectors are in different tangent spaces")
        end
    return Σ(X.u, Πˣ(X), Πˣ(Y))
end

"""
    Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM})

Returns the Hamiltonian that results from the cometric `g`.
"""
function Hamiltonian(u::Frame, p::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    if p.u != u
        error("p is not tangent to u")
    end
    return .5*g(p,p)
end

"""
    Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM})

Different representation of the Hamiltonian as functions of two vectors of size
``d+d^2``
"""
function Hamiltonian(x::Tx, p::Tp, Fℳ::FrameBundle{TM}) where {Tx, Tp<:AbstractArray, TM}
    N = length(x)
    d = Int64((sqrt(1+4*N)-1)/2)
    u = Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d), Fℳ.ℳ)
    P = TangentFrame(u, p[1:d], reshape(p[d+1:d+d^2], d, d))
    return Hamiltonian(u, P, Fℳ)
end

"""
    Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM})

Returns a geodesic on Fℳ starting at `u₀` with initial velocity `v₀` and
evaluated at a discretized time interval `tt`.
"""
function Geodesic(u₀::Frame, v₀::TangentFrame, tt, Fℳ::FrameBundle{TM}) where {TM}
    d = length(u₀.x)
    if d==1
        U₀ = [u₀.x, u₀.ν]
        V₀ = [v₀.ẋ, v₀.ν̇]
    else
        U₀ = vcat(u₀.x, vec(reshape(u₀.ν, d^2, 1)))
        V₀ = vcat(v₀.ẋ, vec(reshape(v₀.ν̇, d^2, 1)))
    end
    xx, pp = Integrate(Hamiltonian, tt, U₀, V₀, Fℳ)
    uu = map(x->Frame(x[1:d] , reshape(x[d+1:d+d^2], d, d), Fℳ.ℳ) , xx)
    vv = map(p->TangentFrame(u₀, p[1:d], reshape(p[d+1:d+d^2], d, d)), pp)
    return uu, vv
end

"""
    ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM})

The exponential map on Fℳ starting from `u₀` with initial velocity `v₀`.
"""
function ExponentialMap(u₀::Frame, v₀::TangentFrame, Fℳ::FrameBundle{TM}) where {TM}
    tt = collect(0:0.01:1)
    uu, vv = Geodesic(u₀, v₀, tt, Fℳ)
    if length(u₀.x)>1
        return uu[end]
    else
        return Frame(uu[end].x[1], uu[end].ν[1], Fℳ.ℳ)
    end
end



function IntegrateStep(dW, u::Frame, ℳ)
    x, ν = u.x, u.ν
    # uᴱ = ExponentialMap(u, sum([Hor(i, u,ℳ)*dW[i] for i in eachindex(dW)]), FrameBundle(ℳ))
    # y = ExponentialMap(u, sum([(Hor(i,uᴱ,ℳ) + Hor(i, u,ℳ))*dW[i]*0.5 for i in eachindex(dW)]), FrameBundle(ℳ))
    uᴱ = u + sum([Hor(i, u,ℳ)*dW[i] for i in eachindex(dW)])
    y = u + sum([(Hor(i,uᴱ,ℳ) + Hor(i, u,ℳ))*dW[i]*0.5 for i in eachindex(dW)])
    return y
end

"""
    StochasticDevelopment!(Y, W, u₀, ℳ; drift)

Simulate the process {Ut} on F(ℳ) starting at `u₀` that solves the SDE
    ```math
    dUt = V⁺(Ut)dt + H(Ut)∘dWt
    ```
This function writes the process in Fℳ in place of `Y`
"""
function StochasticDevelopment!(Y, W, u₀, ℳ; drift)
    tt = W.tt
    ww = W.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:length(tt)-1
        dw = ww[k+1] - ww[k]
        yy[..,k] = y
        y = IntegrateStep(dw, y, ℳ)
        if drift
            dt = tt[k+1] - tt[k]
            y += Vᵒ(tt[k], y, ℳ)*dt
        end
    end
    yy[..,length(tt)] = y
    Y
end

function StochasticDevelopment(W, u₀, ℳ; drift)
    let X = Bridge.samplepath(W.tt, zero(u₀)); StochasticDevelopment!(X, W, u₀,ℳ; drift = drift); X end
end
