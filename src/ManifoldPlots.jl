"""
    SpherePlot(X::T , Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}

Plot a function that takes values X, Y, Z on the `Sphere` 𝕊.
"""
function SpherePlot(X::T , Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    R = 𝕊.R
    du = 2π/100
    dv = π/100

    u = 0.0:du:(2π+du)
    v = 0.0:dv:(π+dv)

    lenu = length(u);
    lenv = length(v);
    x = zeros(lenu, lenv); y = zeros(lenu,lenv); z = zeros(lenu,lenv)
    for i in 1:lenu
        for j in 1:lenv
            x[i,j] = R*cos.(u[i]) * sin(v[j]);
            y[i,j] = R*sin.(u[i]) * sin(v[j]);
            z[i,j] = R*cos(v[j]);
        end
    end


    Plots.surface( x,y,z,
                    axis=true,
                    alpha=0.8,
                    color = fill(RGBA(1.,1.,1.,0.8),lenu,lenv),
                    legend = false)
    Plots.plot!(X,Y,Z,
                axis = true,
                linewidth = 1.5,
                color = palette(:default)[1],
                legend = false,
                xlabel = "x",
                ylabel = "y",
                zlabel = "z")
end

"""
    SpherePlot(X::SamplePath{T}, 𝕊::Sphere) where {T}

Plot a stochastic process of type `Bridge.SamplePath` on 𝕊.
"""
function SpherePlot(X::Bridge.SamplePath{T}, 𝕊::Sphere) where {T}
    X1 = extractcomp(X.yy, 1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    SpherePlot(X1,X2,X3, 𝕊)
end

"""
    SphereScatterPlot(X::T, Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}

Make a scatterplot of coordinates `X, Y, Z` on 𝕊.
"""
function SphereScatterPlot(X::T , Y::T, Z::T, 𝕊::Sphere) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    R = 𝕊.R
    du = 2π/100
    dv = π/100

    u = 0.0:du:(2π+du)
    v = 0.0:dv:(π+dv)

    lenu = length(u);
    lenv = length(v);
    x = zeros(lenu, lenv); y = zeros(lenu,lenv); z = zeros(lenu,lenv)
    for i in 1:lenu
        for j in 1:lenv
            x[i,j] = R*cos.(u[i]) * sin(v[j]);
            y[i,j] = R*sin.(u[i]) * sin(v[j]);
            z[i,j] = R*cos(v[j]);
        end
    end
    Plots.plot(X,Y,Z,
                axis = false,
                seriestype = :scatter,
                color= palette(:default)[1],
                markersize = 1,
                legend = false,
                label = false) #, xlabel = "x", ylabel = "y", zlabel = "z")
    Plots.surface!( x,y,z,
                axis=false,
                alpha=0.8,
                color = fill(RGBA(1.,1.,1.,0.8),lenu,lenv)) # fill(RGBA(1.,1.,1.,0.8),lenu,lenv))
end

"""
    SphereFullPlot(trace, data, target, 𝕊::Sphere; PlotUpdates = true)

A plot of a trace of (for example MCMC-) updates with data and a target added.
This function returns a plot on 𝕊 with a line `trace`, points `data` and a point `target`.
"""
function SphereFullPlot(trace, data, target, 𝕊::Sphere; PlotUpdates = true)
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    Target = Array{Float64}[]
    push!(Target, target)
    SpherePlot(extractcomp(trace,1), extractcomp(trace,2), extractcomp(trace,3), 𝕊)
    if PlotUpdates
        Plots.plot!(extractcomp(trace,1), extractcomp(trace,2), extractcomp(trace,3),
                    seriestype = :scatter,
                    color = :yellow,
                    markersize = 2,
                    label = "Updates")
    end
    Plots.plot!(extractcomp(data,1), extractcomp(data,2), extractcomp(data,3),
                seriestype = :scatter,
                color= :black,
                markersize = 1.5,
                label = "Data")
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2.5,
                label = "Target")
end

"""
    TorusPlot(X::T , Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}

Plot a function that takes values X, Y, Z on the `Torus` 𝕋.
"""
function TorusPlot(X::T, Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    n = 100
    ϑ = [0;2*(0.5:n-0.5)/n;2]
    φ = [0;2*(0.5:n-0.5)/n;2]
    x = [(𝕋.R+𝕋.r*cospi(φ))*cospi(ϑ) for ϑ in ϑ, φ in φ]
    y = [(𝕋.R+𝕋.r*cospi(φ))*sinpi(ϑ) for ϑ in ϑ, φ in φ]
    z = [𝕋.r*sinpi(φ) for ϑ in ϑ, φ in φ]

    lenϑ = length(ϑ)
    lenφ = length(φ)
    rng = 𝕋.R+𝕋.r
    # Set plots
    Plots.surface(x,y,z,
                    axis=true,
                    alpha=0.5,
                    legend = false,
                    color = :grey, #fill(RGBA(1.,1.,1.,0.8),lenu,lenv),
                    xlim = (-rng-1, rng+1),
                    ylim = (-rng-1, rng+1),
                    zlim = (-𝕋.r-1, 𝕋.r+1)
                    )
    Plots.plot!(X,Y,Z,
                    axis = true,
                    linewidth = 2.5,
                    color = palette(:default)[1],
                    legend = false,
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
end

"""
    TorusPlot(X::Bridge.SamplePath{T}, 𝕋::Torus) where {T}

Plot a stochastic process of type `Bridge.SamplePath` on 𝕋.
"""
function TorusPlot(X::SamplePath{T}, 𝕋::Torus) where {T}
    X1 = extractcomp(X.yy,1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    TorusPlot(X1, X2, X3, 𝕋)
end

"""
    TorusScatterPlot(X::T, Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}

Make a scatterplot of coordinates `X, Y, Z` on 𝕋.
"""
function TorusScatterPlot(X::T, Y::T, Z::T, 𝕋::Torus) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    rng = 𝕋.R+𝕋.r
    n = 100
    ϑ = [0;2*(0.5:n-0.5)/n;2]
    φ = [0;2*(0.5:n-0.5)/n;2]
    x = [(𝕋.R+𝕋.r*cospi(φ))*cospi(ϑ) for ϑ in ϑ, φ in φ]
    y = [(𝕋.R+𝕋.r*cospi(φ))*sinpi(ϑ) for ϑ in ϑ, φ in φ]
    z = [𝕋.r*sinpi(φ) for ϑ in ϑ, φ in φ]

    lenϑ = length(ϑ)
    lenφ = length(φ)
    Plots.surface(x,y,z,
                    axis=true,
                    alpha=0.5,
                    legend = false,
                    color = :grey,
                    xlim = (-rng-1, rng+1),
                    ylim = (-rng-1, rng+1),
                    zlim = (-rng-1, rng+1)
                    )
    Plots.plot!(X,Y,Z,
                    axis = true,
                    seriestype = :scatter,
                    color= palette(:default)[1],
                    markersize = 1,
                    legend = false,
                    label = false,
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
end

"""
    TorusFullPlot(trace, data, target, 𝕋::Torus; PlotUpdates = true)

A plot of a trace of (for example MCMC-) updates with data and a target added.
This function returns a plot on 𝕋 with a line `trace`, points `data` and a point `target`.
"""
function TorusFullPlot(trace, data, target, 𝕋; PlotUpdates = true)
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    Target = Array{Float64}[]
    push!(Target, target)
    TorusPlot(extractcomp(trace,1), extractcomp(trace,2), extractcomp(trace,3), 𝕋)
    if PlotUpdates
        Plots.plot!(extractcomp(trace,1), extractcomp(trace,2), extractcomp(trace,3),
                    seriestype = :scatter,
                    color = :yellow,
                    markersize = 2,
                    label = "Updates")
    end
    Plots.plot!(extractcomp(data,1), extractcomp(data,2), extractcomp(data,3),
                seriestype = :scatter,
                color= :black,
                markersize = 1.5,
                label = "Data")
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2.5,
                label = "Target")
end

"""
    ParaboloidPlot(X::T , Y::T, Z::T, ℙ::Paraboloid) where {T<:AbstractArray}

Plot a function that takes values X, Y, Z on the `Paraboloid` ℙ.
"""
function ParaboloidPlot(X::T,Y::T,Z::T, ℙ::Paraboloid) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end

    # Set grid
    n = 100
    rmax = 1.5*maximum(Z)
    dr = rmax/n
    r = 0:dr:rmax
    φ = [0;2*(0.5:n-0.5)/n;2]

    x = [sqrt(r)*ℙ.a*cospi(φ) for r in r, φ in φ]
    y = [sqrt(r)*ℙ.b*sinpi(φ) for r in r, φ in φ]
    z = [r for r in r, φ in φ]

    lenφ = length(φ);
    lenr = length(r);
    Plots.plot(X,Y,Z,
                    axis = true,
                    linewidth = 2.5,
                    color = palette(:default)[1],
                    legend = false,
                    xlims = (minimum([x y]), maximum([x y])),
                    ylims = (minimum([x y]), maximum([x y])),
                    label = "X",
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
    #cgrad([:yellow, :red])
    Plots.surface!( x,y,z,
                    axis=true,
                    alpha=0.8,
                    color = fill(RGBA(1.,1.,1.,0.8), lenr, lenφ),
                    legend = false)
end

"""
    ParaboloidPlot(X::Bridge.SamplePath{T}, ℙ::Paraboloid) where {T}

Plot a stochastic process of type `Bridge.SamplePath` on ℙ.
"""
function ParaboloidPlot(X::Bridge.SamplePath{T}, ℙ::Paraboloid) where {T}
    X1 = extractcomp(X.yy, 1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    ParaboloidPlot(X1,X2,X3, ℙ)
end

"""
    TorusScatterPlot(X::T, Y::T, Z::T, ℙ::Paraboloid) where {T<:AbstractArray}

Make a scatterplot of coordinates `X, Y, Z` on ℙ.
"""
function ParaboloidScatterPlot(X::T, Y::T, Z::T, ℙ::Paraboloid) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    n = 100
    dr = 10/n
    r = 0:dr:10
    φ = [0;2*(0.5:n-0.5)/n;2]

    x = [sqrt(r)*ℙ.a*cospi(φ) for r in r, φ in φ]
    y = [sqrt(r)*ℙ.b*sinpi(φ) for r in r, φ in φ]
    z = [r for r in r, φ in φ]

    lenφ = length(φ);
    lenr = length(r);

    Plots.plot(X,Y,Z,
                    axis = true,
                    seriestype = :scatter,
                    color= palette(:default)[1],
                    markersize = 1,
                    legend = false,
                    label = false,
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
   Plots.surface!( x,y,z,
                   axis=true,
                   alpha=0.8,
                   color = :grey,
                   legend = false)
end

"""
    ParaboloidFullPlot(trace, data, target, ℙ::Paraboloid; PlotUpdates = true)

A plot of a trace of (for example MCMC-) updates with data and a target added.
This function returns a plot on ℙ with a line `trace`, points `data` and a point `target`.
"""
function ParaboloidFullPlot(θ, data, target, ℙ::Paraboloid; PlotUpdates = true)
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    Target = Array{Float64}[]
    push!(Target, target)
    ParaboloidPlot(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3), ℙ)
    if PlotUpdates
        Plots.plot!(extractcomp(θ,1), extractcomp(θ,2), extractcomp(θ,3),
                    seriestype = :scatter,
                    color = :yellow,
                    markersize = 2,
                    label = "updates")
    end
    Plots.plot!(extractcomp(data,1), extractcomp(data,2), extractcomp(data,3),
                seriestype = :scatter,
                color= :black,
                markersize = 1.5,
                label = "data")
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2.5,
                label = "Target")
end
