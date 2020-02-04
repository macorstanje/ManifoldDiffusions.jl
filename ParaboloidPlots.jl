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

function ParaboloidPlot(X::SamplePath{T}, ℙ::Paraboloid) where {T}
    X1 = extractcomp(X.yy, 1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    ParaboloidPlot(X1,X2,X3, ℙ)
end

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

function ParaboloidScatterPlot(X::T, Y::T, Z::T, target::T, ℙ::Paraboloid) where {T<:AbstractArray}
    if Plots.backend() !== Plots.PlotlyBackend()
        error("Plotly() is not enabled")
    end
    ParaboloidScatterPlot(X,Y,Z,ℙ)
    Target = Array{Float64}[]
    push!(Target, target)
    Plots.plot!(extractcomp(Target,1), extractcomp(Target,2), extractcomp(Target,3),
                seriestype = :scatter,
                color= :red,
                markersize = 2)
end


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
