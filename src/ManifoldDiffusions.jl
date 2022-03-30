module ManifoldDiffusions


    # Definitions
    export Manifold, EmbeddedManifold, TangentVector
    export Dimension, AmbientDimension
    export inChart, nCharts, inChartRange
    export Hamiltonian, f, ϕ, ϕ⁻¹,Dϕ, Dϕ⁻¹,P, g, g♯, Γ, Hamiltonian

    # Manifolds
    export Ellipse, Sphere, Torus, Dim4toDim3, Paraboloid
    # Frames
    export Frame, TangentFrame, Π, Πˣ, Hor, getx

    # Geodesics
    export Geodesic, ExponentialMap, ParallelTransport

    # FrameBundles
    export FrameBundle, Hamiltonian, Geodesic, ExponentialMap

    # StochasticDevelopment
    export Heun, IntegrateStep!, StochasticDevelopment!, StochasticDevelopment

    # ManifoldPlots
    export SpherePlot, SphereScatterPlot, SphereFullPlot
    export TorusPlot, TorusScatterPlot, TorusFullPlot
    export ParaboloidPlot, ParaboloidScatterPlot, ParaboloidFullPlot

    # using Bridge
    using Plots
    using ForwardDiff
    using LinearAlgebra
    using StaticArrays
    using Einsum
    using Bridge

    # Definitions of the manifolds additional properties
    include("Definitions.jl")

    # Properties of various manifolds
    include("Manifolds.jl")

    # Definitions and calculation rules for Frames
    include("Frames.jl")

    # Integrators and other functions for finding geodesics and the exponential map
    include("Geodesics.jl")

    # Describes Frame bundle
    # include("FrameBundles.jl")

    # Stocastic development
    include("StochasticDevelopment.jl")

    # Tools for creating plots on manifolds
    include("ManifoldPlots.jl")
end
