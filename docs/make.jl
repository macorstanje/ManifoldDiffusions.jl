using Documenter

include("../src/Manifolds.jl")
using .Manifolds

makedocs(
    modules = [Manifolds],
    sitename = "Manifolds.jl",
    authors = "Marc Corstanje and contributors"
)

deploydocs(
    repo = "https://github.com/macorstanje/Manifolds.jl.git"
)
