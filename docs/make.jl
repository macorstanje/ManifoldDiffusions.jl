using Documenter

include("../src/Manifolds.jl")
using .Manifolds

makedocs(
    modules = [Manifolds],
    sitename = "Manifolds.jl",
    authors = "Marc Corstanje and contributors",
    pages = Any[
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Library" => "library.md",
    ],
)

deploydocs(
    repo = "github.com/macorstanje/Manifolds.jl.git"
)
