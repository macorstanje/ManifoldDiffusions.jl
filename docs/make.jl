using Documenter
Documenter.generate("Manifolds")

include("../src/Manifolds.jl")
using .Manifolds

# makedocs(
#     modules = [Manifolds],
#     sitename = "Manifolds.jl",
#     authors = "Marc Corstanje and contributors",
#     pages = Any[
#         "Home" => "index.md",
#         "Manual" => "manual.md",
#         "Library" => "library.md",
#     ],
# )

makedocs(
    modules = [Manifolds],
    sitename = "Manifolds.jl",
    authors = "Marc Corstanje and contributors",
    doctest = false,
    pages = Any[ # Compat: `Any` for 0.4 compat
    "Home" => "index.md",
    "Manual" => "manual.md",
    "Library" => Any[
        "Basics" => "library/basics.md",
        "Geodesics" => "library/geodesics.md",
        "Frames and the frame bundle" => "library/frames.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/macorstanje/Manifolds.jl.git",
    target = "build",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#", "dev" => "master"],
)
