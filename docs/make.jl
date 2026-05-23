using Documenter
using DocumenterCitations
using HiddenMarkovModels
using Literate

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

open(joinpath(joinpath(@__DIR__, "src"), "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/JuliaStats/HiddenMarkovModels.jl/blob/main/README.md"
        ```
        """,
    )
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

examples_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end

for file in readdir(examples_path)
    Literate.markdown(joinpath(examples_path, file), examples_md_path)
end

function literate_title(path)
    l = first(readlines(path))
    return l[3:end]
end

pages = [
    "Home" => "index.md",
    "Tutorials" => [
        joinpath("examples", "basics.md"),
        joinpath("examples", "types.md"),
        joinpath("examples", "interfaces.md"),
        joinpath("examples", "temporal.md"),
        joinpath("examples", "controlled.md"),
        joinpath("examples", "autoregression.md"),
        joinpath("examples", "autodiff.md"),
        joinpath("examples", "gradientdescent.md"),
        joinpath("examples", "hsmm.md"),
    ],
    "API reference" => "api.md",
    "Advanced" => ["alternatives.md", "debugging.md", "formulas.md"],
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    repolink="https://github.com/JuliaStats/HiddenMarkovModels.jl",
    canonical="https://JuliaStats.github.io/HiddenMarkovModels.jl",
    assets=String[],
)

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle",
    sitename="HiddenMarkovModels.jl",
    format=fmt,
    pages=pages,
    plugins=[bib],
    pagesonly=true,
)

deploydocs(; repo="github.com/JuliaStats/HiddenMarkovModels.jl", devbranch="main")
