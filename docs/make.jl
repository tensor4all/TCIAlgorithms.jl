using TCIalgorithms
using Documenter

DocMeta.setdocmeta!(TCIalgorithms, :DocTestSetup, :(using TCIalgorithms); recursive=true)

makedocs(;
    modules=[TCIalgorithms],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    repo="https://gitlab.com/Ritter.Marc/TCIalgorithms.jl/blob/{commit}{path}#{line}",
    sitename="TCIalgorithms.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Ritter.Marc.gitlab.io/TCIalgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
