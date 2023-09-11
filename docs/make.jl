using TCIAlgorithms
using Documenter

DocMeta.setdocmeta!(TCIAlgorithms, :DocTestSetup, :(using TCIAlgorithms); recursive=true)

makedocs(;
    modules=[TCIAlgorithms],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    repo="https://gitlab.com/tensors4fields/TCIAlgorithms.jl/blob/{commit}{path}#{line}",
    sitename="TCIAlgorithms.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tensors4fields.gitlab.io/TCIAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
