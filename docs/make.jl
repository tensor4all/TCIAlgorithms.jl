using TCIAlgorithms
using Documenter

DocMeta.setdocmeta!(TCIAlgorithms, :DocTestSetup, :(using TCIAlgorithms); recursive=true)

makedocs(;
    modules=[TCIAlgorithms],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="TCIAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/TCIAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/tensor4all/TCIAlgorithms.jl.git", devbranch="main")
