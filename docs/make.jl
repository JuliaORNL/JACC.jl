
push!(LOAD_PATH, "..")

using Documenter, JACC

makedocs(; sitename = "JACC.jl",
    authors = "William F. Godoy, Philip Fackler, Pedro Valero-Lara",
    clean = true,
    pagesonly = true,
    warnonly = [:missing_docs],
    modules = [JACC],
    pages = [
        "Welcome" => "index.md",
        "API Usage" => "api_usage.md",
        "Miscellaneous" => "miscellaneous.md"
    ],
    format = Documenter.HTML(
        ; prettyurls = true)
)

deploydocs(; repo = "github.com/JuliaGPU/JACC.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main"
)
