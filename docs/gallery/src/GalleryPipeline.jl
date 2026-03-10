module GalleryPipeline

using Configurations
using Dates
using SHA
using TOML

export main, load_manifest, validate_manifest, write_metadata_file

const GALLERY_DIR = normpath(joinpath(@__DIR__, ".."))
const REPO_ROOT = normpath(joinpath(GALLERY_DIR, "..", ".."))
const EXAMPLES_DIR = joinpath(REPO_ROOT, "examples")
const DEFAULT_MANIFEST_PATH = joinpath(GALLERY_DIR, "manifest.toml")
const WORKSPACE_ENV = "TRIXIPARTICLES_GALLERY_WORKSPACE"

const ALLOWED_OUTPUTS = Set(["poster", "gif", "mp4"])
const ALLOWED_RENDER_BACKENDS = Set(["paraview", "julia"])
const ID_REGEX = r"^[a-z0-9_]+$"
const METADATA_SCHEMA_VERSION = 1
const ACTIVE_INCLUDE_SETTINGS = Ref{Any}(nothing)
const ACTIVE_INCLUDE_MODULE = Ref{Any}(nothing)

struct GalleryError <: Exception
    message::String
end

Base.showerror(io::IO, err::GalleryError) = print(io, err.message)

timestamp() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"

function log_message(level::AbstractString, message::AbstractString)
    println("[$(timestamp())] [$level] $message")
end

log_info(message::AbstractString) = log_message("info", message)

function fail(message::AbstractString)
    throw(GalleryError(String(message)))
end

walkexpr(f, expr::Expr) = f(Expr(expr.head, (walkexpr(f, arg) for arg in expr.args)...))
walkexpr(f, x) = f(x)

include("ManifestSupport.jl")
include("RunExamplesSupport.jl")
include("CLI.jl")

end
