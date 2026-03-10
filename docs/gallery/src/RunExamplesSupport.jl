Base.@kwdef struct GitProvenance
    commit::String
    commit_short::String
    dirty::Bool
end

Base.@kwdef struct RunLayout
    workspace_root::String
    run_key::String
    run_root::String
    raw_dir::String
    frames_dir::String
    encoded_dir::String
    logs_dir::String
    metadata_dir::String
    log_file::String
    metadata_file::String
end

Base.@kwdef struct RunContext
    manifest_path::String
    manifest_hash::String
    manifest_hash_short::String
    git::GitProvenance
    entry::ManifestEntry
    layout::RunLayout
end

Base.@kwdef struct GalleryIncludeSettings
    raw_output_directory::String
    save_dt::Union{Nothing, Float64} = nothing
end

Base.@kwdef struct RunResult
    entry_id::String
    status::String
    metadata_file::String
    log_file::String
    raw_outputs::Vector{String} = String[]
    solver_retcode::Union{Nothing, String} = nothing
    error_message::Union{Nothing, String} = nothing
end

function path_is_inside(parent::AbstractString, child::AbstractString)
    parent_path = normpath(abspath(parent))
    child_path = normpath(abspath(child))
    parent_components = splitpath(parent_path)
    child_components = splitpath(child_path)
    length(child_components) < length(parent_components) && return false
    return child_components[1:length(parent_components)] == parent_components
end

function git_output(args::AbstractString...)
    cmd = Cmd(["git", "-C", REPO_ROOT, args...])
    try
        return strip(readchomp(cmd))
    catch err
        fail("failed to query git metadata with `$(join(["git", "-C", REPO_ROOT, args...], " "))`: $err")
    end
end

function detect_git_provenance()
    commit = git_output("rev-parse", "HEAD")
    commit_short = git_output("rev-parse", "--short=12", "HEAD")
    dirty = !isempty(git_output("status", "--porcelain"))
    return GitProvenance(; commit, commit_short, dirty)
end

manifest_hash(path::AbstractString) = bytes2hex(sha256(read(path)))

function shorten_hash(hash::AbstractString; n::Int=12)
    last = min(n, lastindex(hash))
    return hash[firstindex(hash):last]
end

function ensure_runtime_environment!()
    normalized_load_path = normpath(REPO_ROOT)
    normalized_load_path in normpath.(LOAD_PATH) || pushfirst!(LOAD_PATH, normalized_load_path)

    if !isdefined(@__MODULE__, :TrixiParticles)
        @eval using TrixiParticles
    end
    if !isdefined(@__MODULE__, :OrdinaryDiffEq)
        @eval using OrdinaryDiffEq
    end
    if !isdefined(@__MODULE__, :SciMLBase)
        @eval using SciMLBase
    end

    return nothing
end

function with_gallery_include_settings(f::Function, settings::GalleryIncludeSettings)
    previous = ACTIVE_INCLUDE_SETTINGS[]
    ACTIVE_INCLUDE_SETTINGS[] = settings
    try
        return f()
    finally
        ACTIVE_INCLUDE_SETTINGS[] = previous
    end
end

function active_gallery_include_settings()
    settings = ACTIVE_INCLUDE_SETTINGS[]
    settings isa GalleryIncludeSettings ||
        fail("internal error: gallery include settings are not active")
    return settings
end

function with_gallery_include_module(f::Function, mod::Module)
    previous = ACTIVE_INCLUDE_MODULE[]
    ACTIVE_INCLUDE_MODULE[] = mod
    try
        return f()
    finally
        ACTIVE_INCLUDE_MODULE[] = previous
    end
end

function active_gallery_include_module()
    mod = ACTIVE_INCLUDE_MODULE[]
    mod isa Module || fail("internal error: active include module is not set")
    return mod
end

is_dot_reference(expr::Expr, name::Symbol) =
    expr.head === :. && length(expr.args) == 2 &&
    expr.args[2] isa QuoteNode && expr.args[2].value === name

function is_solution_saving_call(expr)
    expr isa Expr || return false
    expr.head === :call || return false
    callee = expr.args[1]
    return callee === :SolutionSavingCallback ||
           (callee isa Expr && is_dot_reference(callee, :SolutionSavingCallback))
end

function set_call_keyword!(call::Expr, key::Symbol, value)
    for arg in call.args[2:end]
        if arg isa Expr && arg.head === :kw && arg.args[1] === key
            arg.args[2] = value
            return call
        elseif arg isa Expr && arg.head === :parameters
            for nested_arg in arg.args
                if nested_arg isa Expr && nested_arg.head === :kw && nested_arg.args[1] === key
                    nested_arg.args[2] = value
                    return call
                end
            end
        end
    end

    push!(call.args, Expr(:kw, key, value))
    return call
end

function patch_solution_saving_call(expr::Expr, settings::GalleryIncludeSettings)
    is_solution_saving_call(expr) || return expr

    set_call_keyword!(expr, :output_directory, settings.raw_output_directory)
    set_call_keyword!(expr, :append_timestamp, false)

    if !isnothing(settings.save_dt)
        set_call_keyword!(expr, :interval, 0)
        set_call_keyword!(expr, :save_times, :(Float64[]))
        set_call_keyword!(expr, :dt, settings.save_dt)
    end

    return expr
end

function patch_nested_trixi_include(expr::Expr)
    is_trixi_include = expr.head === :call && !isempty(expr.args) &&
                       (expr.args[1] === :trixi_include ||
                        expr.args[1] === :trixi_include_changeprecision)
    is_trixi_include || return expr

    expr.args[1] = :(Main.GalleryPipeline.gallery_trixi_include)
    return expr
end

function gallery_mapexpr(expr, settings::GalleryIncludeSettings)
    return walkexpr(expr) do node
        node isa Expr || return node
        node = patch_nested_trixi_include(node)
        node = patch_solution_saving_call(node, settings)
        return node
    end
end

function gallery_trixi_include(mod::Module, elixir::AbstractString; kwargs...)
    ensure_runtime_environment!()
    settings = active_gallery_include_settings()
    return with_gallery_include_module(mod) do
        Base.invokelatest(TrixiParticles.trixi_include,
                          expr -> gallery_mapexpr(expr, settings),
                          mod, elixir;
                          replace_assignments_recursive=true,
                          kwargs...)
    end
end

function gallery_trixi_include(elixir::AbstractString; kwargs...)
    return gallery_trixi_include(active_gallery_include_module(), elixir; kwargs...)
end

function build_run_layout(entry::ManifestEntry, workspace_root::AbstractString,
                          git::GitProvenance, manifest_hash_value::AbstractString)
    manifest_hash_short = shorten_hash(manifest_hash_value)
    run_key = join(["git-" * git.commit_short, "manifest-" * manifest_hash_short], "_")
    run_root = joinpath(workspace_root, "runs", entry.id,
                        "git-" * git.commit_short,
                        "manifest-" * manifest_hash_short)

    return RunLayout(;
                     workspace_root,
                     run_key,
                     run_root,
                     raw_dir=joinpath(run_root, "raw"),
                     frames_dir=joinpath(run_root, "frames"),
                     encoded_dir=joinpath(run_root, "encoded"),
                     logs_dir=joinpath(run_root, "logs"),
                     metadata_dir=joinpath(run_root, "metadata"),
                     log_file=joinpath(run_root, "logs", "run-examples.log"),
                     metadata_file=joinpath(run_root, "metadata", "run-examples.toml"))
end

function metadata_tool_versions()
    return Dict(
        "julia" => string(VERSION),
        "gallery_pipeline" => "phase5",
    )
end

function metadata_dict(manifest::GalleryManifest, context::RunContext;
                       status::AbstractString,
                       started_at::Union{Nothing, String}=nothing,
                       finished_at::Union{Nothing, String}=nothing,
                       job_id::Union{Nothing, String}=nothing)
    entry = context.entry
    strategy = manifest.asset_strategies[entry.asset_strategy]

    metadata = Dict{String, Any}(
        "schema_version" => METADATA_SCHEMA_VERSION,
        "status" => String(status),
        "stage" => "run-examples",
        "generated_at" => timestamp(),
        "entry" => Dict(
            "id" => entry.id,
            "title" => entry.title,
            "example" => entry.example,
            "category" => entry_category(entry),
            "asset_strategy" => entry.asset_strategy,
            "outputs" => collect(entry_outputs(manifest, entry)),
            "media_repo_path" => media_repo_path(manifest, entry),
        ),
        "provenance" => Dict(
            "git_commit" => context.git.commit,
            "git_commit_short" => context.git.commit_short,
            "git_dirty" => context.git.dirty,
            "manifest_path" => context.manifest_path,
            "manifest_hash" => context.manifest_hash,
            "manifest_hash_short" => context.manifest_hash_short,
        ),
        "manifest_values" => Dict(
            "entry" => Configurations.to_dict(entry; include_defaults=true, exclude_nothing=true),
            "asset_strategy" => Configurations.to_dict(strategy; include_defaults=true),
            "render_defaults" => Configurations.to_dict(manifest.render_defaults; include_defaults=true),
            "path_conventions" => Configurations.to_dict(manifest.path_conventions; include_defaults=true),
        ),
        "run" => Dict(
            "mode" => "direct_julia",
            "rerun_policy" => "overwrite",
            "run_key" => context.layout.run_key,
            "workspace_root" => context.layout.workspace_root,
            "run_root" => context.layout.run_root,
        ),
        "output_paths" => Dict(
            "raw" => context.layout.raw_dir,
            "frames" => context.layout.frames_dir,
            "encoded" => context.layout.encoded_dir,
            "logs" => context.layout.logs_dir,
            "metadata" => context.layout.metadata_dir,
            "log_file" => context.layout.log_file,
            "metadata_file" => context.layout.metadata_file,
        ),
        "tools" => metadata_tool_versions(),
    )

    if started_at !== nothing || finished_at !== nothing
        metadata["timestamps"] = Dict{String, Any}()
        started_at !== nothing && (metadata["timestamps"]["started_at"] = started_at)
        finished_at !== nothing && (metadata["timestamps"]["finished_at"] = finished_at)
    end

    if job_id !== nothing
        metadata["job"] = Dict("id" => job_id)
    end

    return metadata
end

function write_metadata_file(path::AbstractString, metadata::Dict{String, Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, metadata)
    end
    return path
end

function update_metadata_file!(manifest::GalleryManifest, context::RunContext;
                               status::AbstractString,
                               started_at::Union{Nothing, String}=nothing,
                               finished_at::Union{Nothing, String}=nothing,
                               job_id::Union{Nothing, String}=nothing,
                               solver_retcode::Union{Nothing, String}=nothing,
                               raw_outputs::Vector{String}=String[],
                               failure::Union{Nothing, Dict{String, Any}}=nothing)
    metadata = metadata_dict(manifest, context; status, started_at, finished_at, job_id)

    if !isempty(raw_outputs)
        metadata["artifacts"] = Dict(
            "raw_output_count" => count_raw_artifacts(raw_outputs),
            "files" => raw_outputs,
        )
    end
    if solver_retcode !== nothing
        metadata["solver"] = Dict("retcode" => solver_retcode)
    end
    if failure !== nothing
        metadata["failure"] = failure
    end

    return write_metadata_file(context.layout.metadata_file, metadata)
end

function prepare_run_directories!(layout::RunLayout)
    for directory in (layout.raw_dir, layout.frames_dir, layout.encoded_dir,
                      layout.logs_dir, layout.metadata_dir)
        mkpath(directory)
    end
    return layout
end

function current_job_id()
    for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "PBS_JOBID")
        if haskey(ENV, key) && !isempty(strip(ENV[key]))
            return ENV[key]
        end
    end
    return nothing
end

function collect_output_files(root::AbstractString)
    isdir(root) || return String[]

    collected = String[]
    for (directory, _, files) in walkdir(root)
        for file in files
            path = joinpath(directory, file)
            push!(collected, replace(relpath(path, root), '\\' => '/'))
        end
    end

    sort!(collected)
    return collected
end

function count_raw_artifacts(paths::Vector{String})
    return count(path -> endswith(path, ".pvd") || endswith(path, ".vtu") ||
                         endswith(path, ".pvtu") || endswith(path, ".vtm"),
                 paths)
end

function module_name_for_entry(entry::ManifestEntry, context::RunContext)
    return gensym(Symbol("GalleryRun_", entry.id, "_", context.manifest_hash_short, "_"))
end

make_run_module(entry::ManifestEntry, context::RunContext) = Module(module_name_for_entry(entry, context))

function solver_retcode_text(mod::Module)
    isdefined(mod, :sol) || return nothing
    sol = getfield(mod, :sol)
    hasproperty(sol, :retcode) || return nothing
    return Base.invokelatest(string, getproperty(sol, :retcode))
end

function validate_solver_result!(mod::Module, context::RunContext, strategy::AssetStrategy)
    retcode = solver_retcode_text(mod)
    if retcode === nothing
        strategy.render_backend == "paraview" &&
            fail("entry `$(context.entry.id)` did not define `sol`; cannot confirm simulation success")
        return nothing
    end

    Base.invokelatest(SciMLBase.successful_retcode, getfield(mod, :sol)) ||
        fail("entry `$(context.entry.id)` finished with solver retcode `$retcode`")

    return retcode
end

function run_entry_example!(manifest::GalleryManifest, context::RunContext)
    ensure_runtime_environment!()
    prepare_run_directories!(context.layout)

    started_at = timestamp()
    job_id = current_job_id()
    update_metadata_file!(manifest, context; status="running", started_at, job_id)

    example_path = entry_example_path(context.entry)
    kwargs = entry_run_kwargs(context.entry)
    settings = GalleryIncludeSettings(; raw_output_directory=context.layout.raw_dir,
                                      save_dt=entry_save_dt(context.entry))
    strategy = manifest.asset_strategies[context.entry.asset_strategy]

    solver_retcode = nothing
    raw_outputs = String[]

    try
        open(context.layout.log_file, "w") do log_io
            println(log_io, "[gallery] entry: ", context.entry.id)
            println(log_io, "[gallery] example: ", example_path)
            println(log_io, "[gallery] workspace: ", context.layout.run_root)
            flush(log_io)

            redirect_stdio(stdout=log_io, stderr=log_io) do
                with_gallery_include_settings(settings) do
                    mod = make_run_module(context.entry, context)
                    with_gallery_include_module(mod) do
                        Base.invokelatest(TrixiParticles.trixi_include,
                                          expr -> gallery_mapexpr(expr, settings),
                                          mod, example_path;
                                          replace_assignments_recursive=true,
                                          kwargs...)
                        solver_retcode = validate_solver_result!(mod, context, strategy)
                    end
                end
            end
        end

        raw_outputs = collect_output_files(context.layout.raw_dir)
        if strategy.render_backend == "paraview" && count_raw_artifacts(raw_outputs) == 0
            fail("entry `$(context.entry.id)` did not produce any raw VTK output in `$(context.layout.raw_dir)`")
        end

        update_metadata_file!(manifest, context;
                              status="completed",
                              started_at,
                              finished_at=timestamp(),
                              job_id,
                              solver_retcode,
                              raw_outputs)

        return RunResult(; entry_id=context.entry.id,
                         status="completed",
                         metadata_file=context.layout.metadata_file,
                         log_file=context.layout.log_file,
                         raw_outputs,
                         solver_retcode)
    catch err
        failure = Dict{String, Any}(
            "type" => string(typeof(err)),
            "message" => sprint(showerror, err),
            "backtrace" => sprint(showerror, err, catch_backtrace()),
        )

        raw_outputs = collect_output_files(context.layout.raw_dir)
        update_metadata_file!(manifest, context;
                              status="failed",
                              started_at,
                              finished_at=timestamp(),
                              job_id,
                              solver_retcode,
                              raw_outputs,
                              failure)

        return RunResult(; entry_id=context.entry.id,
                         status="failed",
                         metadata_file=context.layout.metadata_file,
                         log_file=context.layout.log_file,
                         raw_outputs,
                         solver_retcode,
                         error_message=failure["message"])
    end
end

function print_run_results(results::Vector{RunResult})
    println("Run results")
    print_counts("  statuses:", count_by(result -> result.status, results))

    for result in results
        println("  entry: ", result.entry_id)
        println("    status: ", result.status)
        result.solver_retcode !== nothing &&
            println("    solver retcode: ", result.solver_retcode)
        println("    raw outputs: ", length(result.raw_outputs))
        println("    log file: ", result.log_file)
        println("    metadata file: ", result.metadata_file)
        result.error_message !== nothing &&
            println("    error: ", result.error_message)
    end
end

function run_examples!(manifest::GalleryManifest, contexts::Vector{RunContext})
    results = RunResult[]
    for context in contexts
        log_info("running entry `$(context.entry.id)`")
        push!(results, run_entry_example!(manifest, context))
    end

    print_run_results(results)

    failed = [result.entry_id for result in results if result.status != "completed"]
    isempty(failed) || fail("run-examples failed for: " * join(failed, ", "))
    return 0
end
