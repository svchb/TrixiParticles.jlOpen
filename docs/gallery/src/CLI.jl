Base.@kwdef mutable struct CLIOptions
    command::Symbol = :help
    manifest_path::String = DEFAULT_MANIFEST_PATH
    ids::Vector{String} = String[]
    select_all::Bool = false
    dry_run::Bool = false
    workspace_root::Union{Nothing, String} = nothing
end

function print_usage(io::IO=stdout)
    println(io, "Usage:")
    println(io, "  julia docs/gallery/run_cluster_gallery.jl <command> [options]")
    println(io)
    println(io, "Commands:")
    println(io, "  validate       Load and validate the manifest")
    println(io, "  run-examples   Run gallery simulations into an external workspace")
    println(io, "  help           Show this help text")
    println(io)
    println(io, "Options:")
    println(io, "  --manifest PATH   Manifest path (default: docs/gallery/manifest.toml)")
    println(io, "  --id ID           Select one entry ID; may be repeated")
    println(io, "  --all             Select all manifest entries")
    println(io, "  --dry-run         Print the resolved execution plan without performing work")
    println(io, "  --workspace-root PATH")
    println(io, "                    External workspace root for raw outputs, logs, and metadata")
    println(io, "                    May also be set via $WORKSPACE_ENV")
end

function parse_args(args::Vector{String})
    isempty(args) && return CLIOptions()

    command = if args[1] in ("help", "-h", "--help")
        :help
    elseif args[1] == "validate"
        :validate
    elseif args[1] == "run-examples"
        :run_examples
    else
        fail("unknown command `$(args[1])`")
    end

    options = CLIOptions(command=command)
    index = 2
    while index <= length(args)
        arg = args[index]
        if arg == "--manifest"
            index < length(args) || fail("`--manifest` requires a path")
            options.manifest_path = abspath(args[index + 1])
            index += 2
        elseif arg == "--id"
            index < length(args) || fail("`--id` requires a value")
            push!(options.ids, args[index + 1])
            index += 2
        elseif arg == "--all"
            options.select_all = true
            index += 1
        elseif arg == "--dry-run"
            options.dry_run = true
            index += 1
        elseif arg == "--workspace-root"
            index < length(args) || fail("`--workspace-root` requires a path")
            options.workspace_root = abspath(args[index + 1])
            index += 2
        elseif arg in ("-h", "--help")
            return CLIOptions(command=:help)
        else
            fail("unknown option `$arg`")
        end
    end

    return options
end

function resolve_workspace_root(options::CLIOptions)
    root = nothing
    if !isnothing(options.workspace_root)
        root = abspath(options.workspace_root)
    elseif haskey(ENV, WORKSPACE_ENV) && !isempty(strip(ENV[WORKSPACE_ENV]))
        root = abspath(ENV[WORKSPACE_ENV])
    end

    isnothing(root) &&
        fail("`run-examples` requires `--workspace-root PATH` or the `$WORKSPACE_ENV` environment variable")

    path_is_inside(REPO_ROOT, root) &&
        fail("workspace root `$root` must be outside the repository `$REPO_ROOT`")

    return root
end

function build_run_contexts(entries::Vector{ManifestEntry}, options::CLIOptions)
    workspace_root = resolve_workspace_root(options)
    git = detect_git_provenance()
    manifest_hash_value = manifest_hash(options.manifest_path)

    return [RunContext(;
                       manifest_path=options.manifest_path,
                       manifest_hash=manifest_hash_value,
                       manifest_hash_short=shorten_hash(manifest_hash_value),
                       git,
                       entry,
                       layout=build_run_layout(entry, workspace_root, git, manifest_hash_value))
            for entry in entries]
end

function selected_entries(manifest::GalleryManifest, options::CLIOptions)
    options.select_all && !isempty(options.ids) &&
        fail("use either `--all` or `--id`, not both")

    entry_by_id = Dict(entry.id => entry for entry in manifest.entries)

    if options.command == :validate
        if options.select_all
            return manifest.entries
        elseif isempty(options.ids)
            return ManifestEntry[]
        end
    elseif !options.select_all && isempty(options.ids)
        fail("`run-examples` requires `--all` or at least one `--id`")
    end

    if options.select_all
        return manifest.entries
    end

    selected = ManifestEntry[]
    seen = Set{String}()
    for id in options.ids
        haskey(entry_by_id, id) || fail("unknown entry id `$id`")
        id in seen && fail("duplicate `--id $id`")
        push!(selected, entry_by_id[id])
        push!(seen, id)
    end
    return selected
end

function count_by(f, items)
    counts = Dict{String, Int}()
    for item in items
        key = f(item)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function print_counts(title, counts)
    println(title)
    for key in sort!(collect(keys(counts)))
        println("  ", key, ": ", counts[key])
    end
end

function print_entry_summary(manifest::GalleryManifest, entry::ManifestEntry)
    println("  id: ", entry.id)
    println("    title: ", entry.title)
    println("    example: ", entry.example)
    println("    category: ", entry_category(entry))
    println("    asset strategy: ", entry.asset_strategy)
    println("    outputs: ", join(entry_outputs(manifest, entry), ", "))
    println("    media path: ", media_repo_path(manifest, entry))
    if !isnothing(entry.run)
        keys_sorted = sort!(collect(keys(entry.run.overrides)))
        println("    run overrides: ", isempty(keys_sorted) ? "<none>" : join(keys_sorted, ", "))
        if !isnothing(entry.run.save_dt)
            println("    save dt: ", entry.run.save_dt)
        end
    end
end

function print_run_context(manifest::GalleryManifest, context::RunContext)
    print_entry_summary(manifest, context.entry)
    println("    git commit: ", context.git.commit_short, context.git.dirty ? " (dirty)" : "")
    println("    manifest hash: ", context.manifest_hash_short)
    println("    run key: ", context.layout.run_key)
    println("    workspace root: ", context.layout.workspace_root)
    println("    run root: ", context.layout.run_root)
    println("    raw dir: ", context.layout.raw_dir)
    println("    logs dir: ", context.layout.logs_dir)
    println("    metadata file: ", context.layout.metadata_file)
end

function print_validation_summary(manifest::GalleryManifest, manifest_path::AbstractString,
                                  entries::Vector{ManifestEntry})
    println("Manifest OK")
    println("  manifest: ", manifest_path)
    println("  schema_version: ", manifest.schema_version)
    println("  asset strategies: ", length(manifest.asset_strategies))
    println("  entries: ", length(manifest.entries))
    print_counts("  categories:", count_by(entry_category, manifest.entries))
    print_counts("  asset strategies:", count_by(entry -> entry.asset_strategy, manifest.entries))

    if !isempty(entries)
        println("  selected entries: ", length(entries))
        for entry in entries
            print_entry_summary(manifest, entry)
        end
    end
end

function print_run_plan(manifest::GalleryManifest, entries::Vector{ManifestEntry},
                        contexts::Vector{RunContext})
    println("Dry run plan")
    println("  stage: run-examples")
    println("  selected entries: ", length(entries))
    for context in contexts
        print_run_context(manifest, context)
        println("    metadata status: planned")
    end
end

function execute_command(command::Symbol, manifest::GalleryManifest, entries::Vector{ManifestEntry},
                         options::CLIOptions)
    if command == :validate
        print_validation_summary(manifest, options.manifest_path, entries)
        return 0
    end

    contexts = build_run_contexts(entries, options)
    print_run_plan(manifest, entries, contexts)
    options.dry_run && return 0

    command == :run_examples || fail("unsupported command `$command`")
    return run_examples!(manifest, contexts)
end

function main(args::Vector{String}=ARGS)
    try
        options = parse_args(args)
        if options.command == :help
            print_usage()
            return 0
        end

        log_info("loading manifest from `$(options.manifest_path)`")
        manifest = load_manifest(options.manifest_path)
        entries = selected_entries(manifest, options)

        if options.dry_run && options.command == :run_examples
            log_info("running in dry-run mode")
        end

        return execute_command(options.command, manifest, entries, options)
    catch err
        if err isa GalleryError
            println(stderr, "Error: ", err)
            return 1
        end
        rethrow()
    end
end
