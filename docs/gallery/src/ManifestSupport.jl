@option struct RenderDefaults
    background::String = "white"
    color_scheme::String = "viridis"
end

@option struct PathConventions
    derive_category_from_example::Bool = true
    derive_outputs_from_asset_strategy::Bool = true
    media_repo_path_pattern::String = "{category}/{id}"
end

@option struct AssetStrategy
    render_backend::String
    outputs::Vector{String}
end

@option struct EntryRun
    overrides::Dict{String, Any} = Dict{String, Any}()
    save_dt::Union{Nothing, Float64} = nothing
end

@option struct ManifestEntry
    id::String
    title::String
    example::String
    asset_strategy::String
    caption::Union{Nothing, String} = nothing
    outputs::Union{Nothing, Vector{String}} = nothing
    run::Union{Nothing, EntryRun} = nothing
    render::Union{Nothing, Dict{String, Any}} = nothing
    notes::Union{Nothing, String} = nothing
end

@option struct GalleryManifest
    schema_version::Int = 1
    render_defaults::RenderDefaults = RenderDefaults()
    path_conventions::PathConventions = PathConventions()
    asset_strategies::Dict{String, AssetStrategy}
    entries::Vector{ManifestEntry}
end

function Configurations.from_dict(::Type{OptionType}, ::Type{Dict{String, AssetStrategy}}, x) where {OptionType}
    x isa AbstractDict || error("expected an AbstractDict for asset_strategies, got $(typeof(x))")
    return Dict(String(key) => Configurations.from_dict(AssetStrategy, value)
                for (key, value) in pairs(x))
end

function load_manifest(path::AbstractString=DEFAULT_MANIFEST_PATH)
    isfile(path) || fail("manifest file not found at `$path`")
    parsed = TOML.parsefile(path)
    manifest = Configurations.from_dict(GalleryManifest, parsed)
    validate_manifest(manifest; manifest_path=path)
    return manifest
end

entry_category(entry::ManifestEntry) = first(splitpath(entry.example))
entry_example_path(entry::ManifestEntry) = joinpath(EXAMPLES_DIR, splitpath(entry.example)...)

function entry_outputs(manifest::GalleryManifest, entry::ManifestEntry)
    if isnothing(entry.outputs)
        return manifest.asset_strategies[entry.asset_strategy].outputs
    end
    return entry.outputs
end

function entry_save_dt(entry::ManifestEntry)
    isnothing(entry.run) && return nothing
    return entry.run.save_dt
end

function normalize_override_value(key::AbstractString, value)
    if key == "tspan" && value isa Vector
        return tuple((normalize_override_value("", item) for item in value)...)
    elseif value isa Vector
        return [normalize_override_value("", item) for item in value]
    end

    return value
end

function entry_run_kwargs(entry::ManifestEntry)
    isnothing(entry.run) && return Pair{Symbol, Any}[]

    pairs_sorted = sort!(collect(entry.run.overrides); by=first)
    return Pair{Symbol, Any}[Symbol(key) => normalize_override_value(key, value)
                             for (key, value) in pairs_sorted]
end

function media_repo_path(manifest::GalleryManifest, entry::ManifestEntry)
    replace(manifest.path_conventions.media_repo_path_pattern,
            "{category}" => entry_category(entry),
            "{id}" => entry.id)
end

function validate_nonempty_string(value, context, errors)
    value isa String || push!(errors, "$context must be a string")
    value isa String && isempty(strip(value)) && push!(errors, "$context must not be empty")
end

function validate_outputs(outputs::Vector{String}, context, errors)
    isempty(outputs) && push!(errors, "$context must not be empty")
    length(unique(outputs)) == length(outputs) || push!(errors, "$context must not contain duplicates")
    "poster" in outputs || push!(errors, "$context must include `poster`")

    for output in outputs
        output in ALLOWED_OUTPUTS || push!(errors, "$context contains unsupported output `$output`")
    end
end

function validate_kwarg_value(value, context, errors)
    if value isa String || value isa Integer || value isa AbstractFloat || value isa Bool
        return
    elseif value isa Vector
        isempty(value) && push!(errors, "$context must not contain empty arrays")
        for (index, item) in pairs(value)
            validate_kwarg_value(item, "$context[$index]", errors)
        end
        return
    end

    push!(errors, "$context uses unsupported TOML value type `$(typeof(value))`")
end

function validate_manifest(manifest::GalleryManifest; manifest_path::AbstractString=DEFAULT_MANIFEST_PATH)
    errors = String[]

    manifest.schema_version == 1 || push!(errors, "schema_version must be 1")
    manifest.render_defaults.background == "white" ||
        push!(errors, "render_defaults.background must be `white`")
    manifest.render_defaults.color_scheme == "viridis" ||
        push!(errors, "render_defaults.color_scheme must be `viridis`")

    manifest.path_conventions.derive_category_from_example ||
        push!(errors, "path_conventions.derive_category_from_example must be true")
    manifest.path_conventions.derive_outputs_from_asset_strategy ||
        push!(errors, "path_conventions.derive_outputs_from_asset_strategy must be true")
    manifest.path_conventions.media_repo_path_pattern == "{category}/{id}" ||
        push!(errors, "path_conventions.media_repo_path_pattern must be `{category}/{id}`")

    isempty(manifest.asset_strategies) &&
        push!(errors, "asset_strategies must not be empty")

    for (name, strategy) in sort!(collect(manifest.asset_strategies); by=first)
        validate_nonempty_string(name, "asset strategy name", errors)
        validate_nonempty_string(strategy.render_backend,
                                 "asset strategy `$name`.render_backend",
                                 errors)
        strategy.render_backend in ALLOWED_RENDER_BACKENDS ||
            push!(errors, "asset strategy `$name` uses unsupported render backend `$(strategy.render_backend)`")
        validate_outputs(strategy.outputs, "asset strategy `$name`.outputs", errors)
    end

    isempty(manifest.entries) && push!(errors, "entries must not be empty")

    seen_ids = Set{String}()
    seen_examples = Set{String}()
    seen_media_paths = Set{String}()

    for entry in manifest.entries
        validate_nonempty_string(entry.id, "entry.id", errors)
        occursin(ID_REGEX, entry.id) || push!(errors, "entry.id `$(entry.id)` must match `$(ID_REGEX.pattern)`")
        entry.id in seen_ids && push!(errors, "duplicate entry.id `$(entry.id)`")
        push!(seen_ids, entry.id)

        validate_nonempty_string(entry.title, "entry `$(entry.id)`.title", errors)
        validate_nonempty_string(entry.example, "entry `$(entry.id)`.example", errors)

        startswith(entry.example, "/") &&
            push!(errors, "entry `$(entry.id)`.example must be relative")
        occursin("..", entry.example) &&
            push!(errors, "entry `$(entry.id)`.example must not contain `..`")
        endswith(entry.example, ".jl") ||
            push!(errors, "entry `$(entry.id)`.example must end in `.jl`")
        entry.example in seen_examples &&
            push!(errors, "duplicate entry.example `$(entry.example)`")
        push!(seen_examples, entry.example)

        example_path = entry_example_path(entry)
        isfile(example_path) ||
            push!(errors, "entry `$(entry.id)`.example does not exist at `$example_path`")

        haskey(manifest.asset_strategies, entry.asset_strategy) ||
            push!(errors, "entry `$(entry.id)` references unknown asset strategy `$(entry.asset_strategy)`")

        path = media_repo_path(manifest, entry)
        path in seen_media_paths &&
            push!(errors, "derived media_repo_path `$path` is not unique")
        push!(seen_media_paths, path)

        validate_outputs(entry_outputs(manifest, entry), "entry `$(entry.id)`.outputs", errors)

        if !isnothing(entry.caption)
            validate_nonempty_string(entry.caption, "entry `$(entry.id)`.caption", errors)
        end

        if !isnothing(entry.notes)
            validate_nonempty_string(entry.notes, "entry `$(entry.id)`.notes", errors)
        end

        if !isnothing(entry.run)
            isempty(entry.run.overrides) &&
                isnothing(entry.run.save_dt) &&
                push!(errors, "entry `$(entry.id)`.run must not be empty")
            for (key, value) in sort!(collect(entry.run.overrides); by=first)
                validate_nonempty_string(key, "entry `$(entry.id)`.run.overrides key", errors)
                validate_kwarg_value(value, "entry `$(entry.id)`.run.overrides.$key", errors)
            end
            if !isnothing(entry.run.save_dt)
                entry.run.save_dt > 0 ||
                    push!(errors, "entry `$(entry.id)`.run.save_dt must be positive")
            end
        end

        if !isnothing(entry.render)
            isempty(entry.render) &&
                push!(errors, "entry `$(entry.id)`.render must not be empty")
        end
    end

    isempty(errors) || fail(join(errors, "\n"))
    return manifest
end
