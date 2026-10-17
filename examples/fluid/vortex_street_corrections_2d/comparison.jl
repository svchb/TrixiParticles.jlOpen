# Run from the repository root with the environment in this directory; see README.md.
include("simulation.jl")
using .VortexStreetCorrections
using CairoMakie
using CSV
using DataFrames
using JSON
using Printf
using Statistics
using TrixiParticles: ParallelUpdate, SerialUpdate

function comparison_plots(output_directory)
    cases = VortexStreetCorrections.CASES
    labels = VortexStreetCorrections.LABELS
    metadata = [JSON.parsefile(joinpath(output_directory, string(case), "metadata.json"))
                for case in cases]
    cfg = metadata[1]["configuration"]
    for item in metadata
        item["configuration"] == cfg || error("case configurations do not match")
        item["final_time"] == cfg["end_time"] || error("incomplete case")
    end
    fields = [CSV.read(joinpath(output_directory, string(case), "field.csv"), DataFrame)
              for case in cases]
    traces = [CSV.read(joinpath(output_directory, string(case), "diagnostics.csv"),
                       DataFrame) for case in cases]
    nx, ny = metadata[1]["grid_size"]
    diameter = cfg["diameter"]
    xs = fields[1].x[1:nx] / diameter
    ys = fields[1].y[1:nx:end] / diameter .- cfg["center"][2] / diameter
    omega = [reshape(field.vorticity, nx, ny) for field in fields]
    finite_omega = filter(isfinite, reduce(vcat, vec.(omega)))
    # Use a shared robust scale so isolated near-wall peaks do not hide the wake.
    color_limit = max(1.0, quantile(abs.(finite_omega), 0.98))
    circle_x = cfg["center"][1] / diameter .+ 0.5cos.(range(0, 2pi; length=100))
    circle_y = 0.5sin.(range(0, 2pi; length=100))
    fig = Figure(; size=(1400, 850))
    plot_object = nothing
    for (i, case) in enumerate(cases)
        row, col = fldmod1(i, 2)
        ax = Axis(fig[row, col]; title=labels[case], xlabel="x / D",
                  ylabel="(y - y_c) / D",
                  aspect=DataAspect())
        plot_object = heatmap!(ax, xs, ys, omega[i]; colormap=:balance,
                               colorrange=(-color_limit, color_limit))
        poly!(ax, Point2f.(circle_x, circle_y); color=:gray30)
        xlims!(ax, 1.0, last(xs) - 0.5)
        ylims!(ax, -2.5, 2.5)
    end
    Colorbar(fig[1:2, 3], plot_object; label="Vorticity ω D / U")
    Label(fig[0, 1:3],
          "Re = $(Int(cfg["reynolds"])), t U / D = " *
          "$(round(cfg["end_time"] * cfg["velocity"] / diameter; digits=1))";
          fontsize=22)
    save(joinpath(output_directory, "vorticity_comparison.png"), fig)

    difference = [omega[i] .- omega[1] for i in 2:4]
    finite_difference = filter(isfinite, reduce(vcat, vec.(difference)))
    difference_limit = max(0.1, quantile(abs.(finite_difference), 0.98))
    diff_fig = Figure(; size=(1400, 1000))
    for (i, case) in enumerate(cases[2:4])
        ax = Axis(diff_fig[i, 1]; title="$(labels[case]) minus uncorrected",
                  xlabel="x / D", ylabel="(y - y_c) / D", aspect=DataAspect())
        plot_object = heatmap!(ax, xs, ys, difference[i]; colormap=:balance,
                               colorrange=(-difference_limit, difference_limit))
        poly!(ax, Point2f.(circle_x, circle_y); color=:gray30)
        xlims!(ax, 1.0, last(xs) - 0.5)
        ylims!(ax, -1.5, 1.5)
    end
    Colorbar(diff_fig[1:3, 2], plot_object;
             label="Vorticity difference, same physical time")
    save(joinpath(output_directory, "vorticity_differences.png"), diff_fig)

    history_fig = Figure(; size=(1400, 900))
    panels = ((:lift_coefficient, "Cylinder lift", "C_L"),
              (:drag_coefficient, "Cylinder drag", "C_D"),
              (:density_rms_percent, "Density variation", "RMS (rho/rho0 - 1) [%]"),
              (:probe_v, "Transverse wake velocity", "v / U"))
    colors = Makie.wong_colors()
    for (index, (column, title, ylabel)) in enumerate(panels)
        row, col = fldmod1(index, 2)
        ax = Axis(history_fig[row, col]; title, xlabel="t U / D", ylabel)
        for (i, case) in enumerate(cases)
            lines!(ax, traces[i].advective_time, traces[i][!, column];
                   color=colors[i], linewidth=2, label=labels[case])
        end
        index == 1 && axislegend(ax; position=:rt, labelsize=12)
    end
    save(joinpath(output_directory, "diagnostics_comparison.png"), history_fig)

    summary = DataFrame(case=String[], runtime_seconds=Float64[],
                        accepted_steps=Int[], mean_drag=Float64[], lift_rms=Float64[],
                        density_rms_percent=Float64[], vorticity_difference_rms=Float64[])
    for (i, case) in enumerate(cases)
        late = traces[i].time .>= cfg["end_time"] / 2
        delta = omega[i] .- omega[1]
        values = filter(isfinite, vec(delta))
        push!(summary,
              (string(case), metadata[i]["total_seconds"],
               metadata[i]["accepted_steps"],
               mean(traces[i].drag_coefficient[late]),
               sqrt(mean(abs2, traces[i].lift_coefficient[late])),
               mean(traces[i].density_rms_percent[late]),
               sqrt(mean(abs2, values))))
    end
    CSV.write(joinpath(output_directory, "summary.csv"), summary)
    println(summary)
    println("Comparison plots written to ", abspath(output_directory))
end

function main(args)
    options = Dict("case" => "all", "spacing-factor" => "0.125", "end-time" => "1.0",
                   "max-seconds" => "570", "output" => "out/vortex_street_corrections",
                   "neighbor-update" => "serial")
    plot_only = "--plot-only" in args
    vtk = "--vtk" in args
    for arg in args
        arg in ("--plot-only", "--vtk") && continue
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("expected --option=value, got $arg"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(options, key) || throw(ArgumentError("unknown option $key"))
        options[key] = value
    end
    spacing_factor = parse(Float64, options["spacing-factor"])
    end_time = parse(Float64, options["end-time"])
    max_seconds = parse(Float64, options["max-seconds"])
    cfg = VortexStreetCorrections.configuration(; spacing_factor, end_time, max_seconds,
                                                output_directory=options["output"],
                                                save_vtk=vtk)
    if !plot_only
        options["neighbor-update"] in ("parallel", "serial") ||
            throw(ArgumentError("neighbor-update must be parallel or serial"))
        update_strategy = options["neighbor-update"] == "serial" ? SerialUpdate() :
                          ParallelUpdate()
        cases = options["case"] == "all" ? VortexStreetCorrections.CASES :
                (Symbol(options["case"]),)
        for case in cases
            started_ns = time_ns()
            VortexStreetCorrections.run_case(case, cfg; started_ns,
                                             neighborhood_update=update_strategy)
            GC.gc()
        end
    end
    if plot_only || options["case"] == "all"
        comparison_plots(cfg.output_directory)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        main(ARGS)
    catch error
        if error isa ErrorException || error isa ArgumentError
            showerror(stderr, error)
            println(stderr)
            exit(1)
        end
        rethrow()
    end
end
