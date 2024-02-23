include("../validation_util.jl")

# activate for interactive plot
#using GLMakie
using CairoMakie
using CSV
using DataFrames
using JSON
using Glob
using Printf

elastic_plate = (length=0.35, thickness=0.02)

# Load the reference simulation data
ref = CSV.read("validation/oscillating_beam_2d/oscillating_beam_l5_dt_005.csv", DataFrame)

# Get the list of JSON files
reference_files = glob("validation_reference_oscillating_beam_2d_*.json",
                       "validation/oscillating_beam_2d/")
simulation_files = glob("validation_run_oscillating_beam_2d_*.json", "out")
merged_files = vcat(reference_files, simulation_files)
input_files = sort(merged_files, by=extract_number)

# Regular expressions for matching keys
key_pattern_x = r"mid_point_x_solid_\d+"
key_pattern_y = r"mid_point_y_solid_\d+"

# Setup for Makie plotting
fig = Figure(size=(1200, 800))
ax1 = Axis(fig, title="X-Axis Displacement", xlabel="Time [s]", ylabel="X Displacement")
ax2 = Axis(fig, title="Y-Axis Displacement", xlabel="Time [s]", ylabel="Y Displacement")
fig[1, 1] = ax1
fig[2, 1] = ax2

for file_name in input_files
    println("Loading the input file: $file_name")
    json_data = JSON.parsefile(file_name)

    resolution = parse(Int, split(split(file_name, "_")[end], ".")[1])
    particle_spacing = elastic_plate.thickness / resolution

    matching_keys_x = sort(collect(filter(key -> occursin(key_pattern_x, key),
                                          keys(json_data))))
    matching_keys_y = sort(collect(filter(key -> occursin(key_pattern_y, key),
                                          keys(json_data))))

    if isempty(matching_keys_x)
        error("No matching keys found in: $file_name")
    end

    label_prefix = occursin("reference", file_name) ? "Reference" : ""

    for (matching_keys, ax) in ((matching_keys_x, ax1), (matching_keys_y, ax2))
        for key in matching_keys
            data = json_data[key]
            times = Float64.(data["time"])
            positions = Float64.(data["values"])
            initial_position = positions[1]
            displacements = [v - initial_position for v in positions]

            mse_results = occursin(key_pattern_x, key) ?
                          calculate_mse(ref.time, ref.Ux, data["time"], displacements) :
                          calculate_mse(ref.time, ref.Uy, data["time"], displacements)

            label = "$label_prefix dp = $(@sprintf("%.8f", particle_spacing)) mse=$(@sprintf("%.8f", mse_results))"
            lines!(ax, times, displacements, label=label)
        end
    end
end

# Plot reference data
lines!(ax1, ref.time, ref.Ux, color=:black, linestyle=:dash,
       label="Turek and Hron 2006")
lines!(ax2, ref.time, ref.Uy, color=:black, linestyle=:dash,
       label="Turek and Hron 2006")

legend_ax1 = Legend(fig[1, 2], ax1)
legend_ax2 = Legend(fig[2, 2], ax2)
fig
