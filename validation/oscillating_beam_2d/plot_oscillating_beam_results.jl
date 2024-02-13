using PythonPlot
using JSON
using Glob
using CSV
using DataFrames
using Interpolations
using Statistics
using Printf

# Load the reference simulation data
dx_data = CSV.read("validation/Turek_dx_T.csv", DataFrame)
dy_data = CSV.read("validation/Turek_dy_T.csv", DataFrame)

# fix slight misalignment
dx_data.time = dx_data.time .+ 0.015
dx_data.displacement = dx_data.displacement .+ 0.00005
dy_data.displacement = dy_data.displacement .- 0.001

# Get the list of JSON files
json_files = glob("oscillating_beam_2d_positions_*.json", ".")

# Create subplots
fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 5))

# Regular expressions for matching keys
key_pattern_x = r"pos_x_\d+_solid_\d+"
key_pattern_y = r"pos_y_\d+_solid_\d+"

function calculate_mse(reference_data, simulation_data)
    # Interpolate simulation data
    interp_func = LinearInterpolation(simulation_data["time"], simulation_data["values"])

    # Align with reference data time points
    interpolated_values = interp_func(reference_data.time)

    # Calculate MSE
    mse = mean((interpolated_values .- reference_data.displacement) .^ 2)
    return mse
end

for json_file in json_files
    json_data = JSON.parsefile(json_file)

    local resolution = parse(Int, split(split(json_file, "_")[end], ".")[1])

    # Find matching keys and plot data for each key
    matching_keys_x = sort(collect(filter(key -> occursin(key_pattern_x, key),
                                          keys(json_data))))
    matching_keys_y = sort(collect(filter(key -> occursin(key_pattern_y, key),
                                          keys(json_data))))

    particle_spacing = cylinder_diameter / resolution

    # calculate error compared to reference
    mse_results_x = 0
    mse_results_y = 0

    for key in matching_keys_x
        data = json_data[key]
        mse_results_x = calculate_mse(dx_data, data)
    end

    for key in matching_keys_y
        data = json_data[key]
        mse_results_y = calculate_mse(dy_data, data)
    end

    # Plot x-axis displacements
    for key in matching_keys_x
        data = json_data[key]
        times = data["time"]
        values = data["values"]
        initial_position = values[1]
        displacements = [value - initial_position for value in values]
        ax1.plot(times, displacements,
                 label="dp = $(particle_spacing) mse=$(@sprintf("%.5f", mse_results_x))")
    end

    # Plot y-axis displacements
    for key in matching_keys_y
        data = json_data[key]
        times = data["time"]
        values = data["values"]
        initial_position = values[1]
        displacements = [value - initial_position for value in values]
        ax2.plot(times, displacements,
                 label="dp = $(particle_spacing) mse=$(@sprintf("%.5f", mse_results_y))")
    end
end

ax1.plot(dx_data.time, dx_data.displacement, label="Turek and Hron 2006", color="black",
         linestyle="--")
ax2.plot(dy_data.time, dy_data.displacement, label="Turek and Hron 2006", color="black",
         linestyle="--")

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("X Displacement")
ax1.set_title("X-Axis Displacement")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Y Displacement")
ax2.set_title("Y-Axis Displacement")
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

fig.subplots_adjust(right=0.7)
fig.tight_layout()

plotshow()
