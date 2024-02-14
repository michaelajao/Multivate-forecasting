using CSV, DataFrames, Lux, ComponentArrays, DifferentialEquations, Optimization, OptimizationOptimJL
using OptimizationOptimisers, DiffEqFlux, Statistics, Random, LuxAMDGPU, LuxCUDA, LineSearches
using CairoMakie, MakiePublication
CUDA.allowscalar(false)

CairoMakie.activate!()


pwd()
Random.seed!(1234)
# Load data
function load_data(path)
    data = CSV.File(path) |> DataFrame
    return data
end

# Get the device determined by Lux
device = gpu_device()


data_path = "/share/home2/olarinoyem/Project/Multivate-forecasting/data/region_daily_data/East Midlands.csv"
data = load_data(data_path)

infected_data = data[!, "new_confirmed"]
death_data = data[!, "new_deceased"]

function rolling_mean(data, window_size)
    n = length(data)
    [mean(data[i:i+window_size-1]) for i in 1:n-window_size+1]
end

# Assuming 'infected_data' and 'death_data' are your datasets
rolling_infected = rolling_mean(infected_data, 7)
rolling_death = rolling_mean(death_data, 7)

# Prepare data
data_length = min(40, nrow(data))
infected_data = rolling_infected[1:data_length]
death_data = rolling_death[1:data_length]
population = data[1, "population"]

# Model initial conditions
u0 = [population - (infected_data[1] * 100) - infected_data[1] - death_data[2], (infected_data[1] * 100), infected_data[1], 1, 1]
tspan = (1, data_length)
t = range(tspan[1], tspan[2], length=data_length)

# Neural network for β, γ, δ
function create_NN(input_size, hidden_size, output_size)
    nn = Chain(
        Dense(input_size, hidden_size, Lux.tanh_fast),
        Dense(hidden_size, hidden_size, Lux.tanh_fast),
        Dense(hidden_size, hidden_size, Lux.tanh_fast),
        Dense(hidden_size, hidden_size, Lux.tanh_fast),
        Dense(hidden_size, output_size))
    p, st = Lux.setup(MersenneTwister(), nn)
    return nn, p, st
end

NN_β, p_β, st_β = create_NN(1, 50, 1)
NN_γ, p_γ, st_γ = create_NN(1, 50, 1)
NN_δ, p_δ, st_δ = create_NN(1, 50, 1)
NN_α, p_α, st_α = create_NN(1, 50, 1)

const nn_params = ComponentArray{Float64}(β=p_β, γ=p_γ, δ=p_δ, α=p_α)

# SEIRD model with neural network
function seird_model!(du, u, p, t)
    S, E, I, R, D = u
    N = sum(u)

    β = abs(NN_β([t], p.β, st_β)[1][1])
    γ = abs(NN_γ([t], p.γ, st_γ)[1][1])
    δ = abs(NN_δ([t], p.δ, st_δ)[1][1])
    α = abs(NN_α([t], p.α, st_α)[1][1])

    du[1] = -β * S * I / N
    du[2] = β * S * I / N - α * E
    du[3] = α * E - γ * I
    du[4] = γ * I
    du[5] = δ * I
end

prob_pred = ODEProblem(seird_model!, u0, tspan, nn_params)

# Loss function
function predict_adjoint(θ)
    sol = solve(prob_pred, RK4(), p=θ, saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

    # Extracting time-varying parameters β, γ, and δ for each time point
    β_values = [abs(NN_β([Float32(ti)], θ.β, st_β)[1][1]) for ti in t]
    γ_values = [abs(NN_γ([Float32(ti)], θ.γ, st_γ)[1][1]) for ti in t]
    δ_values = [abs(NN_δ([Float32(ti)], θ.δ, st_δ)[1][1]) for ti in t]
    α_values = [abs(NN_α([Float32(ti)], θ.α, st_α)[1][1]) for ti in t]

    # Calculating R_t for each time point based on β, γ, and δ
    Rt_values = β_values ./ (γ_values .+ δ_values)

    return Array(sol), Rt_values
end
# function predict_adjoint(θ)
#     sol = solve(prob_pred, Tsit5(), p=θ, saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
#     return Array(sol) # Just return the solution array
# end


# function loss_adjoint(θ)
#     prediction = predict_adjoint(θ) # Get model predictions

#     # Compute mean squared logarithmic error for infected and deceased
#     c = 1e-1
#     infected_loss = sum(abs2, log.(abs.(infected_data) .+ c) - log.(abs.(prediction[3, :]) .+ c))
#     death_loss = sum(abs2, log.(abs.(death_data) .+ c) - log.(abs.(prediction[5, :]) .+ c))

#     # Compute Rt for each time point and penalize deviations from 1.0
#     β_values = [abs(NN_β([Float32(ti)], θ.β, st_β)[1][1]) for ti in t]
#     γ_values = [abs(NN_γ([Float32(ti)], θ.γ, st_γ)[1][1]) for ti in t]
#     δ_values = [abs(NN_δ([Float32(ti)], θ.δ, st_δ)[1][1]) for ti in t]
#     Rt_values = β_values ./ (γ_values .+ δ_values)
#     Rt_loss = sum(abs2, Rt_values .- 4.0)

#     # Combine losses
#     total_loss = infected_loss + death_loss + Rt_loss
#     return total_loss
# end

function loss_adjoint(θ)
    prediction, Rt_values = predict_adjoint(θ)
    global c = 1e-1
    loss = sum(abs2, log.(abs.(infected_data) .+ c) .- log.(abs.(prediction[3, :]) .+ c)) +
           sum(abs2, log.(abs.(death_data) .+ c) .- log.(abs.(prediction[5, :]) .+ c)) +
           sum(abs2.(Rt_values .- 1.0))
    return loss
end

loss_adjoint(nn_params)

function calculate_mse(predicted, actual)
    mse = mean((predicted .- actual) .^ 2)
    return mse
end

#function to calculate Mean absolute error
function calculate_mae(predicted, actual)
    mae = mean(abs.(predicted .- actual))
    return mae
end

#function to calculate Mean absolute percentage error
function calculate_mape(predicted, actual)
    c = 1e-1
    mape = mean(abs.((predicted .- actual) ./ actual .+ c)) * 100
    return mape
end

function calculate_RMSE(predicted, actual)
    rmse = sqrt(mean((predicted .- actual) .^ 2))
    return rmse
end

#functions to evaluate model performance
function evaluate_model(predicted, actual)
    mse = calculate_mse(predicted, actual)
    mae = calculate_mae(predicted, actual)
    mape = calculate_mape(predicted, actual)
    rmse = calculate_RMSE(predicted, actual)
    return mse, mae, mape, rmse
end

const losses = Float64[]

function callback(θ, l)
    push!(losses, l)
    println("Training || Iteration: $(length(losses)) || Loss: $l")
    return false  # Signal to continue optimization
end

# Define the optimization function
optf = OptimizationFunction((θ, _) -> loss_adjoint(θ), AutoZygote())

# Create an optimization problem
optprob = OptimizationProblem(optf, nn_params)

# Run the first optimization with ADAM
res = Optimization.solve(optprob, ADAM(0.0001), callback=callback, maxiters=1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Prepare for the second optimization step
optprob2 = remake(optprob, u0=res.minimizer)

# Run the second optimization with BFGS
res1 = Optimization.solve(optprob2, BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking()), callback=callback, maxiters=100)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")


# Visualize the training loss for the adam and bfgs optimization
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Loss", title = "Training Loss")
    lines!(ax, losses, label="Loss", linewidth=2, legend=:topleft)
    return fig
end

# save images
save("/share/home2/olarinoyem/Project/Multivate-forecasting/images/ude/east_midlands_training_loss.pdf", fig)

# Optimization
# optf = Optimization.OptimizationFunction((θ, _) -> loss_adjoint(θ), Optimization.AutoZygote())
# optprob = Optimization.OptimizationProblem(optf, nn_params)
# res = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.0001), callback=callback, maxiters=15000)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")
# optprob2 = remake(optprob, u0=res.minimizer)
# res1 = Optimization.solve(optprob2, Optim.BFGS(:, initial_stepnorm=0.01, linesearch=LineSearches.BackTracking()), callback=callback, maxiters=1000)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Evaluate model performance
predicted_data, Rt_values = predict_adjoint(res1.minimizer)
mse, mae, mape, rmse = evaluate_model(predicted_data[3, :], infected_data)
println("Mean Squared Error infected: $mse")
println("Mean Absolute Error infected: $mae")
println("Mean Absolute Percentage Error infected: $mape")
println("Root Mean Squared Error infected: $rmse")

mse, mae, mape, rmse = evaluate_model(predicted_data[5, :], death_data)
println("Mean Squared Error death_data: $mse")
println("Mean Absolute Error death_data: $mae")
println("Mean Absolute Percentage Error death_data: $mape")
println("Root Mean Squared Error death_data: $rmse")


#Extract predicted compartments
β_values = [abs(NN_β([ti], p_β, st_β)[1][1]) for ti in t]
γ_values = [abs(NN_γ([ti], p_γ, st_γ)[1][1]) for ti in t]
δ_values = [abs(NN_δ([ti], p_δ, st_δ)[1][1]) for ti in t]
α_values = [abs(NN_α([ti], p_α, st_α)[1][1]) for ti in t]

# Plot infected and death data
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Number of Cases", title = "Model Predictions vs Actual Data")
    
    # Use `barplot` for bar plots in CairoMakie
    barplot!(ax, t, infected_data, color = :red, alpha = 0.5, label = "I data")
    barplot!(ax, t, death_data, color = :blue, alpha = 0.5, label = "D data")
    
    # Use `lines!` for line plots
    lines!(ax, t, predicted_data[3, :], color = :red, linewidth = 2, label = "I prediction")
    lines!(ax, t, predicted_data[5, :], color = :blue, linewidth = 2, label = "D prediction")

    # Add the legend
    leg = Legend(fig[1, 2], ax, "Legend", labelsize=12, fontsize=10, font="Arial", valign=:top, halign=:right)
    fig[1, 2] = leg  # Assign legend to the right side of the plot

    return fig
end

# Plot Rt values
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Rₜ", title = "Effective Reproduction Number Rₜ Over Time")
    lines!(ax, t, Rt_values, color = :black, label = "Rₜ", linewidth = 2)
    lines!(ax, [minimum(t), maximum(t)], [1, 1], color = :red, linestyle = :dash, linewidth = 2, label = "Threshold Rₜ=1")

    leg = Legend(fig[1, 2], ax, "Legend", valign=:top, halign=:right)
    fig[1, 2] = leg  # Assign legend to the right side of the plot
    return fig
end

# Plot beta
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Parameter Value", title = "Parameter Dynamics Over Time")
    lines!(ax, t, β_values, color = :blue, label = "β (Transmission Rate)", linewidth = 2)
    return fig
end

# Plot gamma
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Parameter Value", title = "Parameter Dynamics Over Time")
    lines!(ax, t, γ_values, color = :green, label = "γ (Recovery Rate)", linewidth = 2)
    return fig
end

# Plot rho
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Parameter Value", title = "Parameter Dynamics Over Time")
    lines!(ax, t, δ_values, color = :red, label = "δ (Mortality Rate)", linewidth = 2)
    return fig
end

# Plot alpha
fig = with_theme(theme_web()) do
    fig = Figure(resolution = (1000, 700))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "Days", ylabel = "Parameter Value", title = "Parameter Dynamics Over Time")
    lines!(ax, t, α_values, color = :orange, label = "α (Incubation Rate)", linewidth = 2)
    return fig
end



# function plot_training_loss(losses)
#     plot(losses, label="Loss", xlabel="Iterations", ylabel="Loss", title="Training Loss", linewidth=2, legend=:topleft)
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/training_loss1.pdf")
# end

# function plot_infection_data(t, infected_data, predicted_infected)
#     p = bar(t, infected_data, label="I data", title="Infection_data Plot", color=:red, alpha=0.5)
#     plot!(p, t, predicted_infected, label="I prediction", color=:red, linewidth=2)
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/plot_infection_data1.pdf")
#     return p
# end

# function plot_death_data(t, death_data, predicted_death)
#     p = bar(t, death_data, label="D data", title="death_data prediction plot", color=:blue, alpha=0.5)
#     plot!(p, t, predicted_death, label="D prediction", color=:blue, linewidth=2)
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/plot_death_data2.pdf")
#     return p
# end

# function plot_rt_values(t, Rt_values)
#     p = plot(t, Rt_values, label="Rₜ", color=:black, ylabel="Rₜ", xlabel="Days", title="Effective Reproduction Number Rₜ Over Time", linewidth=2, legend=:topleft)
#     hline!([1], linestyle=:dash, label="Threshold Rₜ=1")
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/Rt_values1.pdf")
#     return p
# end

# # function plot_parameter_dynamics(t, β_values, γ_values, δ_values)
# #     p = plot(t, β_values, label="β (Transmission Rate)", color=:blue, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
# #     plot!(p, t, γ_values, label="γ (Recovery Rate)", color=:green)
# #     plot!(p, t, δ_values, label="δ (Mortality Rate)", color=:red)
# #     plot!(p, t, α_values, label="α (Incubation Rate)", color=:orange)
# #     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/parameter_dynamics1.pdf")
# #     return p
# # end

# function plot_beta(t, β_values)
#     p = plot(t, β_values, label="β (Transmission Rate)", color=:blue, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/beta_parameter_dynamics1.pdf")
#     return p

# end

# function plot_gamma(t, γ_values)
#     p = plot(t, γ_values, label="γ (Recovery Rate)", color=:green, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/gamma_parameter_dynamics1.pdf")
#     return p

# end

# function plot_rho(t, δ_values)
#     p = plot(t, δ_values, label="δ (Mortality Rate)", color=:red, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/rho_parameter_dynamics1.pdf")
#     return p

# end

# function plot_alpha(t, α_values)
#     p = plot(t, α_values, label="α (Incubation Rate)", color=:orange, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
#     savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/alpha_parameter_dynamics1.pdf")
#     return p

# end

# pl_loss = plot_training_loss(losses)
# pl_infected = plot_infection_data(t, infected_data, predicted_data[3, :])
# pl_death = plot_death_data(t, death_data, predicted_data[5, :])
# pl_Rt = plot_rt_values(t, Rt_values)

# pl_beta = plot_beta(t, β_values)
# pl_gamma = plot_gamma(t, γ_values)
# pl_rho = plot_rho(t, δ_values)
# pl_alpha = plot_alpha(t, α_values)

# final_plot = plot(pl_loss, pl_infected, pl_death, pl_Rt, pl_beta, pl_gamma, pl_rho, pl_alpha, layout=(4, 2), size=(1000, 800))

# pl_parameters = plot_parameter_dynamics(t, β_values, γ_values, δ_values)

# final_plot = plot(pl_loss, pl_infected, pl_death, pl_Rt, pl_parameters, layout=(3, 2), size=(1000, 800))
# savefig(final_plot, "C:\\Users\\ajaoo\\Desktop\\Projects\\hospitalisation-PINN\reports\\figures\\SEIRD_UDE.png")
