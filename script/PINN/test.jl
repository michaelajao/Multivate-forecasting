using CSV, DataFrames, Lux, ComponentArrays, DifferentialEquations, Optimization, OptimizationOptimJL
using OptimizationOptimisers, Plots, DiffEqFlux, Statistics, Random


pwd()
Random.seed!(1234)
# Load data
function load_data(path)
    data = CSV.read(path, DataFrame)
    return data
end


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
data_length = min(10, nrow(data))
infected_data = rolling_infected[1:data_length]
death_data = rolling_death[1:data_length]
population = data[1, "population"]

# Model initial conditions
u0 = [population - (infected_data[1] * 100) - infected_data[1] - death_data[2], (infected_data[1] * 100), infected_data[1], 1, 1]
tspan = (1, data_length)
t = range(tspan[1], tspan[2], length=data_length)
p0 = [0.2, 0.07]
# Neural network for β, γ, δ
function create_NN(input_size, hidden_size, output_size)
    nn = Lux.Chain(
        Lux.Dense(input_size, hidden_size, Lux.tanh),
        Lux.Dense(hidden_size, hidden_size, Lux.tanh),
        Lux.Dense(hidden_size, hidden_size, Lux.tanh),
        Lux.Dense(hidden_size, output_size, sigmoid))
    p, st = Lux.setup(Random.default_rng(), nn)
    return nn, p, st
end

NN_β, p_β, st_β = create_NN(1, 10, 1)
NN_γ, p_γ, st_γ = create_NN(1, 10, 1)
NN_δ, p_δ, st_δ = create_NN(1, 10, 1)
NN_α, p_α, st_α = create_NN(1, 10, 1)

nn_params = ComponentArray(β=p_β, γ=p_γ, δ=p_δ, α=p_α)

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
    sol = solve(prob_pred, Tsit5(), p=θ, saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

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
    c = 1e-2
    loss = sum(abs2, log.(abs.(infected_data) .+ c) .- log.(abs.(prediction[3, :]) .+ c)) +
           sum(abs2, log.(abs.(death_data) .+ c) .- log.(abs.(prediction[5, :]) .+ c)) +
        sum(abs2, Rt_values .- 4.0)
    return loss
end

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
    mape = mean(abs.((predicted .- actual) ./ actual)) * 100
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

global losses = Float64[]
global iter = 0

function callback(opt_state, l)
    global iter += 1
    if iter % 100 == 0
        println("Current loss after $(iter) iterations: $l")
    end
    push!(losses, l)
    return false # Continue optimization
end

# Optimization
# Optimization
optf = Optimization.OptimizationFunction((θ, _) -> loss_adjoint(θ), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, nn_params)
res = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.001), callback=callback, maxiters=20000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# optprob2 = remake(optprob, u0=res.minimizer)
# res1 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.0001), callback=callback, maxiters=100)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Evaluate model performance
predicted_data, Rt_values = predict_adjoint(res.minimizer)
mse, mae, mape, rmse = evaluate_model(predicted_data[3, :], infected_data)
mse, mae, mape, rmse = evaluate_model(predicted_data[5, :], death_data)

println("Mean Squared Error death_data: $mse")
println("Mean Absolute Error death_data: $mae")
println("Mean Absolute Percentage Error death_data: $mape")
println("Root Mean Squared Error death_data: $rmse")


println("Mean Squared Error infected: $mse")
println("Mean Absolute Error infected: $mae")
println("Mean Absolute Percentage Error infected: $mape")
println("Root Mean Squared Error infected: $rmse")



# # Visualization
# pl_loss = plot(losses, label="Loss", xlabel="Iterations", ylabel="Loss", title="Training Loss", lw=2, legend=:topleft)
# pl_infected = bar(t, infected_data, label="I data", color=:red, alpha=0.5)
# pl_death = bar!(t, death_data, label="D data", color=:blue, alpha=0.5)
# pl_pred_infected = plot!(t, predicted_data[3, :], label="I prediction", color=:red, lw=2)
# pl_pred_death = plot!(t, predicted_data[5, :], label="D prediction", color=:blue, lw=2)
# xlabel!("Time (days)")
# ylabel!("Number of Cases")
# title!("Model Predictions vs Actual Data")
# savefig("C:\\Users\\ajaoo\\Desktop\\Projects\\hospitalisation-PINN\\reports\\figures\\SEIRD_UDE.png")


#Extract predicted compartments
β_values = [abs(NN_β([ti], p_β, st_β)[1][1]) for ti in t]
γ_values = [abs(NN_γ([ti], p_γ, st_γ)[1][1]) for ti in t]
δ_values = [abs(NN_δ([ti], p_δ, st_δ)[1][1]) for ti in t]
α_values = [abs(NN_α([ti], p_α, st_α)[1][1]) for ti in t]



function plot_training_loss(losses)
    plot(losses, label="Loss", xlabel="Iterations", ylabel="Loss", title="Training Loss", linewidth=2, legend=:topleft)
    savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/training_loss1.pdf")
end

function plot_infection_data(t, infected_data, predicted_infected)
    p = bar(t, infected_data, label="I data", color=:red, alpha=0.5)
    plot!(p, t, predicted_infected, label="I prediction", color=:red, linewidth=2)
    savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/plot_infection_data1.pdf")
    return p
end

function plot_death_data(t, death_data, predicted_death)
    p = bar(t, death_data, label="D data", color=:blue, alpha=0.5)
    plot!(p, t, predicted_death, label="D prediction", color=:blue, linewidth=2)
    savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/plot_death_data2.pdf")
    return p
end

function plot_rt_values(t, Rt_values)
    plot(t, Rt_values, label="Rₜ", color=:black, ylabel="Rₜ", xlabel="Days", title="Effective Reproduction Number Rₜ Over Time", linewidth=2, legend=:topleft)
    hline!([1], linestyle=:dash, label="Threshold Rₜ=1")
    savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/Rt_values1.pdf")
end

function plot_parameter_dynamics(t, β_values, γ_values, δ_values)
    p = plot(t, β_values, label="β (Transmission Rate)", color=:blue, legend=:topright, xlabel="Time (days)", ylabel="Parameter Value", title="Parameter Dynamics Over Time")
    plot!(p, t, γ_values, label="γ (Recovery Rate)", color=:green)
    plot!(p, t, δ_values, label="δ (Mortality Rate)", color=:red)
    plot!(p, t, α_values, label="α (Incubation Rate)", color=:orange)
    savefig("/share/home2/olarinoyem/Project/Multivate-forecasting/images/parameter_dynamics1.pdf")
    return p
end

pl_loss = plot_training_loss(losses)
pl_infected = plot_infection_data(t, infected_data, predicted_data[3, :])
pl_death = plot_death_data(t, death_data, predicted_data[5, :])
pl_Rt = plot_rt_values(t, Rt_values)

pl_parameters = plot_parameter_dynamics(t, β_values, γ_values, δ_values)

final_plot = plot(pl_loss, pl_infected, pl_death, pl_Rt, pl_parameters, layout=(3, 2), size=(1000, 800))
savefig(final_plot, "C:\\Users\\ajaoo\\Desktop\\Projects\\hospitalisation-PINN\reports\\figures\\SEIRD_UDE.png")
