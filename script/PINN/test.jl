using CSV, DataFrames, Lux, ComponentArrays, DifferentialEquations, Optimization, OptimizationOptimJL
using OptimizationOptimisers, Plots, DiffEqFlux, Statistics, Random, CUDA

function load_data(path)
    data = CSV.File(path) |> DataFrame
    return data
end

function rolling_mean(data, window_size)
    n = length(data)
    [mean(data[i:i+window_size-1]) for i in 1:n-window_size+1]
end

# Load your data
data_path = "/share/home2/olarinoyem/Project/Multivate-forecasting/data/region_daily_data/East Midlands.csv"  # Adjust this path
data = load_data(data_path)

# Preprocess data
infected_data = rolling_mean(data[!, "new_confirmed"], 7)
death_data = rolling_mean(data[!, "new_deceased"], 7)


function create_NN_gpu(input_size, hidden_size, output_size)
    nn = Lux.Chain(
        Lux.Dense(input_size, hidden_size, Lux.tanh_fast),
        Lux.Dense(hidden_size, hidden_size, Lux.tanh_fast),
        Lux.Dense(hidden_size, output_size, Lux.sigmoid_fast)
    )
    p, st = Lux.setup(CUDA.default_rng(), nn)
    return nn, CUDA.cu(p), CUDA.cu(st)
end

# Create neural networks on the GPU
NN_β, p_β, st_β = create_NN_gpu(1, 50, 1)
NN_γ, p_γ, st_γ = create_NN_gpu(1, 50, 1)
NN_δ, p_δ, st_δ = create_NN_gpu(1, 50, 1)
NN_α, p_α, st_α = create_NN_gpu(1, 50, 1)

# Initialize model parameters and conditions
CUDA.@allowscalar begin
    nn_params = ComponentArray(β=p_β, γ=p_γ, δ=p_δ, α=p_α)
end

population = data[1, "population"]
u0 = CUDA.cu([population - (infected_data[1] * 100) - infected_data[1] - death_data[2], 
              (infected_data[1] * 100), infected_data[1], 1, 1])
tspan = (1.0f0, float(data_length))
t = CUDA.cu(range(tspan[1], tspan[2], length=length(infected_data)))

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


function loss_adjoint(θ)
    # Update the model parameters
    nn_params.β .= θ.β
    nn_params.γ .= θ.γ
    nn_params.δ .= θ.δ
    nn_params.α .= θ.α

    # Solve the ODE
    prob = ODEProblem(seird_model!, u0, tspan, nn_params)
    sol = solve(prob, Tsit5(), saveat=t)

    # Calculate the loss
    loss = sum(abs2, sol[3, :] - infected_data) + sum(abs2, sol[5, :] - death_data)
    return loss
end

# Define the optimization function
function optimization_function(θ, _)
    loss_value = loss_adjoint(θ)  # Ensure this function is GPU-compatible
    return loss_value
end

optf = OptimizationFunction(optimization_function, Optimization.AutoZygote())

# Create an optimization problem

CUDA.@allowscalar begin

    optprob = OptimizationProblem(optf, nn_params)
end


# Run the optimization
optimizer = OptimizationOptimisers.ADAM(0.001)
result = Optimization.solve(optprob, optimizer, maxiters=15000)

println("Training completed with final loss: $(result.minimum)")

# Define a callback function



function callback(p, opt_state, iteration)
    loss_cpu = CUDA.@allowscalar opt_state.loss |> cpu
    if iteration % 100 == 0
        println("Iteration $iteration: Loss = $loss_cpu")
    end
    return false  # Signal to continue optimization
end

# Run the optimization with the callback
result = Optimization.solve(optprob, optimizer, maxiters=15000, callback=callback)

println("Optimization completed. Final parameters stored in `result.minimizer`.")


# Define the optimization function
function optimization_function(θ, _)
    loss_value = loss_adjoint(θ)  # Ensure this function is GPU-compatible
    return loss_value
end

optf = OptimizationFunction(optimization_function, Optimization.AutoZygote())

# Create an optimization problem
optprob = OptimizationProblem(optf, CUDA.cu(nn_params))

# Run the optimization
optimizer = OptimizationOptimisers.ADAM(0.001)
result = Optimization.solve(optprob, optimizer, maxiters=15000, callback=callback)

println("Training completed with final loss: $(result.minimum)")
