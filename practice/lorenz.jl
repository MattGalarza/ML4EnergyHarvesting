# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, LuxCore, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

# --------------------------------------- Analytical Model ----------------------------------

# Define the lorenz model
function lorenz!(du, u, p, t)
    # Unpack state variables
    x, y, z = u

    # Unpack parameters
    σ, ρ, β = p

    # Compute derivatives
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

# ------------------------------------ Initialize Parameters --------------------------------

# Initial conditions
u0 = [1, 1, 1]
p_true = [10, 28, 8/3]
tspan = (0.0, 50) # simulation length
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model ---------------------------------

# Define and solve the ODE problem
eqn = ODEProblem(lorenz!, u0, tspan, p_true)

# Solve the system using Rosenbrock23 solver
sol = solve(eqn, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7)

# Verify the solution structure
println("Type of sol.u: ", typeof(sol.u))
println("Size of sol.u: ", size(sol.u))
println("Solver status: ", sol.retcode)

# ----------------------------------------- Plotting -----------------------------------------

# Plot the related terms
x = [u[1] for u in sol.u]
y = [u[2] for u in sol.u]
z = [u[3] for u in sol.u]
u = sol.u
t = sol.t

p1 = plot(sol.t, x, xlabel = "t", ylabel = "x")
display(p1)
p2 = plot(sol.t, y, xlabel = "t", ylabel = "y")
display(p2)
p3 = plot(sol.t, z, xlabel = "t", ylabel = "z")
display(p3)
p4 = plot(x, y, z, xlabel = "x", ylabel = "y", zlabel = "z", legend = false)
display(p4)

# ------------------------------------ Create noisy data ------------------------------------

# Create noisy data from the analytical solution
rng = Random.default_rng(1111)
X_clean = hcat(sol.u...) 
t_data = sol.t

# Add noise to the solution
noise_magnitude = 0.15
X_noisy = X_clean .+ noise_magnitude * randn(rng, size(X_clean))

# Plot data comparison
p5 = plot(t_data, X_clean[1,:], label="Clean x", xlabel="t", ylabel="x", linewidth=2)
plot!(p5, t_data, X_noisy[1,:], label="Noisy x", linestyle=:dash, alpha=0.7)
display(p5)
p6 = plot(t_data, X_clean[2,:], label="Clean x", xlabel="t", ylabel="x", linewidth=2)
plot!(p6, t_data, X_noisy[2,:], label="Noisy x", linestyle=:dash, alpha=0.7)
display(p6)
p7 = plot(t_data, X_clean[3,:], label="Clean x", xlabel="t", ylabel="x", linewidth=2)
plot!(p7, t_data, X_noisy[3,:], label="Noisy x", linestyle=:dash, alpha=0.7)
display(p7)
p8 = plot(X_clean[1,:], X_clean[2,:], X_clean[3,:], label="Clean x", xlabel = "x", ylabel = "y", zlabel = "z")
plot!(p8, X_noisy[1,:], X_noisy[2,:], X_noisy[3,:], label="Noisy x", xlabel = "x", ylabel = "y", zlabel = "z", linestyle=:dash, alpha=0.7)
display(p8)

# ------------------------------------ Setting up the UDE ------------------------------------

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Regular deep NN chain
const U = Lux.Chain(Lux.Dense(3, 32, rbf),
                    Lux.Dense(32, 32, rbf),
                    Lux.Dense(32, 3) 
) 

# Initialize NN
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

p_nn, reconstruct = LuxCore.vectorize(nn_params)
p_data = [9.5, 27.0, 2.5]
p_combined = vcat(p_nn, p_data)  # Physical parameters + NN parameters

# Define the hybrid model
function ude_dynamics!(du, u, p_combined, t, p_true)
    # Split combined parameters
    len = length(p_nn)
    p_nn = p_combined[1:len]
    p_data = p_combined[len+1:end]

    u_pred = U(u, p, _st)[1] # Network prediction
    du[1] = u_pred[1]
    du[2] = u_pred[2]
    du[3] = u_pred[3]
end

function ude_dynamics!(du, u, θ_combined, t)
    # Split combined parameters
    n_nn = length(θ_nn)
    θ_nn_current = θ_combined[1:n_nn]
    p_current = θ_combined[n_nn+1:end]
    
    # Reconstruct NN parameters
    nn_params_current = reconstruct(θ_nn_current)
    
    # Get NN correction term
    nn_correction = U(u, nn_params_current, _st)[1]
    
    # Analytical Lorenz model with current parameters
    x, y, z = u
    σ, ρ, β = p_current
    
    # Hybrid dynamics: Analytical model + NN correction
    du[1] = σ * (y - x) + nn_correction[1]
    du[2] = x * (ρ - z) - y + nn_correction[2]
    du[3] = x * y - β * z + nn_correction[3]
end


# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, sol_data[:, 1], tspan, p)

# Prediction function
function predict(θ, X = sol_data[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    pred = solve(_prob, Rosenbrock23(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
    return Array(pred)
end

# Loss function
function loss(θ)
    u_pred = predict(θ)
    return mean(abs2, u_data - u_pred)
end

# Simple callback
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 5 == 0
        println("Iteration $(length(losses)): Loss = $(l)")
    end
    return false
end

# Optimization setup - optimize the neural network parameters, not the Lorenz parameters
optf = OptimizationFunction((θ, p) -> loss(θ), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, θ)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 1000)
println("\nOptimization complete!")
println("Final loss: ", losses[end])

# ------------------------------------ Results Analysis ------------------------------------

# Test the trained model
u_pred_final = predict(res.u)

# Plot comparison
p_compare = plot(t, u_data[1,:], label="True x", xlabel="t", ylabel="x")
plot!(p_compare, t, u_pred_final[1,:], label="Predicted x", linestyle=:dash)
display(p_compare)

println("Final loss: ", loss(res.u))
println("Training completed successfully!")