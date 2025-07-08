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

    # Reconstruct NN parameters
    nn_params = reconstruct(p_nn)
    u_pred = U(u, nn_params, _st)[1] # Network prediction
    
    # Add NN correction to the analytical model
    lorenz!(du, u, p_data, t)
    du .+= u_pred
end

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p_combined)

# Prediction function
function predict(p_combined, X = u0, T = t_data)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = p_combined)
    pred = solve(_prob, Rosenbrock23(), saveat = T, abstol = 1e-6, reltol = 1e-6,
            sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
    return Array(pred)
end

# Loss function
function loss(p_combined)
    u_pred = predict(p_combined)
    return mean(abs2, X_noisy - u_pred)
end

# Simple callback
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        # Extract current model parameters for monitoring
        len = length(θ_nn)
        p_current = θ[len+1:end]
        println("Iteration $(length(losses)): Loss = $(l)")
        println("Current p: [$(round(p_current[1], digits=3)), $(round(p_current[2], digits=3)), $(round(p_current[3], digits=3))]")
        println("True p: [$(p_true[1]), $(p_true[2]), $(round(p_true[3], digits=3))]")
    end
    return false
end

# Optimization setup
optf = OptimizationFunction((θ, p) -> loss(θ), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_combined)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 1000)
println("\nOptimization complete!")
println("Final loss: ", losses[end])

# ------------------------------------ Results Analysis ------------------------------------

# Extract final parameters
len = length(θ_nn)
nn_final = res.u[1:len]
p_final = res.u[len+1:end]

println("\nFinal Results:")
println("True parameters: ", p_true)
println("Initial parameters: ", p_data)
println("Final parameters: ", [round(p, digits=3) for p in p_final])
println("Parameter errors: ", [round(abs(p_true[i] - p_final[i]), digits=3) for i in 1:3])

# Test the trained model
u_pred_final = predict(res.u)

# Compare all results (clean, noisy, and prediction)
p_compare1 = plot(t_data, X_clean[1,:], label="True (clean)", xlabel="t", ylabel="x", linewidth=2, color=:black)
scatter!(p_compare1, t_data[1:5:end], X_noisy[1,1:5:end], label="Noisy data", color=:red, alpha=0.6, markersize=3)
plot!(p_compare1, t_data, u_pred_final[1,:], label="UDE prediction", linestyle=:dash, linewidth=2, color=:blue)
title!("X Component Comparison")
display(p_compare1)

# NN correction terms
function get_nn_correction(p_combined, X_state)
    len = length(θ_nn)
    nn_current = p_combined[1:len]
    nn_params = reconstruct(nn_current)
    return U(X_state, nn_params, _st)[1]
end

# Sample correction at a few time points
correction_sample = [get_nn_correction(res.u, X_noisy[:, i]) for i in 1:10:length(t_data)]
correction_matrix = hcat(correction_sample...)

p_correction = plot(t_data[1:10:end], correction_matrix[1,:], label="NN correction x", 
                   xlabel="t", ylabel="Correction", linewidth=2)
plot!(p_correction, t_data[1:10:end], correction_matrix[2,:], label="NN correction y")
plot!(p_correction, t_data[1:10:end], correction_matrix[3,:], label="NN correction z")
title!("Neural Network Correction Terms")
display(p_correction)

println("Final loss: ", loss(res.u))
println("Training completed successfully!")
println("\nThe NN learned to correct for the differences between the analytical model and noisy data.")