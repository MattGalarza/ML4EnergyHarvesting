# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

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
p = [10, 28, 8/3]
tspan = (0.0, 50) # simulation length
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model ---------------------------------

# Define and solve the ODE problem
eqn = ODEProblem(lorenz!, u0, tspan, p)

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

p1 = plot(sol.t, x, xlabel = "t", ylabel = "x")
display(p1)
p2 = plot(sol.t, y, xlabel = "t", ylabel = "y")
display(p2)
p3 = plot(sol.t, z, xlabel = "t", ylabel = "z")
display(p3)
p4 = plot(x, y, z, xlabel = "x", ylabel = "y", zlabel = "z", legend = false)
display(p4)

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

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    u_pred = U(u, p, _st)[1] # Network prediction
    du[1] = u_pred[1]
    du[2] = u_pred[2]
    du[3] = u_pred[3]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# Prediction function
function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

# Loss function
function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
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

# Optimization setup
optf = OptimizationFunction((θ, p) -> loss(θ), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_flat)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 1000)
println("\nOptimization complete!")
println("Final loss: ", losses[end])