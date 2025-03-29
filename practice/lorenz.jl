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
    du[1] = ρ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

# ------------------------------------ Initialize Parameters --------------------------------

# Initial conditions
u0 = [1, 0, 0]
p_ = [10, 28, 8/3]
tspan = (0.0, 25) # simulation length
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model ---------------------------------

# Define and solve the ODE problem
eqn = ODEProblem(lorenz!, u0, tspan, p_)

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