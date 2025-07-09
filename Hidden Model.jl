# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, LuxCore, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

# --------------------------------------- Analytical Model ----------------------------------

# Define the multi-mass system model
function hidden_model!(du, u, p, t, current_acceleration)
    # Unpack state variables
    x1, v1, x2, v2, x3, v3, x4, v4 = u
    
    # Unpack parameters 
    m1, m2, m3, m4, k1, k2, k3, k4, c1, c2, c3, c4 = p

    # Use current_acceleration as the external force
    Fext = current_acceleration
    
    # Force calculations for each mass
    # Mass 1
    F1_spring = -(k1 + k2) * x1 + k2 * x2 
    F1_damping = -c1 * x1 + c2 * (x2 - x1) * (v2 - v1) * abs(v2 - v1)
    F1_total = F1_spring + F1_damping
    
    # Mass 2
    F2_spring = k2 * x1 - (k2 + k3) * x2 + k3 * x3
    F2_damping = c2 * (x1 - x2) * (v1 - v2) * abs(v1 - v2) + c3 * (v3 - v2)
    F2_total = F2_spring + F2_damping
    
    # Mass 3
    F3_spring = k3 * (x2 - x3) + k4 * (x4 - x3)^3
    F3_damping = c3 * (v2 - v3) + c4 * (v4 - v3)
    F3_total = F3_spring + F3_damping
    
    # Mass 4
    F4_spring = k4 * (x3 - x4)^3  
    F4_damping = c4 * (v3 - v4)  
    F4_total = F4_spring + F4_damping
    
    # Model state space
    du[1] = v1
    du[2] = F1_total / m1 + Fext
    du[3] = v2
    du[4] = F2_total / m2 
    du[5] = v3
    du[6] = F3_total / m3  
    du[7] = v4  
    du[8] = F4_total / m4
end

# --------------------------------------- External Force ------------------------------------

# Sine Wave External Force
f = 2.0 # Frequency (Hz)
alpha = 0.05 # Applied acceleration constant
g = 9.81 # Gravitational constant 
A = alpha * g
t_ramp = 1.0 # Ramp-up duration (s)
# Define the ramp function (linear ramp)
ramp(t) = t < t_ramp ? t / t_ramp : 1.0
Fext_sine = t -> A * ramp(t) * sin(2 * π * f * t)

# ------------------------------------- Set Input Force ------------------------------------

# Set to `true` to use sine wave, `false` otherwise
use_sine = true

# Define Fext_input based on your choice
if use_sine
    Fext_input = Fext_sine
else
    Fext_input = t -> 0.0  # Default to zero force
end

# ------------------------------------ Initialize Parameters --------------------------------

# Initial conditions
u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
p_true = [1.0, 0.75, 1.2, 0.9, 10.0, 8.0, 6.0, 0.75, 0.5, 0.3, 0.4, 0.2]                
tspan = (0.0, 20.0) # simulation length
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model ---------------------------------

# Define a wrapper function
function hidden_model_wrapper!(dz, z, p, t) 
    current_acceleration = Fext_input(t)
    hidden_model!(dz, z, p, t, current_acceleration)
end

# Define and solve the ODE problem
eqn = ODEProblem(hidden_model_wrapper!, u0, tspan, p_true)

# Solve the system using Rosenbrock23 solver
sol = solve(eqn, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7, saveat=tspan[1]:0.001:tspan[2])

# Verify the solution structure
println("Type of sol.u: ", typeof(sol.u))
println("Size of sol.u: ", size(sol.u))
println("Solver status: ", sol.retcode)

# ----------------------------------------- Plotting -----------------------------------------

# Plot the related terms
x1 = [u[1] for u in sol.u]
v1 = [u[2] for u in sol.u]
x2 = [u[3] for u in sol.u]
v2 = [u[4] for u in sol.u]
x3 = [u[5] for u in sol.u]
v3 = [u[6] for u in sol.u]
x4 = [u[7] for u in sol.u]
v4 = [u[8] for u in sol.u]
u = sol.u
t = sol.t

# Plot positions vs time

p1 = plot(sol.t, [x1 x2 x3 x4], 
          xlabel = "Time (s)", 
          ylabel = "Position", 
          label = ["Mass 1" "Mass 2" "Mass 3" "Mass 4"],
          title = "Mass Positions vs Time")
display(p1)

# Plot velocities vs time  
p2 = plot(sol.t, [v1 v2 v3 v4], 
          xlabel = "Time (s)", 
          ylabel = "Velocity", 
          label = ["Mass 1" "Mass 2" "Mass 3" "Mass 4"],
          title = "Mass Velocities vs Time")
display(p2)

# Phase portrait for each mass
p3 = plot(x1, v1, xlabel = "Position", ylabel = "Velocity", 
          title = "Phase Portrait - Mass 1", legend = false)
display(p3)

# Configuration plot (masses positions relative to each other)
p4 = plot(sol.t, [x1 x2-x1 x3-x2 x4-x3], 
          xlabel = "Time (s)", 
          ylabel = "Relative Position", 
          label = ["Mass 1 (abs)" "Mass 2-1" "Mass 3-2" "Mass 4-3"],
          title = "Relative Mass Positions")
display(p4)