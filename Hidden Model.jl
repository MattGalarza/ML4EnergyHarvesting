# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, LuxCore, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

# --------------------------------------- Analytical Model ----------------------------------

# Define the multi-mass system 
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
    F1_damping = c2 * (x2 - x1) * (v2 - v1) * abs(v2 - v1)
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

# Define the multi-mass system with additional hidden physics
function hidden_physics_model!(du, u, p, t, current_acceleration)
    # Unpack state variables
    x1, v1, x2, v2, x3, v3, x4, v4 = u
    
    # Unpack parameters 
    m1, m2, m3, m4, k1, k2, k3, k4, c1, c2, c3, c4 = p

    # Use current_acceleration as the external force
    Fext = current_acceleration
    
    # Force calculations for each mass
    # Mass 1: Added nonlinear damping term, reduced spring force
    F1_spring = -(0.95 * k1 + k2) * x1 + k2 * x2 
    F1_damping = c1 * x1^3 + c2 * (x2 - x1) * (v2 - v1) * abs(v2 - v1)
    F1_total = F1_spring + F1_damping
    
    # Mass 2
    F2_spring = k2 * x1 - (k2 + k3) * x2 + k3 * x3
    F2_damping = c2 * (x1 - x2) * (v1 - v2) * abs(v1 - v2) + c3 * (v3 - v2)
    F2_total = F2_spring + F2_damping
    
    # Mass 3: Reduced damping force
    F3_spring = k3 * (x2 - x3) + k4 * (x4 - x3)^3
    F3_damping = c3 * (v2 - v3) + 0.75 * c4 * (v4 - v3)
    F3_total = F3_spring + F3_damping
    
    # Mass 4
    F4_spring = k4 * (x3 - x4)^3  
    F4_damping = 0.75 * c4 * (v3 - v4)  
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
alpha = 0.75 # Applied acceleration constant
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
p_true = [12.0, 8.5, 10.0, 9.25, 125.0, 80.0, 93.0, 75.0, 1.35, 3.75, 4.5, 2.0]                
tspan = (0.0, 20.0) # simulation length
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model: Original Physics ---------------------------------

# Define a wrapper function
function hidden_model_wrapper!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    hidden_model!(dz, z, p, t, current_acceleration)
end

# Define and solve the ODE problem
eqn1 = ODEProblem(hidden_model_wrapper!, u0, tspan, p_true)

# Solve the system using Rosenbrock23 solver
sol1 = solve(eqn1, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7) # saveat=tspan[1]:0.001:tspan[2]

# Verify the solution structure
println("Type of sol.u: ", typeof(sol1.u))
println("Size of sol.u: ", size(sol1.u))
println("Solver status: ", sol1.retcode)

# ----------------------------------------- Plotting: Hidden Model -----------------------------------------

# Plot the related terms
x1 = [u[1] for u in sol1.u]
v1 = [u[2] for u in sol1.u]
x2 = [u[3] for u in sol1.u]
v2 = [u[4] for u in sol1.u]
x3 = [u[5] for u in sol1.u]
v3 = [u[6] for u in sol1.u]
x4 = [u[7] for u in sol1.u]
v4 = [u[8] for u in sol1.u]
u1 = sol1.u
t1 = sol1.t

# Displacement plots
p1 = plot(sol.t, x1, ylabel = "x1 (m)", title = "Displacement", legend = false, seriescolor = 1, palette = :Dark2_5)
p2 = plot(sol.t, x2, ylabel = "x2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p3 = plot(sol.t, x3, ylabel = "x3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p4 = plot(sol.t, x4, xlabel = "Time (s)", ylabel = "x4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined1 = plot(p1, p2, p3, p4, layout = (4, 1), size = (800, 600))
display(p_combined1)

# Velocity plots
p5 = plot(sol.t, v1, ylabel = "v1 (m)", title = "Velocity", legend = false, seriescolor = 1, palette = :Dark2_5)
p6 = plot(sol.t, v2, ylabel = "v2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p7 = plot(sol.t, v3, ylabel = "v3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p8 = plot(sol.t, v4, xlabel = "Time (s)", ylabel = "v4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined2 = plot(p5, p6, p7, p8, layout = (4, 1), size = (800, 600))
display(p_combined2)

# Phase portrait plots
p9 = plot(x1, v1, xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", legend = false, seriescolor = 1, palette = :Dark2_5)
p10 = plot(x2, v2, xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", legend = false, seriescolor = 2, palette = :Dark2_5)
p11 = plot(x3, v3, xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", legend = false, seriescolor = 3, palette = :Dark2_5)
p12 = plot(x4, v4, xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined3 = plot(p9, p10, p11, p12, layout = (2, 2), size = (800, 800))
display(p_combined3)

# ---------------------------------- Solve Analytical Model: Additional Physics Model ---------------------------------

# Define a wrapper function
function hidden_physics_model_wrapper!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    hidden_model!(dz, z, p, t, current_acceleration)
end

# Define and solve the ODE problem
eqn2 = ODEProblem(hidden_physics_model_wrapper!, u0, tspan, p_true)

# Solve the system using Rosenbrock23 solver
sol2 = solve(eqn2, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7) # saveat=tspan[1]:0.001:tspan[2]

# Verify the solution structure
println("Type of sol.u: ", typeof(sol2.u))
println("Size of sol.u: ", size(sol2.u))
println("Solver status: ", sol2.retcode)

# ----------------------------------------- Plotting: Additional Physics Model -----------------------------------------

# Plot the related terms
x1 = [u[1] for u in sol2.u]
v1 = [u[2] for u in sol2.u]
x2 = [u[3] for u in sol2.u]
v2 = [u[4] for u in sol2.u]
x3 = [u[5] for u in sol2.u]
v3 = [u[6] for u in sol2.u]
x4 = [u[7] for u in sol2.u]
v4 = [u[8] for u in sol2.u]
u2 = sol2.u
t2 = sol2.t

# Displacement plots
p1 = plot(sol.t, x1, ylabel = "x1 (m)", title = "Displacement", legend = false, seriescolor = 1, palette = :Dark2_5)
p2 = plot(sol.t, x2, ylabel = "x2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p3 = plot(sol.t, x3, ylabel = "x3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p4 = plot(sol.t, x4, xlabel = "Time (s)", ylabel = "x4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined1 = plot(p1, p2, p3, p4, layout = (4, 1), size = (800, 600))
display(p_combined1)

# Velocity plots
p5 = plot(sol.t, v1, ylabel = "v1 (m)", title = "Velocity", legend = false, seriescolor = 1, palette = :Dark2_5)
p6 = plot(sol.t, v2, ylabel = "v2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p7 = plot(sol.t, v3, ylabel = "v3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p8 = plot(sol.t, v4, xlabel = "Time (s)", ylabel = "v4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined2 = plot(p5, p6, p7, p8, layout = (4, 1), size = (800, 600))
display(p_combined2)

# Phase portrait plots
p9 = plot(x1, v1, xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", legend = false, seriescolor = 1, palette = :Dark2_5)
p10 = plot(x2, v2, xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", legend = false, seriescolor = 2, palette = :Dark2_5)
p11 = plot(x3, v3, xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", legend = false, seriescolor = 3, palette = :Dark2_5)
p12 = plot(x4, v4, xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined3 = plot(p9, p10, p11, p12, layout = (2, 2), size = (800, 800))
display(p_combined3)

# ----------------------------------------- Plotting: Compare Models -----------------------------------------

# Displacement plots
p1 = plot(sol1.t, [u1[:,1] u2[:,1]], ylabel = "x1 (m)", title = "Displacement Comparison", label = ["HM" "HM + Physics"], seriescolor = 1, palette = :Dark2_5)
p2 = plot(sol1.t, [u1[:,3] u2[:,3]], ylabel = "x2 (m)", label = ["HM" "HM + Physics"], seriescolor = 2, palette = :Dark2_5)
p3 = plot(sol1.t, [u1[:,5] u2[:,5]], ylabel = "x3 (m)", label = ["HM" "HM + Physics"], seriescolor = 3, palette = :Dark2_5)
p4 = plot(sol1.t, [u1[:,7] u2[:,7]], xlabel = "Time (s)", ylabel = "x4 (m)", label = ["HM" "HM + Physics"], seriescolor = 4, palette = :Dark2_5)
p_combined1 = plot(p1, p2, p3, p4, layout = (4, 1), size = (800, 600))
display(p_combined1)

# Velocity plots
p5 = plot(sol.t, v1, ylabel = "v1 (m)", title = "Velocity", legend = false, seriescolor = 1, palette = :Dark2_5)
p6 = plot(sol.t, v2, ylabel = "v2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p7 = plot(sol.t, v3, ylabel = "v3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p8 = plot(sol.t, v4, xlabel = "Time (s)", ylabel = "v4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined2 = plot(p5, p6, p7, p8, layout = (4, 1), size = (800, 600))
display(p_combined2)

# Phase portrait plots
p9 = plot(x1, v1, xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", legend = false, seriescolor = 1, palette = :Dark2_5)
p10 = plot(x2, v2, xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", legend = false, seriescolor = 2, palette = :Dark2_5)
p11 = plot(x3, v3, xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", legend = false, seriescolor = 3, palette = :Dark2_5)
p12 = plot(x4, v4, xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined3 = plot(p9, p10, p11, p12, layout = (2, 2), size = (800, 800))
display(p_combined3)
