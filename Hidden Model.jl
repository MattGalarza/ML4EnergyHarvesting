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
    # Mass 1: Added nonlinear coulomb friction damping term, reduced k1 and increased k2
    F1_spring = -(0.95 * k1 + 1.17 * k2) * x1 + 1.17 * k2 * x2 
    F1_damping = -c1 * sign(v1) * (1 + 0.3 * abs(x1)) + c2 * (x2 - x1) * (v2 - v1) * abs(v2 - v1)
    F1_total = F1_spring + F1_damping
    
    # Mass 2: Increased k2 and reduced c3
    F2_spring = 1.17 * k2 * x1 - (1.17 * k2 + k3) * x2 + k3 * x3
    F2_damping = c2 * (x1 - x2) * (v1 - v2) * abs(v1 - v2) + 0.83 * c3 * (v3 - v2)
    F2_total = F2_spring + F2_damping
    
    # Mass 3: Reduced c3 and c4
    F3_spring = k3 * (x2 - x3) + k4 * (x4 - x3)^3
    F3_damping = 0.83 * c3 * (v2 - v3) + 0.75 * c4 * (v4 - v3)
    F3_total = F3_spring + F3_damping
    
    # Mass 4: Reduced c4
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
p_true = [12.0, 8.5, 10.0, 9.25, 125.0, 80.0, 93.0, 75.0, 0.5, 3.75, 4.5, 2.0]                
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
sol1 = solve(eqn1, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7, saveat=tspan[1]:0.00025:tspan[2])

# Verify the solution structure
println("Type of sol.u: ", typeof(sol1.u))
println("Size of sol.u: ", size(sol1.u))
println("Solver status: ", sol1.retcode)

# ----------------------------------------- Plotting: Hidden Model -----------------------------------------

# Plot the related terms
x1_1 = [u[1] for u in sol1.u]
v1_1 = [u[2] for u in sol1.u]
x2_1 = [u[3] for u in sol1.u]
v2_1 = [u[4] for u in sol1.u]
x3_1 = [u[5] for u in sol1.u]
v3_1 = [u[6] for u in sol1.u]
x4_1 = [u[7] for u in sol1.u]
v4_1 = [u[8] for u in sol1.u]
u1 = sol1.u
t1 = sol1.t

# Displacement plots
p1 = plot(t1, x1_1, ylabel = "x1 (m)", title = "Displacement", legend = false, seriescolor = 1, palette = :Dark2_5)
p2 = plot(t1, x2_1, ylabel = "x2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p3 = plot(t1, x3_1, ylabel = "x3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p4 = plot(t1, x4_1, xlabel = "Time (s)", ylabel = "x4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined1 = plot(p1, p2, p3, p4, layout = (4, 1), size = (800, 600))
display(p_combined1)

# Velocity plots
p5 = plot(t1, v1_1, ylabel = "v1 (m)", title = "Velocity", legend = false, seriescolor = 1, palette = :Dark2_5)
p6 = plot(t1, v2_1, ylabel = "v2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p7 = plot(t1, v3_1, ylabel = "v3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p8 = plot(t1, v4_1, xlabel = "Time (s)", ylabel = "v4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined2 = plot(p5, p6, p7, p8, layout = (4, 1), size = (800, 600))
display(p_combined2)

# Phase portrait plots
p9 = plot(x1_1, v1_1, xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", legend = false, seriescolor = 1, palette = :Dark2_5)
p10 = plot(x2_1, v2_1, xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", legend = false, seriescolor = 2, palette = :Dark2_5)
p11 = plot(x3_1, v3_1, xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", legend = false, seriescolor = 3, palette = :Dark2_5)
p12 = plot(x4_1, v4_1, xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined3 = plot(p9, p10, p11, p12, layout = (2, 2), size = (800, 800))
display(p_combined3)

# ---------------------------------- Solve Analytical Model: Additional Physics Model ---------------------------------

# Define a wrapper function
function hidden_physics_model_wrapper!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    hidden_physics_model!(dz, z, p, t, current_acceleration)
end

# Define and solve the ODE problem
eqn2 = ODEProblem(hidden_physics_model_wrapper!, u0, tspan, p_true)

# Solve the system using Rosenbrock23 solver
sol2 = solve(eqn2, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7, saveat=tspan[1]:0.00025:tspan[2])

# Verify the solution structure
println("Type of sol.u: ", typeof(sol2.u))
println("Size of sol.u: ", size(sol2.u))
println("Solver status: ", sol2.retcode)

# ----------------------------------------- Plotting: Additional Physics Model -----------------------------------------

# Plot the related terms
x1_2 = [u[1] for u in sol2.u]
v1_2 = [u[2] for u in sol2.u]
x2_2 = [u[3] for u in sol2.u]
v2_2 = [u[4] for u in sol2.u]
x3_2 = [u[5] for u in sol2.u]
v3_2 = [u[6] for u in sol2.u]
x4_2 = [u[7] for u in sol2.u]
v4_2 = [u[8] for u in sol2.u]
u2 = sol2.u
t2 = sol2.t

# Displacement plots
p13 = plot(t2, x1_2, ylabel = "x1 (m)", title = "Displacement", legend = false, seriescolor = 1, palette = :Dark2_5)
p14 = plot(t2, x2_2, ylabel = "x2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p15 = plot(t2, x3_2, ylabel = "x3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p16 = plot(t2, x4_2, xlabel = "Time (s)", ylabel = "x4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined4 = plot(p13, p14, p15, p16, layout = (4, 1), size = (800, 600))
display(p_combined4)

# Velocity plots
p17 = plot(t2, v1_2, ylabel = "v1 (m)", title = "Velocity", legend = false, seriescolor = 1, palette = :Dark2_5)
p18 = plot(t2, v2_2, ylabel = "v2 (m)", legend = false, seriescolor = 2, palette = :Dark2_5)
p19 = plot(t2, v3_2, ylabel = "v3 (m)", legend = false, seriescolor = 3, palette = :Dark2_5)
p20 = plot(t2, v4_2, xlabel = "Time (s)", ylabel = "v4 (m)", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined5 = plot(p17, p18, p19, p20, layout = (4, 1), size = (800, 600))
display(p_combined5)

# Phase portrait plots
p21 = plot(x1_2, v1_2, xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", legend = false, seriescolor = 1, palette = :Dark2_5)
p22 = plot(x2_2, v2_2, xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", legend = false, seriescolor = 2, palette = :Dark2_5)
p23 = plot(x3_2, v3_2, xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", legend = false, seriescolor = 3, palette = :Dark2_5)
p24 = plot(x4_2, v4_2, xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", legend = false, seriescolor = 4, palette = :Dark2_5)
p_combined6 = plot(p21, p22, p23, p24, layout = (2, 2), size = (800, 800))
display(p_combined6)

# ----------------------------------------- Plotting: Compare Models -----------------------------------------

# Displacement plots
p25 = plot(t1, [x1_1 x1_2], ylabel = "x1 (m)", title = "Displacement Comparison", label = ["HM" "HM + Physics"], seriescolor = [1 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p26 = plot(t1, [x2_1 x2_2], ylabel = "x2 (m)", label = ["HM" "HM + Physics"], seriescolor = [2 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p27 = plot(t1, [x3_1 x3_2], ylabel = "x3 (m)", label = ["HM" "HM + Physics"], seriescolor = [3 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p28 = plot(t1, [x4_1 x4_2], xlabel = "Time (s)", ylabel = "x4 (m)", label = ["HM" "HM + Physics"], seriescolor = [4 :black],
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p_combined7 = plot(p25, p26, p27, p28, layout = (4, 1), size = (800, 600))
display(p_combined7)

# Velocity plots
p29 = plot(t1, [v1_1 v1_2], ylabel = "v1 (m)", title = "Velocity Comparison", label = ["HM" "HM + Physics"], seriescolor = [1 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p30 = plot(t1, [v2_1 v2_2], ylabel = "v2 (m)", label = ["HM" "HM + Physics"], seriescolor = [2 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p31 = plot(t1, [v3_1 v3_2], ylabel = "v3 (m)", label = ["HM" "HM + Physics"], seriescolor = [3 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p32 = plot(t1, [v4_1 v4_2], xlabel = "Time (s)", ylabel = "v4 (m)", label = ["HM" "HM + Physics"], seriescolor = [4 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p_combined8 = plot(p29, p30, p31, p32, layout = (4, 1), size = (800, 600))
display(p_combined8)

# Phase portrait plots
p33 = plot([x1_1 x1_2], [v1_1 v1_2], xlabel = "x1 (m)", ylabel = "v1 (m)", title = "M1 Phase Portrait", label = ["HM" "HM + Physics"], seriescolor = [1 :black], linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p34 = plot([x2_1 x2_2], [v2_1 v2_2], xlabel = "x2 (m)", ylabel = "v2 (m)", title = "M2 Phase Portrait", label = ["HM" "HM + Physics"], seriescolor = [2 :black], linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p35 = plot([x3_1 x3_2], [v3_1 v3_2], xlabel = "x3 (m)", ylabel = "v3 (m)", title = "M3 Phase Portrait", label = ["HM" "HM + Physics"], seriescolor = [3 :black], linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p36 = plot([x4_1 x4_2], [v4_1 v4_2], xlabel = "x4 (m)", ylabel = "v4 (m)", title = "M4 Phase Portrait", label = ["HM" "HM + Physics"], seriescolor = [4 :black], linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p_combined9 = plot(p33, p34, p35, p36, layout = (2, 2), size = (800, 800))
display(p_combined9)

# ----------------------------------------- Optional: Add noise -----------------------------------------



# ------------------------------------ State Reconstruction from Taken's Embedding Theorem --------------------------------



# ----------------------------------------- Data Normalization -----------------------------------------

# Define a function to normalize data
function normalizer(data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return new_min .+ (data .- min_val) .* (new_max - new_min) ./ (max_val - min_val)
end

# Define a function to denormalize data
function denormalizer(norm_data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return min_val .+ (norm_data .- new_min) .* (max_val - min_val) ./ (new_max - new_min)
end

x1_min, x1_max = -0.1, 0.1          
v1_min, v1_max = -1.0, 1.0          
x2_min, x2_max = -0.02, 0.02  
v2_min, v2_max = -0.2, 0.2     
x3_min, x3_max = -0.02, 0.02      
v3_min, v3_max = -0.1, 0.1         
x4_min, x4_max = -0.002, 0.002     
v4_min, v4_max = -0.01, 0.01     

# Store normalization bounds
norm_bounds = (
    x1 = (x1_min, x1_max),
    v1 = (v1_min, v1_max),
    x2 = (x2_min, x2_max),
    v2 = (v2_min, v2_max),
    x3 = (x3_min, x3_max),
    v3 = (v3_min, v3_max),
    x4 = (x4_min, x4_max),
    v4 = (v4_min, v4_max),
    Fext = (-15.0, 15.0)
)

# Normalize solution data
# Hidden model (sol1)
x1_1_norm = normalizer(x1_1, norm_bounds.x1...)
v1_1_norm = normalizer(v1_1, norm_bounds.v1...)
x2_1_norm = normalizer(x2_1, norm_bounds.x2...)
v2_1_norm = normalizer(v2_1, norm_bounds.v2...)
x3_1_norm = normalizer(x3_1, norm_bounds.x3...)
v3_1_norm = normalizer(v3_1, norm_bounds.v3...)
x4_1_norm = normalizer(x4_1, norm_bounds.x4...)
v4_1_norm = normalizer(v4_1, norm_bounds.v4...)

# Hidden physics model (sol2)
x1_2_norm = normalizer(x1_2, norm_bounds.x1...)
v1_2_norm = normalizer(v1_2, norm_bounds.v1...)
x2_2_norm = normalizer(x2_2, norm_bounds.x2...)
v2_2_norm = normalizer(v2_2, norm_bounds.v2...)
x3_2_norm = normalizer(x3_2, norm_bounds.x3...)
v3_2_norm = normalizer(v3_2, norm_bounds.v3...)
x4_2_norm = normalizer(x4_2, norm_bounds.x4...)
v4_2_norm = normalizer(v4_2, norm_bounds.v4...)
Fext_val = [Fext_input(tt) for tt in t1]
Fext_norm = normalizer(Fext_val, norm_bounds.Fext...)

# Create normalized state matrices
u1_norm = hcat(x1_1_norm, v1_1_norm, x2_1_norm, v2_1_norm, x3_1_norm, v3_1_norm, x4_1_norm, v4_1_norm)
u2_norm = hcat(x1_2_norm, v1_2_norm, x2_2_norm, v2_2_norm, x3_2_norm, v3_2_norm, x4_2_norm, v4_2_norm)

# Displacement plots (normalized)
p25 = plot(t1, [x1_1_norm x1_2_norm], ylabel = "x1 (m)", title = "Displacement (Normalized)", label = ["HM" "HM + Physics"], seriescolor = [1 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p26 = plot(t1, [x2_1_norm x2_2_norm], ylabel = "x2 (m)", label = ["HM" "HM + Physics"], seriescolor = [2 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p27 = plot(t1, [x3_1_norm x3_2_norm], ylabel = "x3 (m)", label = ["HM" "HM + Physics"], seriescolor = [3 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p28 = plot(t1, [x4_1_norm x4_2_norm], xlabel = "Time (s)", ylabel = "x4 (m)", label = ["HM" "HM + Physics"], seriescolor = [4 :black],
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p_combined7 = plot(p25, p26, p27, p28, layout = (4, 1), size = (800, 600))
display(p_combined7)

# Velocity plots (normalized)
p29 = plot(t1, [v1_1_norm v1_2_norm], ylabel = "v1 (m)", title = "Velocity (Normalized)", label = ["HM" "HM + Physics"], seriescolor = [1 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p30 = plot(t1, [v2_1_norm v2_2_norm], ylabel = "v2 (m)", label = ["HM" "HM + Physics"], seriescolor = [2 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p31 = plot(t1, [v3_1_norm v3_2_norm], ylabel = "v3 (m)", label = ["HM" "HM + Physics"], seriescolor = [3 :black], linestyle = [:solid :dash], 
            palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p32 = plot(t1, [v4_1_norm v4_2_norm], xlabel = "Time (s)", ylabel = "v4 (m)", label = ["HM" "HM + Physics"], seriescolor = [4 :black], 
            linestyle = [:solid :dash], palette = :Dark2_5, legend = :topright, legendfontsize = 6)
p_combined8 = plot(p29, p30, p31, p32, layout = (4, 1), size = (800, 600))
display(p_combined8)

# ----------------------------------------- Neural ODE/UDE -----------------------------------------

# Deep NN chain, ReLu activation function
const U = Lux.Chain(Lux.Dense(9, 32, relu), # 9 inputs: 8 states + 1 input force
                    Lux.Dense(32, 64, relu),
                    Lux.Dense(64, 32, relu),
                    Lux.Dense(32, 8) # 8 outputs for state corrections
) 

# Initialize NN parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Combine physical and neural parameters into a ComponentArray
p_ude = ComponentArray(physical = p_true, neural = ComponentArray(nn_params))

# Define UDE dynamics with normalized states
function ude_dynamics!(du, u, p, t)
    # Extract parameters
    p_phys = p.physical
    p_nn = p.neural
    
    # Get current external force (normalized)
    Fext_current = Fext_input(t)
    Fext_norm_current = normalizer(Fext_current, norm_bounds.Fext...)
    
    # Denormalize states for analytical model
    u_denorm = [
        denormalizer(u[1], norm_bounds.x1...),  
        denormalizer(u[2], norm_bounds.v1...),  
        denormalizer(u[3], norm_bounds.x2...),
        denormalizer(u[4], norm_bounds.v2...),  
        denormalizer(u[5], norm_bounds.x3...), 
        denormalizer(u[6], norm_bounds.v3...),  
        denormalizer(u[7], norm_bounds.x4...), 
        denormalizer(u[8], norm_bounds.v4...)  
    ]
    
    # Get analytical model derivatives
    du_analytical = similar(du) # zero(u)
    hidden_model!(du_analytical, u_denorm, p_phys, t, Fext_current)
    
    # Normalize the analytical derivatives
    correction_scales = [
        (norm_bounds.x1[2] - norm_bounds.x1[1]),
        (norm_bounds.v1[2] - norm_bounds.v1[1]),  
        (norm_bounds.x2[2] - norm_bounds.x2[1]),  
        (norm_bounds.v2[2] - norm_bounds.v2[1]),   
        (norm_bounds.x3[2] - norm_bounds.x3[1]), 
        (norm_bounds.v3[2] - norm_bounds.v3[1]),   
        (norm_bounds.x4[2] - norm_bounds.x4[1]),  
        (norm_bounds.v4[2] - norm_bounds.v4[1])   
    ]
    du_analytical_norm = du_analytical ./ correction_scales
    
    # Get neural network corrections
    nn_input = vcat(u, Fext_norm_current)
    nn_corrections = U(nn_input, p_nn, _st)[1]
    
    # Combine analytical model + neural corrections
    du .= du_analytical_norm .+ nn_corrections
end

# Define the UDE problem with normalized initial conditions
u0_norm = u0 # All ICs are zero so already normalized

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0_norm, tspan, p_ude)

# Create training data from solutions
t_data = t1
X_data = u2_norm'

# Prediction function
function predict(p, X = u0_norm, T = t_data)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = p)
    pred = solve(_prob, Tsit5(), saveat = T, abstol = abstol, reltol = reltol, maxiters = 1e7, verbose = false)
    # pred = solve(_prob, Rodas4(), saveat = T, abstol = abstol, reltol = reltol, sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), verbose = false)
    return Array(pred)
end

# Loss function
function loss(p)
    u_pred = predict(p)
    if any(isnan, u_pred)
        return Inf
    end
    scale_factor = 1
    return scale_factor * mean(abs2, X_data .- u_pred)
end

# Simple callback
losses = Float64[]
callback = function(state, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        # Print epoch losses
        println("Iteration $(length(losses)): Loss = $(l)")
    end
    return false
end

# Optimization setup
optf = OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_ude)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 20)
println("\nOptimization complete!")
println("Final loss: ", losses[end])

# Plot final results
println("\nGenerating final plots...")
final_pred = predict(res.u)

# Extract time data
time_data = t_data

# DEBUG: Check dimensions first
println("Debugging dimensions:")
println("Size of time_data (t_data): ", size(time_data))
println("Size of X_data: ", size(X_data))
println("Size of final_pred: ", size(final_pred))
println("Length of time_data: ", length(time_data))
println("Number of time points in X_data: ", size(X_data, 2))
println("Number of time points in final_pred: ", size(final_pred, 2))

# Fix dimension mismatch by finding the minimum length
n_time_points = min(length(time_data), size(X_data, 2), size(final_pred, 2))
println("Using n_time_points: ", n_time_points)

# Trim all arrays to the same length
time_plot = time_data[1:n_time_points]
X_plot = X_data[:, 1:n_time_points]
pred_plot = final_pred[:, 1:n_time_points]

# Plot comparison for a specific state (e.g., state 4 which might be velocity)
p1 = plot(time_plot, X_plot[4,:], label = "Training Data", xlabel = "Time (s)", ylabel = "Normalized State 4 (v2)")
plot!(p1, time_plot, pred_plot[4,:], label = "UDE Prediction")
title!(p1, "UDE Performance - State 4")
display(p1)

# Zoomed in view (last part of simulation)
zoom_start = max(1, n_time_points ÷ 3)  # Last 2/3 of the data
p2 = plot(time_plot[zoom_start:end], X_plot[4,zoom_start:end], label = "Training Data", 
         xlabel = "Time (s)", ylabel = "Normalized State 4 (v2)")
plot!(p2, time_plot[zoom_start:end], pred_plot[4,zoom_start:end], label = "UDE Prediction")
title!(p2, "UDE Performance - State 4 (Zoomed)")
display(p2)

# State labels for your 8-state system
println("\nPlotting trajectories comparison...")
p_traj = plot(layout=(4,2), size=(1200,1000))
state_labels = ["x1", "v1", "x2", "v2", "x3", "v3", "x4", "v4"]

# Plot each state
for i in 1:8
    plot!(p_traj[i], time_plot, X_plot[i,:], label = "Training Data", 
          xlabel = "Time (s)", ylabel = "Normalized "*state_labels[i])
    plot!(p_traj[i], time_plot, pred_plot[i,:], label = "UDE Prediction")
    title!(p_traj[i], state_labels[i])
end
suptitle = plot(title = "State Trajectories: Training Data vs UDE", grid = false, showaxis = false, bottom_margin = -50Plots.px)
p_combined = plot(suptitle, p_traj, layout = @layout([A{0.01h}; B]))
display(p_combined)

# Loss vs Epochs
p3 = plot(1:length(losses), losses, xlabel = "Epoch", ylabel = "Loss", label = "Training Loss")
title!(p3, "Loss vs Epochs")
display(p3)

# Log scale loss plot (if helpful)
p4 = plot(1:length(losses), losses, xlabel = "Epoch", ylabel = "Loss (log scale)", 
         label = "Training Loss", yscale = :log10)
title!(p4, "Loss vs Epochs (Log Scale)")
display(p4)

# Prediction vs actual for multiple key states
p5 = plot(layout=(2,2), size=(1000,800))
key_states = [1, 3, 5, 7]  # Position states
key_labels = ["x1", "x2", "x3", "x4"]

for (idx, state_idx) in enumerate(key_states)
    plot!(p5[idx], time_plot, X_plot[state_idx,:], label = "Training Data", 
          xlabel = "Time (s)", ylabel = "Normalized "*key_labels[idx])
    plot!(p5[idx], time_plot, pred_plot[state_idx,:], label = "UDE Prediction")
    title!(p5[idx], key_labels[idx])
end
suptitle2 = plot(title = "Key Position States: Training Data vs UDE Prediction", grid = false, showaxis = false, bottom_margin = -50Plots.px)
p_combined2 = plot(suptitle2, p5, layout = @layout([A{0.01h}; B]))
display(p_combined2)

# Compute and plot prediction errors for each state
println("\nComputing prediction errors...")
p6 = plot(layout=(4,2), size=(1200,1000))

for i in 1:8
    error = pred_plot[i,:] .- X_plot[i,:]
    plot!(p6[i], time_plot, error, xlabel = "Time (s)", ylabel = "Error", 
          label = "Prediction Error", color = :red)
    title!(p6[i], state_labels[i]*" Error")
end
suptitle3 = plot(title = "Prediction Errors for All States", grid = false, showaxis = false, bottom_margin = -50Plots.px)
p_combined3 = plot(suptitle3, p6, layout = @layout([A{0.01h}; B]))
display(p_combined3)

# Compute and plot L2 norm error across all states
l2_errors = zeros(n_time_points)
for t in 1:n_time_points
    # Compute L2 norm error across all states at each time point
    error_vector = pred_plot[:,t] .- X_plot[:,t]
    l2_errors[t] = sqrt(sum(error_vector.^2))
end

p7 = plot(time_plot, l2_errors, xlabel = "Time (s)", ylabel = "L2 Norm Error", 
         label = "State Space Error", color = :purple, linewidth = 2)
title!(p7, "L2 Norm Error Across All States")
display(p7)

# Individual state errors (RMS)
state_rms_errors = zeros(8)
state_max_errors = zeros(8)
for i in 1:8
    error = pred_plot[i,:] .- X_plot[i,:]
    state_rms_errors[i] = sqrt(mean(error.^2))
    state_max_errors[i] = maximum(abs.(error))
end

# Bar plot of RMS errors by state
p8 = bar(state_labels, state_rms_errors, xlabel = "State", ylabel = "RMS Error", 
        title = "RMS Error by State", legend = false, color = :orange)
display(p8)

# Bar plot of Max errors by state
p9 = bar(state_labels, state_max_errors, xlabel = "State", ylabel = "Max Absolute Error", 
        title = "Maximum Absolute Error by State", legend = false, color = :red)
display(p9)

# Print comprehensive error statistics
println("\nError Statistics:")
println("Overall L2 Error:")
println("  Mean L2 Error: ", mean(l2_errors))
println("  Max L2 Error: ", maximum(l2_errors))
println("  Final L2 Error: ", l2_errors[end])

println("\nPer-State RMS Errors:")
for i in 1:8
    println("  $(state_labels[i]): ", round(state_rms_errors[i], digits=6))
end

println("\nPer-State Max Absolute Errors:")
for i in 1:8
    println("  $(state_labels[i]): ", round(state_max_errors[i], digits=6))
end

println("\nFinal Training Loss: ", losses[end])
println("Total Training Iterations: ", length(losses))