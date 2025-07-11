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




# ------------------------------------ State Reconstruction from Takens Embedding --------------------------------

using LinearAlgebra, Statistics

# Function to find optimal time delay with better criteria
function find_optimal_delay(x, max_delay=200)
    """
    Find optimal delay where autocorrelation drops significantly
    """
    autocorr = [cor(x[1:end-k], x[k+1:end]) for k in 1:max_delay]
    
    # Find where autocorrelation drops below threshold
    threshold = 0.3  # Much lower threshold than before
    for i in 1:length(autocorr)
        if autocorr[i] < threshold
            return i
        end
    end
    
    # If no drop below threshold, find first local minimum
    for i in 2:length(autocorr)-1
        if autocorr[i] < autocorr[i-1] && autocorr[i] < autocorr[i+1]
            return i
        end
    end
    
    return div(max_delay, 4)  # Default fallback
end

# Create improved embedding
function create_embedding(x, m, τ)
    """
    Create time delay embedding with proper indexing
    """
    N = length(x)
    M = N - (m-1) * τ
    
    if M <= 100  # Need sufficient data points
        error("Insufficient data for embedding with m=$m, τ=$τ")
    end
    
    embedding = zeros(M, m)
    for i in 1:M
        for j in 1:m
            embedding[i, j] = x[i + (j-1) * τ]
        end
    end
    
    return embedding
end

# State reconstruction using local linear approximation
function reconstruct_states(x4_embedding, true_states, train_fraction=0.7)
    """
    Reconstruct all states from x4 embedding using local linear approximation
    x4_embedding: [N × m] embedding matrix
    true_states: [N × 8] matrix of true states [x1,v1,x2,v2,x3,v3,x4,v4]
    """
    N, m = size(x4_embedding)
    n_states = size(true_states, 2)
    
    # Split into training and testing
    train_size = Int(round(train_fraction * N))
    train_idx = 1:train_size
    test_idx = (train_size+1):N
    
    # Training data
    X_train = x4_embedding[train_idx, :]
    Y_train = true_states[train_idx, :]
    
    # Testing data  
    X_test = x4_embedding[test_idx, :]
    Y_test = true_states[test_idx, :]
    
    # Linear reconstruction (least squares)
    # Y = X * A + b  →  A = (X'X)^(-1) X'Y
    X_train_aug = [X_train ones(length(train_idx), 1)]  # Add bias term
    
    try
        A = X_train_aug \ Y_train  # Solve least squares
        
        # Predict on test set
        X_test_aug = [X_test ones(length(test_idx), 1)]
        Y_pred = X_test_aug * A
        
        # Calculate reconstruction errors
        errors = zeros(n_states)
        r_squared = zeros(n_states)
        
        for i in 1:n_states
            mse = mean((Y_test[:, i] - Y_pred[:, i]).^2)
            var_true = var(Y_test[:, i])
            errors[i] = sqrt(mse)
            r_squared[i] = 1 - mse / var_true
        end
        
        return (
            predictions = Y_pred,
            true_values = Y_test,
            reconstruction_matrix = A,
            rmse_errors = errors,
            r_squared = r_squared,
            test_size = length(test_idx)
        )
        
    catch e
        println("Linear reconstruction failed: $e")
        return nothing
    end
end

# ------------------------------------ Apply to Both Models --------------------------------

println("=== Improved State Reconstruction Analysis ===")

# Use your actual time array from the solutions
t_actual = t1  # or sol1.t - your actual time array from saveat
dt_actual = t_actual[2] - t_actual[1]  # Should be 0.00025

println("Using actual time step: dt = $dt_actual seconds")

# Find better time delay
τ_new_1 = find_optimal_delay(x4_1, 300)
τ_new_2 = find_optimal_delay(x4_2, 300)
τ_opt = max(τ_new_1, τ_new_2)

println("Improved time delay: τ = $τ_opt samples")
println("Time delay: $(round(τ_opt * dt_actual, digits=4)) seconds")

# Test different embedding dimensions
embedding_dims = [3, 5, 8, 12]
reconstruction_results = Dict()

for m in embedding_dims
    println("\n--- Testing $(m)D embedding ---")
    
    try
        # Create embeddings
        emb_1 = create_embedding(x4_1, m, τ_opt)
        emb_2 = create_embedding(x4_2, m, τ_opt)
        
        # Prepare true state matrices (aligned with embedding)
        start_idx = 1
        end_idx = size(emb_1, 1)
        
        true_states_1 = hcat(x1_1[start_idx:end_idx], v1_1[start_idx:end_idx],
                            x2_1[start_idx:end_idx], v2_1[start_idx:end_idx],
                            x3_1[start_idx:end_idx], v3_1[start_idx:end_idx],
                            x4_1[start_idx:end_idx], v4_1[start_idx:end_idx])
                            
        true_states_2 = hcat(x1_2[start_idx:end_idx], v1_2[start_idx:end_idx],
                            x2_2[start_idx:end_idx], v2_2[start_idx:end_idx],
                            x3_2[start_idx:end_idx], v3_2[start_idx:end_idx],
                            x4_2[start_idx:end_idx], v4_2[start_idx:end_idx])
        
        # Reconstruct states
        result_1 = reconstruct_states(emb_1, true_states_1)
        result_2 = reconstruct_states(emb_2, true_states_2)
        
        if result_1 !== nothing && result_2 !== nothing
            reconstruction_results[m] = (hm = result_1, physics = result_2)
            
            state_names = ["x1", "v1", "x2", "v2", "x3", "v3", "x4", "v4"]
            
            println("Hidden Model - R² scores:")
            for (i, name) in enumerate(state_names)
                println("  $name: $(round(result_1.r_squared[i], digits=4))")
            end
            
            println("Hidden Physics Model - R² scores:")
            for (i, name) in enumerate(state_names)
                println("  $name: $(round(result_2.r_squared[i], digits=4))")
            end
            
            # Calculate embedding quality (correlation between dimensions)
            corr_1 = cor(emb_1)
            corr_2 = cor(emb_2)
            max_corr_1 = maximum(abs.(corr_1 - I))
            max_corr_2 = maximum(abs.(corr_2 - I))
            
            println("Embedding correlations - HM: $(round(max_corr_1, digits=4)), Physics: $(round(max_corr_2, digits=4))")
        end
        
    catch e
        println("Failed for $(m)D: $e")
    end
end

# ------------------------------------ Visualization of Best Reconstruction --------------------------------

# Find best embedding dimension (highest average R²)
best_m = 0
best_avg_r2 = -Inf

for (m, results) in reconstruction_results
    avg_r2_1 = mean(results.hm.r_squared[1:7])  # Exclude x4 since it's the observable
    avg_r2_2 = mean(results.physics.r_squared[1:7])
    avg_r2 = (avg_r2_1 + avg_r2_2) / 2
    
    if avg_r2 > best_avg_r2
        best_avg_r2 = avg_r2
        best_m = m
    end
end

if best_m > 0
    println("\n=== Best Reconstruction: $(best_m)D Embedding ===")
    
    best_results = reconstruction_results[best_m]
    
    # Get the correct test indices based on embedding size
    N_predictions = size(best_results.hm.predictions, 1)
    train_size = Int(round(0.7 * N_predictions))
    test_size = N_predictions - train_size
    
    # Create time array that exactly matches the test data size
    t_test = collect(range(0.0, 20.0, length=test_size))
    
    println("Data sizes - Predictions: $N_predictions, Train: $train_size, Test: $test_size")
    println("Time array size: $(length(t_test))")
    
    # Create comprehensive comparison plots
    state_names = ["x1", "v1", "x2", "v2", "x3", "v3", "x4", "v4"]
    units = ["(m)", "(m/s)", "(m)", "(m/s)", "(m)", "(m/s)", "(m)", "(m/s)"]
    
    # Plot reconstructed vs true for most interesting states (excluding x4)
    interesting_states = [1, 3, 5, 7]  # x1, x2, x3, x4 positions
    
    p_recon = plot(layout=(2,2), size=(1000, 800))
    
    for (plot_idx, state_idx) in enumerate(interesting_states)
        # True values (test portion only)
        true_1 = best_results.hm.true_values[:, state_idx]
        true_2 = best_results.physics.true_values[:, state_idx]
        
        # Reconstructed values (test portion only)
        recon_1 = best_results.hm.predictions[:, state_idx]
        recon_2 = best_results.physics.predictions[:, state_idx]
        
        # Verify sizes match
        println("State $(state_idx) - t_test: $(length(t_test)), true_1: $(length(true_1)), recon_1: $(length(recon_1))")
        
        # Ensure all arrays have the same length
        min_length = min(length(t_test), length(true_1), length(recon_1))
        
        plot!(p_recon[plot_idx], t_test[1:min_length], [true_1[1:min_length] recon_1[1:min_length]], 
              label=["HM True" "HM Reconstructed"],
              title="$(state_names[state_idx]) Reconstruction",
              xlabel="Time (s)", ylabel="$(state_names[state_idx]) $(units[state_idx])",
              seriescolor=[plot_idx :gray], linestyle=[:solid :dash])
              
        plot!(p_recon[plot_idx], t_test[1:min_length], [true_2[1:min_length] recon_2[1:min_length]], 
              label=["Physics True" "Physics Reconstructed"],
              seriescolor=[:black :red], linestyle=[:solid :dash])
    end
    
    display(p_recon)
    
    # Reconstruction error analysis
    p_errors = plot(layout=(2,2), size=(1000, 800))
    
    for (plot_idx, state_idx) in enumerate(interesting_states)
        error_1 = abs.(best_results.hm.true_values[:, state_idx] - best_results.hm.predictions[:, state_idx])
        error_2 = abs.(best_results.physics.true_values[:, state_idx] - best_results.physics.predictions[:, state_idx])
        
        # Ensure arrays match time array size
        min_length = min(length(t_test), length(error_1), length(error_2))
        
        plot!(p_errors[plot_idx], t_test[1:min_length], [error_1[1:min_length] error_2[1:min_length]],
              label=["HM Error" "Physics Error"],
              title="$(state_names[state_idx]) Reconstruction Error",
              xlabel="Time (s)", ylabel="Absolute Error $(units[state_idx])",
              seriescolor=[plot_idx :black])
    end
    
    display(p_errors)
    
    # Summary statistics
    println("\n=== Reconstruction Quality Summary ===")
    println("Embedding dimension: $(best_m)D")
    println("Time delay: τ = $τ_opt samples")
    println("Average R² (excluding x4):")
    
    avg_r2_hm = mean(best_results.hm.r_squared[1:7])
    avg_r2_physics = mean(best_results.physics.r_squared[1:7])
    
    println("  Hidden Model: $(round(avg_r2_hm, digits=4))")
    println("  Physics Model: $(round(avg_r2_physics, digits=4))")
    
    # Store results for further use
    reconstruction_summary = (
        best_dimension = best_m,
        optimal_delay = τ_opt,
        results = best_results,
        avg_r_squared = (hm = avg_r2_hm, physics = avg_r2_physics),
        test_time_range = (t_test[1], t_test[end])
    )
    
else
    println("No successful reconstructions found!")
end

println("\nState reconstruction analysis complete!")





# ----------------------------------------- Data Normalization -----------------------------------------

# Define a function to normalize data
function normalizer(data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return new_min .+ (data .- min_val) .* (new_max - new_min) ./ (max_val - min_val)
end

# Define a function to denormalize data
function denormalizer(norm_data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return min_val .+ (norm_data .- new_min) .* (max_val - min_val) ./ (new_max - new_min)
end

# Calculate min and max values from state data
x1_min, x1_max = min(minimum(x1_1), minimum(x1_2)), max(maximum(x1_1), maximum(x1_2))
v1_min, v1_max = min(minimum(v1_1), minimum(v1_2)), max(maximum(v1_1), maximum(v1_2))
x2_min, x2_max = min(minimum(x2_1), minimum(x2_2)), max(maximum(x2_1), maximum(x2_2))
v2_min, v2_max = min(minimum(v2_1), minimum(v2_2)), max(maximum(v2_1), maximum(v2_2))
x3_min, x3_max = min(minimum(x3_1), minimum(x3_2)), max(maximum(x3_1), maximum(x3_2))
v3_min, v3_max = min(minimum(v3_1), minimum(v3_2)), max(maximum(v3_1), maximum(v3_2))
x4_min, x4_max = min(minimum(x4_1), minimum(x4_2)), max(maximum(x4_1), maximum(x4_2))
v4_min, v4_max = min(minimum(v4_1), minimum(v4_2)), max(maximum(v4_1), maximum(v4_2))
Fext_val = [Fext_input(tt) for tt in t1]        
Fext_min, Fext_max = minimum(Fext_val), maximum(Fext_val)

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
    Fext = (Fext_min, Fext_max)
)

# Print bounds for verification
println("Calculated normalization bounds:")
println("x1: [$x1_min, $x1_max]")
println("v1: [$v1_min, $v1_max]")
println("x2: [$x2_min, $x2_max]")
println("v2: [$v2_min, $v2_max]")
println("x3: [$x3_min, $x3_max]")
println("v3: [$v3_min, $v3_max]")
println("x4: [$x4_min, $x4_max]")
println("v4: [$v4_min, $v4_max]")
println("Fext: [$Fext_min, $Fext_max]")

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
Fext_norm = normalizer(Fext_val, norm_bounds.Fext...)

# Create normalized state matrices
u1_norm = hcat(x1_1_norm, v1_1_norm, x2_1_norm, v2_1_norm, x3_1_norm, v3_1_norm, x4_1_norm, v4_1_norm)
u2_norm = hcat(x1_2_norm, v1_2_norm, x2_2_norm, v2_2_norm, x3_2_norm, v3_2_norm, x4_2_norm, v4_2_norm)

println()
println("Normalized state matrices shapes:")
println("Hidden model states: ", size(u1_norm))
println("Hidden physics states: ", size(u2_norm))
println("External force: ", size(Fext_norm))

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
p_combined = ComponentArray(physical = p_true, neural = ComponentArray(nn_params))
n_physical = length(p_true) # Store original structure for reference

# Define UDE dynamics with normalized states
function ude_dynamics!(du, u, p, t)
    # Extract parameters directly from ComponentArray
    p_current = p.physical # Physical parameters
    nn_params_current = p.neural # Neural network parameters
    
    # Get NN correction terms
    nn_correction = U(u, nn_params_current, _st)[1]
    
    # Add NN correction to the analytical model
    lorenz!(du, u, p_current, t)
    du .+= nn_correction

    # Denormalize states for analytical model
    u_denorm = [
        denormalizer(u[1], x1_min, x1_max),
        denormalizer(u[2], x1dot_min, x1dot_max),
        denormalizer(u[3], x2_min, x2_max),
        denormalizer(u[4], x2dot_min, x2dot_max),
        denormalizer(u[5], Qvar_min, Qvar_max),
        denormalizer(u[6], V_min, V_max)
    ]
    
    # Get analytical model derivatives using modified parameters
    du_model = similar(du)
    hidden_model!(du_model, u_denorm, p_current, t, current_acceleration)
    
    # Normalize the model derivatives
    correction_scales = [
        (x1_max - x1_min),
        (x1dot_max - x1dot_min),
        (x2_max - x2_min),
        (x2dot_max - x2dot_min),
        (Qvar_max - Qvar_min),
        (V_max - V_min)
    ]
    du_model_norm = du_model ./ correction_scales
    
    # Neural network prediction
    nn_input = vcat(u, acc_norm)
    nn_correction = U(nn_input, p.neural, _st)[1]
    
    # Combine normalized derivatives
    du .= du_model_norm + nn_correction
end

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p_combined)

# Prediction function
function predict(p, X = u0, T = t_data)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = p)
    pred = solve(_prob, Rosenbrock23(), saveat = T, abstol = 1e-6, reltol = 1e-6,
            sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
    return Array(pred)
end

# Loss function
function loss(p)
    u_pred = predict(p)
    return mean(abs2, X_noisy - u_pred)
end

# Simple callback
losses = Float64[]
callback = function (state, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        # Extract current model parameters for monitoring
        p_current = state.u.physical  # Access parameters through state.u
        println("Iteration $(length(losses)): Loss = $(l)")
        println("Current p: [$(round(p_current[1], digits=3)), $(round(p_current[2], digits=3)), $(round(p_current[3], digits=3))]")
        println("True p: [$(p_true[1]), $(p_true[2]), $(round(p_true[3], digits=3))]")
    end
    return false
end

# Optimization setup
optf = OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_combined)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 100)
println("\nOptimization complete!")
println("Final loss: ", losses[end])

# ------------------------------------ Results Analysis ------------------------------------

# Extract final parameters
p_final = res.u.physical

println("\nFinal Results:")
println("True parameters: ", p_true)
println("Initial parameters: ", p_init)
println("Final parameters: ", [round(p, digits=3) for p in p_final])
println("Parameter errors: ", [round(abs(p_true[i] - p_final[i]), digits=3) for i in 1:3])

# Test the trained model
u_pred_final = predict(res.u)

# Compare all results (clean, noisy, and prediction)
p9 = plot(t_data, X_clean[1,:], label="True (clean)", xlabel="t", ylabel="x", linewidth=2, color=:black)
scatter!(p9, t_data[1:5:end], X_noisy[1,1:5:end], label="Noisy data", color=:red, alpha=0.6, markersize=3)
plot!(p9, t_data, u_pred_final[1,:], label="UDE prediction", linestyle=:dash, linewidth=2, color=:blue)
title!("X Component Comparison")
display(p9)

p10 = plot(t_data, X_clean[2,:], label="True (clean)", xlabel="t", ylabel="y", linewidth=2, color=:black)
scatter!(p10, t_data[1:5:end], X_noisy[2,1:5:end], label="Noisy data", color=:red, alpha=0.6, markersize=3)
plot!(p10, t_data, u_pred_final[2,:], label="UDE prediction", linestyle=:dash, linewidth=2, color=:blue)
title!("Y Component Comparison")
display(p10)

p11 = plot(t_data, X_clean[3,:], label="True (clean)", xlabel="t", ylabel="z", linewidth=2, color=:black)
scatter!(p11, t_data[1:5:end], X_noisy[3,1:5:end], label="Noisy data", color=:red, alpha=0.6, markersize=3)
plot!(p11, t_data, u_pred_final[3,:], label="UDE prediction", linestyle=:dash, linewidth=2, color=:blue)
title!("Z Component Comparison")
display(p11)

# NN correction terms
function get_nn_correction(p, X_state)
    return U(X_state, p.neural, _st)[1]
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