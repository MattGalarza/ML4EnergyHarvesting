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
use_sine = false

# Define Fext_input based on your choice
if use_sine
    Fext_input = Fext_sine
else
    Fext_input = t -> 0.0  # Default to zero force
end

# ------------------------------------ Initialize Parameters --------------------------------

# Initial conditions
u0 = [1.0, 0.0, -0.74, 0.0, 0.350, 0.0, 1.120, 0.0]
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

# Additional packages needed for embedding analysis
using FFTW, DSP

# ================================== Enhanced Takens Embedding Analysis ==================================

# Simple peak finding function
function find_peaks(signal, min_prominence=0.1)
    """
    Simple peak finding - finds local maxima with minimum prominence
    """
    peaks = Int[]
    n = length(signal)
    
    if n < 3
        return peaks
    end
    
    # Normalize signal for prominence calculation
    signal_range = maximum(signal) - minimum(signal)
    prominence_threshold = min_prominence * signal_range
    
    for i in 2:(n-1)
        # Check if it's a local maximum
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            # Check prominence (simple version)
            local_max = signal[i]
            left_min = minimum(signal[max(1, i-10):i])
            right_min = minimum(signal[i:min(n, i+10)])
            prominence = local_max - max(left_min, right_min)
            
            if prominence > prominence_threshold
                push!(peaks, i)
            end
        end
    end
    
    return peaks
end

# Function to compute Dynamic Mode Decomposition (DMD)
function compute_dmd(X, dt; rank_truncation=nothing)
    """
    Compute DMD of data matrix X where columns are snapshots
    Returns: eigenvalues, modes, mode amplitudes
    """
    m, n = size(X)
    if n < 2
        error("Need at least 2 time snapshots for DMD")
    end
    
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    
    # SVD of X1
    U, S, V = svd(X1)
    
    # Rank truncation
    if rank_truncation !== nothing
        r = min(rank_truncation, length(S), size(U, 2))
        U = U[:, 1:r]
        S = S[1:r]
        V = V[:, 1:r]
    else
        r = min(length(S), size(U, 2))
        U = U[:, 1:r]
        S = S[1:r]
        V = V[:, 1:r]
    end
    
    # Build Atilde
    Atilde = U' * X2 * V * Diagonal(1 ./ S)
    
    # Eigendecomposition of Atilde
    λ, W = eigen(Atilde)
    
    # Compute DMD modes
    Φ = X2 * V * Diagonal(1 ./ S) * W
    
    # Continuous-time eigenvalues
    μ = log.(λ) / dt
    
    return μ, Φ, λ, U, S, V
end

# Function to characterize system dynamics
function characterize_system(timeseries, t; plot_results=true)
    """
    Characterize if system is modal-dominated, chaotic, or mixed
    """
    dt = t[2] - t[1]
    
    # 1. Spectral analysis
    Y = fft(timeseries)
    freqs = fftfreq(length(timeseries), 1/dt)
    power_spectrum = abs2.(Y)
    
    # Find dominant frequencies
    pos_freqs = freqs[1:length(freqs)÷2]
    pos_power = power_spectrum[1:length(power_spectrum)÷2]
    
    # Identify peaks using our custom function
    peak_indices = find_peaks(pos_power, 0.1)
    
    if !isempty(peak_indices)
        dominant_freqs = pos_freqs[peak_indices]
        peak_powers = pos_power[peak_indices]
        
        # Sort by power
        sorted_indices = sortperm(peak_powers, rev=true)
        top_freqs = dominant_freqs[sorted_indices[1:min(5, length(sorted_indices))]]
        top_powers = peak_powers[sorted_indices[1:min(5, length(sorted_indices))]]
    else
        top_freqs = Float64[]
        top_powers = Float64[]
    end
    
    # 2. Compute spectral entropy (measure of broadband vs tonal)
    normalized_power = pos_power ./ sum(pos_power)
    # Add small epsilon to avoid log(0)
    spectral_entropy = -sum(normalized_power .* log.(normalized_power .+ 1e-12))
    max_entropy = log(length(normalized_power))
    normalized_entropy = spectral_entropy / max_entropy
    
    # 3. DMD analysis for modal content
    # Reshape time series for DMD (need matrix with columns as snapshots)
    window_size = min(100, length(timeseries)÷4)  # Use sliding windows
    if length(timeseries) > 2 * window_size
        # Create Hankel matrix for DMD
        num_windows = length(timeseries) - window_size + 1
        X_dmd = zeros(window_size, num_windows)
        for i in 1:num_windows
            X_dmd[:, i] = timeseries[i:i+window_size-1]
        end
        
        try
            μ, Φ, λ, U, S, V = compute_dmd(X_dmd, dt, rank_truncation=min(10, size(X_dmd, 1)÷2))
            
            # Assess modal dominance using singular values
            modal_amplitudes = S[1:min(5, length(S))]  # Use singular values instead
            dominant_mode_ratio = modal_amplitudes[1] / sum(modal_amplitudes)
        catch e
            println("DMD analysis failed: $e")
            dominant_mode_ratio = 0.0
        end
    else
        dominant_mode_ratio = 0.0
    end
    
    # 4. Assess dynamics type
    if normalized_entropy < 0.4 && dominant_mode_ratio > 0.4
        dynamics_type = "modal_dominated"
    elseif normalized_entropy > 0.7 && dominant_mode_ratio < 0.3
        dynamics_type = "chaotic"
    else
        dynamics_type = "mixed"
    end
    
    if plot_results
        # Limit frequency range for better visualization
        max_freq_plot = min(100, length(pos_freqs))
        p1 = plot(pos_freqs[1:max_freq_plot], pos_power[1:max_freq_plot], 
                 xlabel="Frequency (Hz)", ylabel="Power", 
                 title="Power Spectrum", yscale=:log10)
        
        # Mark dominant frequencies
        if !isempty(top_freqs)
            valid_indices = findall(f -> f <= pos_freqs[max_freq_plot], top_freqs)
            if !isempty(valid_indices)
                scatter!(p1, top_freqs[valid_indices], top_powers[valid_indices], 
                        color=:red, markersize=6, label="Dominant Peaks")
            end
        end
        
        display(p1)
    end
    
    results = (
        dynamics_type = dynamics_type,
        spectral_entropy = normalized_entropy,
        dominant_mode_ratio = dominant_mode_ratio,
        dominant_frequencies = top_freqs,
        power_spectrum = (freqs=pos_freqs, power=pos_power)
    )
    
    println("System Characterization:")
    println("  Dynamics Type: $dynamics_type")
    println("  Spectral Entropy: $(round(normalized_entropy, digits=3))")
    println("  Dominant Mode Ratio: $(round(dominant_mode_ratio, digits=3))")
    if !isempty(top_freqs)
        println("  Top Frequencies: $(round.(top_freqs[1:min(3,end)], digits=3)) Hz")
    else
        println("  Top Frequencies: None detected")
    end
    
    return results
end

# Classical Takens embedding
function takens_embedding(timeseries, τ, m)
    """
    Create Takens embedding with delay τ and dimension m
    """
    N = length(timeseries)
    embedded_dim = m
    num_points = N - (m-1)*τ
    
    if num_points <= 0
        error("Insufficient data points for embedding. Try smaller τ or m.")
    end
    
    embedded = zeros(num_points, embedded_dim)
    
    for i in 1:num_points
        for j in 1:embedded_dim
            embedded[i, j] = timeseries[i + (j-1)*τ]
        end
    end
    
    return embedded
end

# Derivative embedding (enhanced approach)
function derivative_embedding(timeseries, t, max_deriv=2)
    """
    Create embedding using derivatives: [y(t), dy/dt, d²y/dt²]
    """
    dt = t[2] - t[1]
    
    # Start with the original signal
    embedded_data = timeseries[1:end-max_deriv]
    
    if max_deriv >= 1
        # First derivative using central differences where possible
        dy_dt = zeros(length(timeseries))
        dy_dt[1] = (timeseries[2] - timeseries[1]) / dt  # Forward difference
        dy_dt[end] = (timeseries[end] - timeseries[end-1]) / dt  # Backward difference
        
        for i in 2:length(timeseries)-1
            dy_dt[i] = (timeseries[i+1] - timeseries[i-1]) / (2*dt)  # Central difference
        end
        
        dy_dt_truncated = dy_dt[1:end-max_deriv+1]
        embedded_data = hcat(embedded_data, dy_dt_truncated)
    end
    
    if max_deriv >= 2
        # Second derivative
        d2y_dt2 = zeros(length(timeseries)-1)
        for i in 2:length(dy_dt)-1
            d2y_dt2[i-1] = (dy_dt[i+1] - dy_dt[i-1]) / (2*dt)
        end
        
        embedded_data = hcat(embedded_data, d2y_dt2[1:length(embedded_data)])
    end
    
    return embedded_data
end

# Optimal embedding parameters
function estimate_embedding_params(timeseries; max_dim=10, max_delay=50)
    """
    Estimate optimal embedding dimension and delay using autocorrelation
    """
    # Autocorrelation-based delay estimation
    N = length(timeseries)
    max_lag = min(max_delay, N÷4)
    
    autocorr = zeros(max_lag)
    mean_ts = mean(timeseries)
    var_ts = var(timeseries)
    
    for lag in 1:max_lag
        autocorr[lag] = sum((timeseries[1:N-lag] .- mean_ts) .* (timeseries[lag+1:N] .- mean_ts)) / ((N-lag) * var_ts)
    end
    
    # Find first minimum or 1/e point
    τ_optimal = findfirst(x -> x < exp(-1), autocorr)
    if τ_optimal === nothing
        τ_optimal = min(10, max_lag÷2)
    end
    
    # Conservative embedding dimension for your type of system
    m_optimal = min(4, max_dim)
    
    return τ_optimal, m_optimal
end

# Adaptive embedding strategy
function adaptive_embedding(timeseries, t, characterization)
    """
    Choose embedding strategy based on system characterization
    """
    try
        if characterization.dynamics_type == "modal_dominated"
            println("Using derivative embedding for modal-dominated system")
            embedded = derivative_embedding(timeseries, t, 2)
            method = "derivative"
            
        elseif characterization.dynamics_type == "chaotic"
            println("Using classical Takens embedding for chaotic system")
            τ, m = estimate_embedding_params(timeseries)
            println("  Estimated τ=$τ, m=$m")
            embedded = takens_embedding(timeseries, τ, m)
            method = "takens"
            
        else  # mixed
            println("Using hybrid embedding for mixed dynamics")
            # Try derivative embedding first (more reliable for your system)
            embedded = derivative_embedding(timeseries, t, 2)
            method = "hybrid_derivative"
        end
        
        return embedded, method
        
    catch e
        println("Embedding failed with error: $e")
        println("Falling back to simple delay embedding")
        
        # Fallback: simple delay embedding
        τ = 5
        m = 3
        try
            embedded = takens_embedding(timeseries, τ, m)
            return embedded, "fallback_takens"
        catch e2
            println("All embedding methods failed: $e2")
            # Return trivial embedding
            embedded = reshape(timeseries[1:min(100, length(timeseries))], :, 1)
            return embedded, "failed"
        end
    end
end

# Validation function
function validate_embedding(original_series, embedded_coords, t, method)
    """
    Validate embedding quality using prediction accuracy
    """
    if size(embedded_coords, 1) < 20
        println("Insufficient data for validation")
        return (rmse=Inf, mae=Inf, correlation=0.0)
    end
    
    # Simple validation: try to predict next value using nearest neighbors
    train_frac = 0.8
    train_size = Int(floor(train_frac * size(embedded_coords, 1)))
    
    train_embedded = embedded_coords[1:train_size, :]
    test_size = min(50, size(embedded_coords, 1) - train_size)  # Limit test size
    test_embedded = embedded_coords[train_size+1:train_size+test_size, :]
    test_original = original_series[train_size+1:train_size+test_size]
    
    predictions = zeros(test_size)
    
    # Simple k-nearest neighbors prediction
    k = min(5, train_size÷10)
    for i in 1:test_size
        test_point = test_embedded[i, :]
        
        # Find k nearest neighbors in training set
        distances = [norm(test_point - train_embedded[j, :]) for j in 1:size(train_embedded, 1)]
        nearest_indices = sortperm(distances)[1:min(k, length(distances))]
        
        # Predict as average of nearest neighbors' corresponding original values
        if !isempty(nearest_indices)
            next_values = original_series[nearest_indices .+ 1]
            valid_values = next_values[nearest_indices .+ 1 .<= length(original_series)]
            if !isempty(valid_values)
                predictions[i] = mean(valid_values)
            end
        end
    end
    
    # Compute prediction accuracy
    valid_mask = .!isnan.(predictions) .&& predictions .!= 0
    if sum(valid_mask) > 0
        valid_predictions = predictions[valid_mask]
        valid_actual = test_original[valid_mask]
        
        rmse = sqrt(mean((valid_predictions - valid_actual).^2))
        mae = mean(abs.(valid_predictions - valid_actual))
        
        if length(valid_predictions) > 1 && var(valid_predictions) > 1e-10
            correlation = cor(valid_predictions, valid_actual)
        else
            correlation = 0.0
        end
    else
        rmse, mae, correlation = Inf, Inf, 0.0
    end
    
    println("Embedding Validation ($method):")
    println("  RMSE: $(round(rmse, digits=4))")
    println("  MAE: $(round(mae, digits=4))")
    println("  Correlation: $(round(correlation, digits=4))")
    
    return (rmse=rmse, mae=mae, correlation=correlation)
end

# ================================== Apply Enhanced Takens Analysis ==================================

# For this example, let's use x1 as our single "observable" (simulating voltage in your real problem)
vout = x4_1  # This represents your single observable

println("\n" * "="^60)
println("ENHANCED TAKENS EMBEDDING ANALYSIS")
println("="^60)

# Step 1: Characterize the system
println("\nStep 1: System Characterization")
characterization = characterize_system(vout, t1, plot_results=true)

# Step 2: Apply adaptive embedding
println("\nStep 2: Adaptive Embedding")
embedded_coords, embedding_method = adaptive_embedding(vout, t1, characterization)

println("Embedded coordinates shape: $(size(embedded_coords))")

# Step 3: Validate embedding
println("\nStep 3: Validation")
validation_results = validate_embedding(vout, embedded_coords, t1, embedding_method)

# Step 4: Visualize embedding
println("\nStep 4: Visualization")

if size(embedded_coords, 2) >= 3
    # 3D embedding plot
    p_embed = plot(embedded_coords[:, 1], embedded_coords[:, 2], embedded_coords[:, 3],
                   xlabel="Coordinate 1", ylabel="Coordinate 2", zlabel="Coordinate 3",
                   title="Reconstructed Attractor ($embedding_method)", 
                   seriestype=:path, linewidth=1)
    display(p_embed)
elseif size(embedded_coords, 2) == 2
    # 2D embedding plot
    p_embed = plot(embedded_coords[:, 1], embedded_coords[:, 2],
                   xlabel="Coordinate 1", ylabel="Coordinate 2",
                   title="Reconstructed Attractor ($embedding_method)", 
                   seriestype=:path, linewidth=1)
    display(p_embed)
end

# Step 5: Compare with true phase space (if available)
println("\nStep 5: Comparison with True Phase Space")

# Plot true phase space for comparison
p_true = plot(x1_1, v1_1, xlabel="x1", ylabel="v1", 
              title="True Phase Space", seriestype=:path, linewidth=1, color=:red)
display(p_true)

# Step 6: Test reconstruction of other states (validation)
println("\nStep 6: State Reconstruction Assessment")

# Simple test: see if we can distinguish between different system states using embedding
if embedding_method != "failed"
    println("Embedding successful with method: $embedding_method")
    println("Reconstructed manifold has $(size(embedded_coords, 2)) dimensions")
    
    # Assess if embedding captures system complexity
    if size(embedded_coords, 1) > 1
        embedded_variance = var(embedded_coords, dims=1)
        println("Variance across embedding dimensions: $(round.(embedded_variance[:], digits=6))")
        
        # Check if different embedding coordinates are capturing different dynamics
        if size(embedded_coords, 2) > 1
            correlation_matrix = cor(embedded_coords)
            println("Correlation matrix of embedding coordinates:")
            display(correlation_matrix)
        end
    end
else
    println("Embedding failed - consider alternative approaches")
end

println("\n" * "="^60)
println("ANALYSIS COMPLETE")
println("="^60)


# ================================== State Reconstruction from Takens Embedding ==================================

println("\n" * "="^60)
println("TESTING STATE RECONSTRUCTION FROM TAKENS EMBEDDING")
println("="^60)

function reconstruct_states_from_embedding(embedded_coords, true_states, t_data; test_fraction=0.2)
    """
    Attempt to reconstruct hidden states from Takens embedding coordinates
    using various regression approaches
    """
    
    # Align data lengths
    min_len = min(size(embedded_coords, 1), size(true_states, 1))
    embedded_aligned = embedded_coords[1:min_len, :]
    true_states_aligned = true_states[1:min_len, :]
    t_aligned = t_data[1:min_len]
    
    # Split into train/test
    n_train = Int(floor((1 - test_fraction) * min_len))
    
    X_train = embedded_aligned[1:n_train, :]
    X_test = embedded_aligned[n_train+1:end, :]
    
    Y_train = true_states_aligned[1:n_train, :]
    Y_test = true_states_aligned[n_train+1:end, :]
    
    t_train = t_aligned[1:n_train]
    t_test = t_aligned[n_train+1:end]
    
    println("Training data: $(size(X_train)) -> $(size(Y_train))")
    println("Test data: $(size(X_test)) -> $(size(Y_test))")
    
    # Method 1: Linear Regression
    println("\nMethod 1: Linear Regression")
    
    # Add bias term
    X_train_bias = hcat(ones(size(X_train, 1)), X_train)
    X_test_bias = hcat(ones(size(X_test, 1)), X_test)
    
    # Solve for each state variable
    n_states = size(Y_train, 2)
    linear_predictions = zeros(size(Y_test))
    linear_coeffs = zeros(size(X_train_bias, 2), n_states)
    
    for i in 1:n_states
        # Ridge regression to handle potential multicollinearity
        λ = 1e-6
        A = X_train_bias' * X_train_bias + λ * I
        b = X_train_bias' * Y_train[:, i]
        coeffs = A \ b
        linear_coeffs[:, i] = coeffs
        linear_predictions[:, i] = X_test_bias * coeffs
    end
    
    # Method 2: Polynomial Features
    println("\nMethod 2: Polynomial Regression (degree 2)")
    
    function create_poly_features(X, degree=2)
        n_samples, n_features = size(X)
        # Create polynomial features: [1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
        poly_features = ones(n_samples, 1)  # bias
        
        # Linear terms
        poly_features = hcat(poly_features, X)
        
        if degree >= 2
            # Quadratic terms
            for i in 1:n_features
                for j in i:n_features
                    poly_features = hcat(poly_features, X[:, i] .* X[:, j])
                end
            end
        end
        
        return poly_features
    end
    
    X_train_poly = create_poly_features(X_train, 2)
    X_test_poly = create_poly_features(X_test, 2)
    
    poly_predictions = zeros(size(Y_test))
    
    for i in 1:n_states
        λ = 1e-4
        A = X_train_poly' * X_train_poly + λ * I
        b = X_train_poly' * Y_train[:, i]
        coeffs = A \ b
        poly_predictions[:, i] = X_test_poly * coeffs
    end
    
    # Method 3: Local Linear Regression (k-NN based)
    println("\nMethod 3: Local Linear Regression")
    
    knn_predictions = zeros(size(Y_test))
    k = min(20, size(X_train, 1) ÷ 10)  # Number of neighbors
    
    for i in 1:size(X_test, 1)
        test_point = X_test[i, :]
        
        # Find k nearest neighbors
        distances = [norm(test_point - X_train[j, :]) for j in 1:size(X_train, 1)]
        neighbor_indices = sortperm(distances)[1:k]
        
        # Local linear regression using neighbors
        X_local = X_train[neighbor_indices, :]
        Y_local = Y_train[neighbor_indices, :]
        
        # Add bias and solve
        X_local_bias = hcat(ones(k), X_local)
        test_point_bias = [1; test_point]
        
        for j in 1:n_states
            try
                if k > size(X_local_bias, 2)  # Overdetermined
                    coeffs = X_local_bias \ Y_local[:, j]
                else  # Underdetermined, use mean
                    coeffs = [mean(Y_local[:, j]); zeros(size(X_local, 2))]
                end
                knn_predictions[i, j] = test_point_bias' * coeffs
            catch
                knn_predictions[i, j] = mean(Y_local[:, j])
            end
        end
    end
    
    return (
        linear = linear_predictions,
        polynomial = poly_predictions,
        knn = knn_predictions,
        true_test = Y_test,
        t_test = t_test,
        coefficients = linear_coeffs
    )
end

# Prepare data for reconstruction
# True states: [x1, v1, x2, v2, x3, v3, x4, v4]
true_states_matrix = hcat(x1_1, v1_1, x2_1, v2_1, x3_1, v3_1, x4_1, v4_1)

# We want to reconstruct states 3-8 (x2, v2, x3, v3, x4, v4) from embedding of x1
target_states = true_states_matrix[:, 3:8]  # Exclude x1 and v1
state_names = ["x2", "v2", "x3", "v3", "x4", "v4"]

println("Attempting to reconstruct $(length(state_names)) hidden states from Takens embedding...")

# Perform reconstruction
results = reconstruct_states_from_embedding(embedded_coords, target_states, t1)

# ================================== Evaluate Reconstruction Quality ==================================

function evaluate_reconstruction(predictions, true_values, method_name, state_names)
    """
    Compute reconstruction metrics for each state
    """
    println("\n$(method_name) Results:")
    println("-" * length("$(method_name) Results:"))
    
    n_states = size(predictions, 2)
    metrics = []
    
    for i in 1:n_states
        pred = predictions[:, i]
        true_val = true_values[:, i]
        
        # Remove any NaN or Inf values
        valid_mask = .!isnan.(pred) .&& .!isinf.(pred) .&& .!isnan.(true_val)
        pred_clean = pred[valid_mask]
        true_clean = true_val[valid_mask]
        
        if length(pred_clean) > 0
            # Metrics
            rmse = sqrt(mean((pred_clean - true_clean).^2))
            mae = mean(abs.(pred_clean - true_clean))
            
            # Normalized metrics
            true_range = maximum(true_clean) - minimum(true_clean)
            nrmse = rmse / (true_range + 1e-12)
            
            # Correlation
            if length(pred_clean) > 1 && var(pred_clean) > 1e-12
                correlation = cor(pred_clean, true_clean)
            else
                correlation = 0.0
            end
            
            # R-squared
            ss_tot = sum((true_clean .- mean(true_clean)).^2)
            ss_res = sum((true_clean - pred_clean).^2)
            r_squared = 1 - ss_res / (ss_tot + 1e-12)
            
            push!(metrics, (rmse=rmse, mae=mae, nrmse=nrmse, correlation=correlation, r_squared=r_squared))
            
            println("  $(state_names[i]): RMSE=$(round(rmse, digits=4)), NRMSE=$(round(nrmse, digits=3)), Correlation=$(round(correlation, digits=3)), R²=$(round(r_squared, digits=3))")
        else
            push!(metrics, (rmse=Inf, mae=Inf, nrmse=Inf, correlation=0.0, r_squared=-Inf))
            println("  $(state_names[i]): No valid predictions")
        end
    end
    
    return metrics
end

# Evaluate all methods
linear_metrics = evaluate_reconstruction(results.linear, results.true_test, "Linear Regression", state_names)
poly_metrics = evaluate_reconstruction(results.polynomial, results.true_test, "Polynomial Regression", state_names)
knn_metrics = evaluate_reconstruction(results.knn, results.true_test, "Local Linear (k-NN)", state_names)

# ================================== Visualization ==================================

println("\nGenerating reconstruction plots...")

# Plot reconstruction for each state (showing best method)
function plot_reconstruction_comparison(results, state_names)
    plots_array = []
    
    for i in 1:length(state_names)
        p = plot(results.t_test, results.true_test[:, i], 
                label="True $(state_names[i])", 
                linewidth=2, color=:red,
                title="Reconstruction of $(state_names[i])")
        
        plot!(p, results.t_test, results.linear[:, i], 
              label="Linear", linewidth=1, color=:blue, linestyle=:dash)
        
        plot!(p, results.t_test, results.polynomial[:, i], 
              label="Polynomial", linewidth=1, color=:green, linestyle=:dot)
        
        plot!(p, results.t_test, results.knn[:, i], 
              label="k-NN", linewidth=1, color=:orange, linestyle=:dashdot)
        
        xlabel!(p, "Time (s)")
        ylabel!(p, "$(state_names[i])")
        
        push!(plots_array, p)
    end
    
    return plot(plots_array..., layout=(3, 2), size=(1200, 900))
end

# Create comparison plot
p_reconstruction = plot_reconstruction_comparison(results, state_names)
display(p_reconstruction)

# ================================== Summary Assessment ==================================

println("\n" * "="^60)
println("RECONSTRUCTION ASSESSMENT SUMMARY")
println("="^60)

# Calculate average performance across states
function summarize_performance(metrics, method_name)
    valid_correlations = [m.correlation for m in metrics if !isnan(m.correlation) && !isinf(m.correlation)]
    valid_nrmse = [m.nrmse for m in metrics if !isnan(m.nrmse) && !isinf(m.nrmse)]
    valid_r2 = [m.r_squared for m in metrics if !isnan(m.r_squared) && !isinf(m.r_squared)]
    
    avg_corr = isempty(valid_correlations) ? 0.0 : mean(valid_correlations)
    avg_nrmse = isempty(valid_nrmse) ? Inf : mean(valid_nrmse)
    avg_r2 = isempty(valid_r2) ? -Inf : mean(valid_r2)
    
    println("$method_name Average Performance:")
    println("  Correlation: $(round(avg_corr, digits=3))")
    println("  NRMSE: $(round(avg_nrmse, digits=3))")
    println("  R²: $(round(avg_r2, digits=3))")
    
    return (correlation=avg_corr, nrmse=avg_nrmse, r_squared=avg_r2)
end

linear_summary = summarize_performance(linear_metrics, "Linear Regression")
poly_summary = summarize_performance(poly_metrics, "Polynomial Regression")
knn_summary = summarize_performance(knn_metrics, "k-NN Regression")

# Final verdict
println("\nFINAL VERDICT:")
best_correlation = max(linear_summary.correlation, poly_summary.correlation, knn_summary.correlation)
best_r2 = max(linear_summary.r_squared, poly_summary.r_squared, knn_summary.r_squared)

if best_correlation > 0.8 && best_r2 > 0.6
    println("✓ TAKENS EMBEDDING SHOWS PROMISE: Good reconstruction of hidden states")
    println("  Best correlation: $(round(best_correlation, digits=3))")
    println("  Best R²: $(round(best_r2, digits=3))")
elseif best_correlation > 0.5 && best_r2 > 0.25
    println("~ TAKENS EMBEDDING SHOWS LIMITED UTILITY: Moderate reconstruction quality")
    println("  Best correlation: $(round(best_correlation, digits=3))")
    println("  Best R²: $(round(best_r2, digits=3))")
else
    println("✗ TAKENS EMBEDDING FAILS: Poor reconstruction of hidden states")
    println("  Best correlation: $(round(best_correlation, digits=3))")
    println("  Best R²: $(round(best_r2, digits=3))")
    println("  Conclusion: Not suitable for closure modeling of this system")
end

println("\n" * "="^60)



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
X_data = u2_norm

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
    return mean(abs2, X_data .- u_pred)
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

