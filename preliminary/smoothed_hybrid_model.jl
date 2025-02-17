# Import helpful libraries
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq
using LinearAlgebra, Statistics, Interpolations, ComponentArrays, Plots

# --------------------------------------- Analytical Model ----------------------------------

# Define the AnalyticalModel Module
module AnalyticalModel
using DifferentialEquations
using ForwardDiff
using Parameters
using SpecialFunctions
export Params, p, electrostatic, CoupledSystem!

@with_kw mutable struct Params{T<:Real}
    # Fundamental parameters
    m1::T = 2.0933e-6        # Shuttle mass (kg)
    E::T = 180e9             # Young's modulus (Pa)
    eta::T = 1.849e-5        # Dynamic viscosity of air (Pa·s)
    c::T = 0.015             # Damping scaling factor
    g0::T = 14e-6            # Electrode gap (m)
    Tp::T = 120e-9           # Thickness of parylene layer (m)
    Tf::T = 25e-6            # Device thickness (m)
    gss::T = 14e-6           # Soft-stopper initial gap (m)
    rho::T = 2333.0          # Density of silicon (kg/m³)
    cp::T = 5e-12            # Capacitance of parylene (F)
    wt::T = 9e-6             # Electrode width, top (m)
    wb::T = 30e-6            # Electrode width, bottom (m)
    ws::T = 14.7e-6          # Suspension spring width (m)
    Lss::T = 1400e-6         # Suspension spring length (m)
    Lff::T = 450e-6          # Electrode length (m)
    Leff::T = 400e-6         # Electrode effective overlap length (m)
    e::T = 8.85e-12          # Permittivity of free space (F/m)
    ep::T = 3.2              # Permittivity of parylene
    Vbias::T = 3.0           # Bias voltage (V)
    Rload::T = 0.42e6        # Load resistance (Ω)
    N::T = 160               # Number of electrodes
    kss::T = 6.0             # Soft-stopper spring force (N/m)

    # Dependent parameters
    gp::T = :($(g0 - 2 * Tp))                      # Initial gap including parylene layer (m)
    wavg::T = :($((wt + wb) / 2))                  # Average electrode width (m)
    a::T = :($((wb - wt) / Leff))                  # Tilt factor
    k1::T = :($((4 / 6) * ((E * Tf * (ws^3)) / (Lss^3))))  # Spring constant (N/m)
    k3::T = :($((18 / 25) * ((E * Tf * ws) / (Lss^3))))    # Cubic spring constant (N/m³)
    I::T = :($((1 / 12) * Tf * (wavg^3)))          # Electrode moment of inertia (m⁴)
    m2::T = :($((33 / 140) * rho * Tf * Lff * wavg))  # Mass of electrode (kg)
    ke::T = :($((1 / 4) * E / Lff^3 * I))          # Electrode spring constant (N/m)
end

function Params{T}(p::Params{S}) where {T<:Real, S<:Real}
    # Extract field names
    fnames = fieldnames(typeof(p))
    # Get field values and convert to type T
    fvalues = T.(getfield.(Ref(p), fnames))
    # Construct a NamedTuple of field values
    params_nt_T = NamedTuple{fnames}(fvalues)
    # Return a new Params{T} instance
    return Params{T}(; params_nt_T...)
end

# Initialize a default Params instance
p = Params{Float64}()

# Suspension spring force, Fs
function spring(x1, k1, k3, gss, kss, alpha=1e6)
    # Suspension beam force
    # Fsp = - k1 * x1 - k3 * (x1^3) 
    Fsp = -k1 * x1 

    # Soft stopper force
    S = 0.5 * (1 + tanh(alpha * (abs(x1) - gss))) # Smooth transition function
    Fss = -kss * (abs(x1) - gss) * sign(x1) * S
    
    # Total suspension spring force
    Fs = Fsp + Fss
    return Fs
end

function CoupledSystem!(dz, z, p, t, current_acceleration)
    # Unpack state variables
    z1, z2 = z

    # Compute forces
    Fs = spring(z1, p.k1, p.k3, p.gss, p.kss)

    # Use current_acceleration as the external force
    Fext = current_acceleration

    # Compute derivatives
    dz[1] = z2
    dz[2] = (Fs - 0.005*z2) / p.m1 - Fext 

end
end

# Import the AnalyticalModel module without bringing all exports into Main
import .AnalyticalModel  

# Function to convert Params to NamedTuple
function params_to_namedtuple(p)
    param_names = fieldnames(typeof(p))
    param_values = getfield.(Ref(p), param_names)
    return NamedTuple{param_names}(param_values)
end

# --------------------------------------- External Force ------------------------------------

# Sine Wave External Force
f = 20.0 # Frequency (Hz)
alpha = 3.0 # Applied acceleration constant
g = 9.81 # Gravitational constant 
A = alpha * g
t_ramp = 0.2 # Ramp-up duration (s)
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

# Create a new Params instance by copying the default
p_new = deepcopy(AnalyticalModel.p)

# Initial conditions
x10 = 0.0 # Initial displacement
x10dot = 0.0 # Initial velocity
x20 = 0.0 # Initial displacement
x20dot = 0.0 # Initial velocity

z0 = [x10, x10dot]
tspan = (0.0, 0.5) # simulation length
# teval = () # evaluation steps
abstol = 1e-9 # absolute solver tolerance
reltol = 1e-6 # relative solver tolerance

# ---------------------------------- Solve Analytical Model ---------------------------------

# Define a wrapper function
function CoupledSystem_wrapper!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    AnalyticalModel.CoupledSystem!(dz, z, p, t, current_acceleration)
end

# Define and solve the ODE problem
eqn = ODEProblem(CoupledSystem_wrapper!, z0, tspan, p_new)
# eqn = ODEProblem(AnalyticalModel.CoupledSystem!, z0, tspan, p_new)

# Solve the system using Rosenbrock23 solver
sol = solve(eqn, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7)
# If the system is too stiff, use CVODE_BDF from Sundials
# sol = solve(eqn, CVODE_BDF(), abstol=abstol, reltol=reltol, maxiters=Int(1e9))

# Verify the solution structure
println("Type of sol.u: ", typeof(sol.u))
println("Size of sol.u: ", size(sol.u))
println("Solver status: ", sol.retcode)

# ----------------------------------------- Plotting -----------------------------------------

# Plot the related terms
x1 = [u[1] for u in sol.u]
x1dot = [u[2] for u in sol.u]

p3 = plot(sol.t, x1, xlabel = "Time (s)", ylabel = "x1 (m)", title = "Shuttle Mass Displacement (x1)")
display(p3)
p4 = plot(sol.t, x1dot, xlabel = "Time (s)", ylabel = "x1dot (m/s)", title = "Shuttle Mass Velocity (x1dot)")
display(p4)
p5 = plot(sol.t[130000:end], x1[130000:end], xlabel = "Time (s)", ylabel = "x1 (m)", title = "Shuttle Mass Displacement (x1)")
display(p5)
p6 = plot(sol.t[130000:end], x1dot[130000:end], xlabel = "Time (s)", ylabel = "x1dot (m/s)", title = "Shuttle Mass Velocity (x1dot)")
display(p6)

# Generate forces during the simulation
# Initialize arrays to store forces
Fs_array = Float64[] # Suspension spring force
Fc_array = Float64[] # Collision force

# Iterate over each solution point to compute forces
for (i, t) in enumerate(sol.t)
    # Extract state variables at time t
    z = sol.u[i]
    z1, z2 = z
    
    # Compute Fs (Suspension spring force)
    Fs = AnalyticalModel.spring(z1, p_new.k1, p_new.k3, p_new.gss, p_new.kss)
    push!(Fs_array, Fs)
end

# Plotting respective forces
p9 = plot(sol.t, Fs_array, xlabel = "Time (s)", ylabel = "Fs (N)", title = "Suspension + Soft-stopper Spring Force")
display(p9)
p10 = plot(sol.t[130000:end], Fs_array[130000:end], xlabel = "Time (s)", ylabel = "Fs (N)", title = "Suspension + Soft-stopper Spring Force")
display(p10)

p5 = plot(sol.t[132500:142000], x1[132500:142000], xlabel = "Time (s)", ylabel = "x1 (m)", title = "Shuttle Mass Displacement (x1)")
plot!(sol.t[132500:142000], Fs_array[132500:142000])
hline!([-p_new.gss], label="Contact threshold", linestyle=:dash, color=:red)
display(p5)

p5 = plot(sol.t[132500:142000], x1[132500:142000], xlabel = "Time (s)", ylabel = "x1 (m)", title = "Shuttle Mass Displacement (x1)")
display(p5)


# Function to find transition points with threshold
function find_transitions(sol, gss, threshold=1e-7)
    transition_points = Float64[]
    transition_indices = Int[]
    
    for i in 2:length(sol.t)
        prev_beyond = abs(sol.u[i-1][1]) > (gss - threshold)
        curr_beyond = abs(sol.u[i][1]) > (gss - threshold)
        
        if prev_beyond != curr_beyond
            push!(transition_points, sol.t[i])
            push!(transition_indices, i)
        end
    end
    return transition_points, transition_indices
end

# Function to analyze window around transition
function analyze_transition_window(sol, t_transition, window_size=0.001)
    window_indices = findall(t -> abs(t - t_transition) <= window_size, sol.t)
    t_window = sol.t[window_indices]
    states_window = sol.u[window_indices]
    return t_window, states_window
end

# Calculate gap distance relative to soft-stopper
gap_distance = [p_new.gss - abs(u[1]) for u in sol.u]
p_gap = plot(sol.t, gap_distance,
            xlabel="Time (s)",
            ylabel="Gap Distance (m)",
            title="Distance to Soft-stopper")
hline!([0.0], label="Contact threshold", linestyle=:dash, color=:red)
display(p_gap)

# Find transitions
transition_times, transition_idx = find_transitions(sol, p_new.gss)
println("Found $(length(transition_times)) transitions at times: ", transition_times)

# Analyze each transition
for (i, t_trans) in enumerate(transition_times)
    t_window, states_window = analyze_transition_window(sol, t_trans)
    
    # Extract state variables
    x1_window = [s[1] for s in states_window]
    x1dot_window = [s[2] for s in states_window]
    
    # Calculate forces for this window
    Fs_window = [AnalyticalModel.spring(s[1], p_new.k1, p_new.k3, p_new.gss, p_new.kss) 
                 for s in states_window]
    
    # Phase space plot
    p_phase = plot(x1_window, x1dot_window,
                  xlabel="x1 (m)",
                  ylabel="x1dot (m/s)",
                  title="Phase Portrait Around Transition $i (t=$(round(t_trans, digits=4)))",
                  marker=:circle,
                  markersize=2)
    vline!([p_new.gss, -p_new.gss], label="±gss threshold", linestyle=:dash, color=:red)
    display(p_phase)
    
    # Positions plot
    p_pos = plot(t_window .- t_trans,
                x1_window,
                label="x1",
                xlabel="Time from transition (s)",
                ylabel="Position (m)",
                title="Position Around Transition $i")
    vline!([0], label="Transition point", linestyle=:dash, color=:red)
    hline!([p_new.gss, -p_new.gss], label="±gss threshold", linestyle=:dash, color=:red)
    display(p_pos)
    
    # Velocities plot
    p_vel = plot(t_window .- t_trans,
                x1dot_window,
                label="x1dot",
                xlabel="Time from transition (s)",
                ylabel="Velocity (m/s)",
                title="Velocities Around Transition $i")
    vline!([0], label="Transition point", linestyle=:dash, color=:red)
    display(p_vel)
    
    # Spring force plot
    p_fs = plot(t_window .- t_trans,
               Fs_window,
               label="Fs",
               xlabel="Time from transition (s)",
               ylabel="Force (N)",
               title="Spring Force Around Transition $i")
    vline!([0], label="Transition point", linestyle=:dash, color=:red)
    display(p_fs)
    
    # Calculate and display metrics
    dt = diff(t_window)
    dx1 = diff(x1_window)
    dx1dot = diff(x1dot_window)
    
    println("\nTransition $i Analysis (t = $(round(t_trans, digits=4))):")
    println("Max dx1/dt: ", maximum(abs.(dx1 ./ dt)))
    println("Max dx1dot/dt: ", maximum(abs.(dx1dot ./ dt)))
    println("Gap at transition: ", p_new.gss - abs(x1_window[length(x1_window)÷2]))
end

# Print min/max gap values
println("\nGap Analysis:")
println("Min gap: ", minimum(gap_distance))
println("Max gap: ", maximum(gap_distance))
println("gss value: ", p_new.gss)

# Plot gap histogram
histogram(gap_distance, 
         bins=100,
         xlabel="Gap Distance (m)",
         ylabel="Count",
         title="Distribution of Gap Distances")
display(current())