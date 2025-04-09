# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

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
    c::T = 0.001             # Damping scaling factor
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
    Rload::T = 72e6          # Load resistance (Ω)
    N::T = 160               # Number of electrodes
    kss::T = 6.0             # Soft-stopper spring force (N/m)

    # Dependent parameters
    gp::T = :($(g0 - 2 * Tp))                      # Initial gap including parylene layer (m)
    wavg::T = :($((wt + wb) / 2))                  # Average electrode width (m)
    a::T = :($((wb - wt) / Leff))                  # Tilt factor
    k1::T = :($((4 / 60) * ((E * Tf * (ws^3)) / (Lss^3))))  # Spring constant (N/m)
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

# Suspension spring force, Fsp
function spring(x1, k1, k3, gss, kss)
    # Fsp = - k1 * x1 - k3 * (x1^3) # Suspension beam force
    Fsp = -k1 * x1 # Suspension beam force
    if abs(x1) < gss
        Fss = 0.0
    else
        Fss = -kss * (abs(x1) - gss) * sign(x1) # Soft-stopper force
    end
    Fs = Fsp + Fss
    return Fs
end

# Electrode collision force, Fc
function collision(x1, x2, m2, ke, gp)
    if abs(x2) < gp
        m2 = m2
        Fc = -ke * (x1 - x2)
    else
        m2 = 2 * m2
        Fc = -ke * (x1 - x2) + ke * (abs(x2) - gp) * sign(x2)
    end
    return m2, Fc
end

# Viscous damping, Fd
function damping(x2, x2dot, a, c, gp, Leff, Tf, eta)  
    # Damping LHS
    ul = gp + x2
    ll = ul + a * Leff
    A1_l = -ul * Leff / (ul + 2 * a) * x2dot
    A2_l = 12 * eta * a^-2 / (2 * ul + a * Leff) * x2dot
    Fd_l = Tf * (Leff * A2_l - 6 * eta * Leff / a / ul / ll * A1_l + 
                6 * eta * Leff * a^-4 / ll * x2dot + 
                12 * eta * a^-3 * x2dot * log(abs(ul / ll)))
    
    # Damping RHS
    ur = gp - x2
    lr = ur + a * Leff
    A1_r = -ur * Leff / (ur + 2 * a) * x2dot
    A2_r = 12 * eta * a^-2 / (2 * ur + a * Leff) * x2dot
    Fd_r = Tf * (Leff * A2_r - 6 * eta * Leff / a / ur / lr * A1_r + 
                6 * eta * Leff * a^-4 / lr * x2dot - 
                12 * eta * a^-3 * x2dot * log(abs(ur / lr)))
    
    # Total damping
    Fd = -c * (Fd_l + Fd_r)
    return Fd
end

# Electrostatic coupling, Fe
function electrostatic(x1, x2, Qvar, g0, gp, a, e, ep, cp, wt, wb, ke, E, I, Leff, Tf, Tp, N)
    # Function for variable capacitance, Cvar
    function Cvar_func(x2)
        if abs(x2) < gp # electrodes are NOT in contact
            crl = (e * ep * Leff * Tf) / Tp
            # RHS                           
            Cair_r = ((e * Tf) / (2 * a)) * log((gp - x2 + 2 * a * Leff) / (gp - x2))
            Cvar_r = 1 / (1/crl + 1/Cair_r + 1/crl)
            # LHS
            Cair_l = ((e * Tf) / (2 * a)) * log((gp + x2 + 2 * a * Leff) / (gp + x2))
            Cvar_l = 1 / (1/crl + 1/Cair_l + 1/crl)
            # Total variable capacitance, Cvar
            Cvar_value = (N / 2) * (Cvar_r + Cvar_l)
        elseif abs(x2) >= gp # electrodes are in contact
            crl = (e * ep * Leff * Tf) / Tp
            # d = ((wb - wt) + (abs(x1) - gp)) / Leff
            # d = (wb - wt) - (((ke * (abs(x2) - gp)) * (Leff^3)) / (3 * E * I))
            # Colliding electrode
            Cair_c = ((e * Tf * Leff)/abs(x2)) * (log((2 * Tp + abs(x2))/(2 * Tp)))
            Cvar_c = 1 / (1/crl + 1/Cair_c + 1/crl)
            # Non-colliding electrode
            Cair_nc = ((e * Tf) / (2 * a)) * log((gp + abs(x2) + 2 * a * Leff) / (gp + abs(x2)))
            Cvar_nc = 1 / (1/crl + 1/Cair_nc + 1/crl)
            # Total variable capacitance, Cvar
            Cvar_value = (N / 2) * (Cvar_c + Cvar_nc)
        end
        return Cvar_value
    end
    # Compute Cvar and its derivative
    Cvar = Cvar_func(x2)
    dC = ForwardDiff.derivative(Cvar_func, x2)
    # Total capacitance, Ctotal
    Ctotal = Cvar + cp
    # Electrostatic force, Fe
    NFe = -((Qvar^2) / (2 * Ctotal^2)) * dC
    Fe = NFe / (N / 2)
    return Ctotal, Fe
end

function CoupledSystem!(dz, z, p, t, current_acceleration)
    # Unpack state variables
    z1, z2, z3, z4, z5, Vout = z

    # Compute forces
    Fs = spring(z1, p.k1, p.k3, p.gss, p.kss)
    m2, Fc = collision(z1, z3, p.m2, p.ke, p.gp)
    Fd = damping(z3, z4, p.a, p.c, p.gp, p.Leff, p.Tf, p.eta)
    Ctotal, Fe = electrostatic(z1, z3, z5, p.g0, p.gp, p.a, p.e, p.ep, p.cp, p.wt, p.wb, p.ke, p.E, p.I, p.Leff, p.Tf, p.Tp, p.N)

    # Use current_acceleration as the external force
    Fext = current_acceleration

    # Compute derivatives
    dz[1] = z2
    dz[2] = (Fs + (p.N / 2) * Fc) / p.m1 - Fext
    dz[3] = z4
    dz[4] = (-Fc + Fd + Fe) / m2
    dz[5] = (p.Vbias - (z5 / Ctotal)) / p.Rload
    dz[6] = (p.Vbias - z5 / Ctotal - Vout) / (p.Rload * Ctotal)
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
f = 150.0 # Frequency (Hz)
alpha = 0.3 # Applied acceleration constant
g = 9.81 # Gravitational constant 
A = alpha * g
t_ramp = 0.2 # Ramp-up duration (s)
# Define the ramp function (linear ramp)
ramp(t) = t < t_ramp ? t / t_ramp : 1.0
Fext_sine = t -> A * ramp(t) * sin(2 * π * f * t)

# Define the external force function for the collected acceleration data
function Fext_data_function(t::Real)
    return Fext_data(t)  # Return the acceleration value at time t
end

# ------------------------------------- Set Input Force ------------------------------------

# Set to `true` to use sine wave, `false` otherwise
use_sine = true

# Set to `true` to use collected data, `false` otherwise
use_data = false

# Define Fext_input based on your choice
if use_sine
    Fext_input = Fext_sine
elseif use_data
    Fext_input = Fext_data_function
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

# Compute initial electrostatic parameters
Ctotal0, Fe0 = AnalyticalModel.electrostatic(
    x10, x20, 0.0, 
    p_new.g0, p_new.gp, p_new.a, 
    p_new.e, p_new.ep, p_new.cp, 
    p_new.wt, p_new.wb, p_new.ke, 
    p_new.E, p_new.I, p_new.Leff, 
    p_new.Tf, p_new.Tp, p_new.N
)
Q0 = p_new.Vbias * Ctotal0 # Initial charge
Vout0 = p_new.Vbias - (Q0 / Ctotal0) # Initial voltage
z0 = [x10, x10dot, x20, x20dot, Q0, Vout0]
tspan = (0.0, 0.3) # simulation length
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
sol = solve(eqn, Rosenbrock23(); abstol=abstol, reltol=reltol, maxiters=1e7) # saveat=tspan[1]:0.000001:tspan[2]
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
x2 = [u[3] for u in sol.u]
x2dot = [u[4] for u in sol.u]
Qvar = [u[5] for u in sol.u]
V = [u[6] for u in sol.u]

p3 = plot(sol.t, x1, xlabel = "Time (s)", ylabel = "x1 (m)", title = "Shuttle Mass Displacement (x1)")
display(p3)
p4 = plot(sol.t, x1dot, xlabel = "Time (s)", ylabel = "x1dot (m/s)", title = "Shuttle Mass Velocity (x1dot)")
display(p4)
p5 = plot(sol.t, x2, xlabel = "Time (s)", ylabel = "x2 (m)", title = "Mobile Electrode Displacement (x2)")
display(p5)
p6 = plot(sol.t, x2dot, xlabel = "Time (s)", ylabel = "x2dot (m/s)", title = "Mobile Electrode Velocity (x2)")
display(p6)
p7 = plot(sol.t, Qvar, xlabel = "Time (s)", ylabel = "Qvar (C)", title = "Charge (Qvar)")
display(p7)
p8 = plot(sol.t, V, xlabel = "Time (s)", ylabel = "Vout (V)", title = "Output Voltage")
display(p8)

# Generate forces during the simulation
# Initialize arrays to store forces
Fs_array = Float64[] # Suspension spring force
Fc_array = Float64[] # Collision force
Fd_array = Float64[] # Damping force
Fe_array = Float64[] # Electrostatic force

# Iterate over each solution point to compute forces
for (i, t) in enumerate(sol.t)
    # Extract state variables at time t
    z = sol.u[i]
    z1, z2, z3, z4, z5, Vout = z
    
    # Compute Fs (Suspension spring force)
    Fs = AnalyticalModel.spring(z1, p_new.k1, p_new.k3, p_new.gss, p_new.kss)
    push!(Fs_array, Fs)
    
    # Compute Fc (Collision force)
    m2_val, Fc = AnalyticalModel.collision(z1, z3, p_new.m2, p_new.ke, p_new.gp)
    push!(Fc_array, Fc)
    
    # Compute Fd (Damping force)
    Fd = AnalyticalModel.damping(z3, z4, p_new.a, p_new.c, p_new.gp, p_new.Leff, p_new.Tf, p_new.eta)
    push!(Fd_array, Fd)
    
    # Compute Fe (Electrostatic force)
    Ctotalx, Fe = AnalyticalModel.electrostatic(z1, z3, z5, p_new.g0, p_new.gp, p_new.a, 
                                            p_new.e, p_new.ep, p_new.cp, p_new.wt, 
                                            p_new.wb, p_new.ke, p_new.E, p_new.I, 
                                            p_new.Leff, p_new.Tf, p_new.Tp, p_new.N)
    push!(Fe_array, Fe)
end

# Plotting respective forces
p9 = plot(sol.t, Fs_array, xlabel = "Time (s)", ylabel = "Fs (N)", title = "Suspension + Soft-stopper Spring Force")
display(p9)
p10 = plot(sol.t, Fc_array, xlabel = "Time (s)", ylabel = "Fc (N)", title = "Mobile Electrode Collision Force")
display(p10)
p11 = plot(sol.t, Fd_array, xlabel = "Time (s)", ylabel = "Fd (N)", title = "Viscous Damping Force")
display(p11)
p12 = plot(sol.t, Fe_array, xlabel = "Time (s)", ylabel = "Fe (N)", title = "Electrostatic Force")
display(p12)
p13 = plot(sol.t, Fext_input, xlabel = "Time (s)", ylabel = "Fext (N)", title = "Applied External Force")
display(p13)

# ----------------------------------- Normalizing the Data -----------------------------------

# Define a function to normalize data
function normalizer(data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    # min_val, max_val = minimum(data), maximum(data)
    return new_min .+ (data .- min_val) .* (new_max - new_min) ./ (max_val - min_val)
end

# Define a function to denormalize data
function denormalizer(norm_data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return min_val .+ (norm_data .- new_min) .* (max_val - min_val) ./ (new_max - new_min)
end

# Min and max values for state variables + applied force
x1_min, x1_max = -1.5e-5, 1.5e-5          
x1dot_min, x1dot_max = -0.0025, 0.0025  
x2_min, x2_max = -1.5e-5, 1.5e-5         
x2dot_min, x2dot_max = -0.0025, 0.0025    
Qvar_min, Qvar_max = -1.5e-11, 2.25e-11  
V_min, V_max = -0.0025, 0.0025        
Fs_min, Fs_max = -1.0e-4, 1.0e-4          
Fc_min, Fc_max = -1.0e-6, 1.0e-6       
Fd_min, Fd_max = -1.0e-6, 1.0e-6      
Fe_min, Fe_max = -1.0e-6, 1.0e-6
Fext_val = [Fext_input(tt) for tt in sol.t]        
Fext_min, Fext_max = minimum(Fext_val), maximum(Fext_val)

# Apply normalization to model data
x1_norm = normalizer(x1, x1_min, x1_max)
x1dot_norm = normalizer(x1dot, x1dot_min, x1dot_max)
x2_norm = normalizer(x2, x2_min, x2_max)
x2dot_norm = normalizer(x2dot, x2dot_min, x2dot_max)
Qvar_norm = normalizer(Qvar, Qvar_min, Qvar_max)
V_norm = normalizer(V, V_min, V_max)
Fs_norm = normalizer(Fs_array, Fs_min, Fs_max)
Fc_norm = normalizer(Fc_array, Fc_min, Fc_max)
Fd_norm = normalizer(Fd_array, Fd_min, Fd_max)
Fe_norm = normalizer(Fe_array, Fe_min, Fe_max)
Fext_norm = normalizer(Fext_val, Fext_min, Fext_max)

p14 = plot(sol.t, x1_norm, xlabel = "Time (s)", ylabel = "x1 (m)", title = "Norm (x1)")
display(p14)
p15 = plot(sol.t, x1dot_norm, xlabel = "Time (s)", ylabel = "x1dot (m/s)", title = "Norm (x1dot)")
display(p15)
p16 = plot(sol.t, x2_norm, xlabel = "Time (s)", ylabel = "x2 (m)", title = "Norm (x2)")
display(p16)
p17 = plot(sol.t, x2dot_norm, xlabel = "Time (s)", ylabel = "x2dot (m/s)", title = "Norm (x2)")
display(p17)
p18 = plot(sol.t, Qvar_norm, xlabel = "Time (s)", ylabel = "Qvar (C)", title = "Norm (Qvar)")
display(p18)
p19 = plot(sol.t, V_norm, xlabel = "Time (s)", ylabel = "Vout (V)", title = "Norm (V)")
display(p19)
p20 = plot(sol.t, Fs_norm, xlabel = "Time (s)", ylabel = "Fs (N)", title = "Norm (Fs)")
display(p20)
p21 = plot(sol.t, Fc_norm, xlabel = "Time (s)", ylabel = "Fc (N)", title = "Norm (Fc)")
display(p21)
p22 = plot(sol.t, Fd_norm, xlabel = "Time (s)", ylabel = "Fd (N)", title = "Norm (Fd)")
display(p22)
p23 = plot(sol.t, Fe_norm, xlabel = "Time (s)", ylabel = "Fe (N)", title = "Norm (Fe)")
display(p23)

# ------------------------------- Setting up the simulated data ------------------------------

# Create fixed time points for consistent sampling
dt = 0.0001  # Choose an appropriate timestep
t_points = 0:dt:0.3  # Match your tspan

# First run - generate "true" data
println("\nGenerating true data...")
sol1 = solve(eqn, Rosenbrock23(); 
             abstol=abstol, reltol=reltol, maxiters=1e7,
             saveat=t_points)  # Save at fixed points

# Extract and normalize the voltage data from first run
V1 = [u[6] for u in sol1.u]
V1_norm = normalizer(V1, V_min, V_max)

# Save this as our "experimental" data
y_voltage = V1_norm
x_time = t_points

# Second run - modify a parameter
println("\nGenerating modified data...")
p_modified = deepcopy(AnalyticalModel.p)
p_modified.c = 0.002

# Solve with modified parameter
eqn_modified = ODEProblem(CoupledSystem_wrapper!, z0, tspan, p_modified)
sol2 = solve(eqn_modified, Rosenbrock23(); 
             abstol=abstol, reltol=reltol, maxiters=1e7,
             saveat=t_points)  # Same fixed points

# Extract and normalize voltage from second run
V2 = [u[6] for u in sol2.u]
V2_norm = normalizer(V2, V_min, V_max)

println("Data sizes:")
println("Time points: ", length(t_points))
println("Original solution: ", length(V1_norm))
println("Modified solution: ", length(V2_norm))

# Plot both solutions to verify difference
p24 = plot(Array(x_time), V1_norm, label="Original Model", 
           xlabel="Time (s)", ylabel="Normalized Voltage")
plot!(p24, Array(x_time), V2_norm, label="Modified Model")
title!(p24, "Comparison of Model Outputs")
display(p24)

p24 = plot(Array(x_time[2500:end]), V1_norm[2500:end], label="Original Model", 
           xlabel="Time (s)", ylabel="Normalized Voltage")
plot!(p24, Array(x_time[2500:end]), V2_norm[2500:end], label="Modified Model")
title!(p24, "Comparison of Model Outputs")
display(p24)

println("\nData generation complete!")
println("Original damping: ", AnalyticalModel.p.c)
println("Modified damping: ", p_modified.c)

# ------------------------------------ Setting up the UDE ------------------------------------

# Create normalized initial conditions
u0_norm = [
    normalizer(z0[1], x1_min, x1_max),
    normalizer(z0[2], x1dot_min, x1dot_max),
    normalizer(z0[3], x2_min, x2_max),
    normalizer(z0[4], x2dot_min, x2dot_max),
    normalizer(z0[5], Qvar_min, Qvar_max),
    normalizer(z0[6], V_min, V_max)
]

# Define UDE dynamics with normalized states
function ude_dynamics!(du, u, p, t)
    # Extract physical and neural parameters from p
    p_physical = p[1:n_physical]
    p_neural = p[n_physical+1:end]

    # Create parameter struct with current values
    p_current = deepcopy(p_modified)
    for i in 1:n_physical
        setfield!(p_current, physical_param_keys[i], p_physical[i])
    end
    current_acceleration = Fext_input(t)
    acc_norm = normalizer(current_acceleration, Fext_min, Fext_max)
    
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
    AnalyticalModel.CoupledSystem!(du_model, u_denorm, p_current, t, current_acceleration)
    
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
    nn_correction = U(nn_input, p_neural, _st)[1]
    
    # Combine normalized derivatives
    du .= du_model_norm + nn_correction
end

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Regular deep NN chain
const U = Lux.Chain(Lux.Dense(7, 64, rbf), # 7 inputs: 6 states + 1 acceleration
                    # Lux.Dense(64, 64, rbf),
                    Lux.Dense(64, 32, rbf),
                    Lux.Dense(32, 6) # 6 outputs for state corrections
) 

# Separate deep NN for each state
#const U = Lux.Chain(Lux.Dense(7, 64, rbf),
#                    Lux.Split(
#                        Lux.Chain(Lux.Dense(64, 32, rbf), Lux.Dense(32, 2)),  # For x1, x1dot
#                        Lux.Chain(Lux.Dense(64, 32, rbf), Lux.Dense(32, 2)),  # For x2, x2dot
#                        Lux.Chain(Lux.Dense(64, 32, rbf), Lux.Dense(32, 2))   # For Qvar, V
#                        )
#)

# Initialize NN
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Choose fundamental parameters to optimize
physical_param_keys = [:m1, :E, :eta, :c, :g0, :Tp, :Tf, :gss, :rho, :cp, :wt, :wb, :ws, :Lss, :Lff, :Leff, :e, :ep, :Vbias, :Rload, :N, :kss]  
n_physical = length(physical_param_keys)
p_physical = Float64[getfield(p_modified, key) for key in physical_param_keys]

# Convert parameters to a flat vector
p_neural = ComponentArray(nn_params)
p_combined = vcat(p_physical, p_neural)  # Physical parameters + NN parameters

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0_norm, tspan, p_combined)

# Predict function
function predict(θ)
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Tsit5(), saveat = x_time, abstol = abstol, 
                reltol = reltol, maxiters = 1e7, verbose = false)
    
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(x_time))
    end
    return Array(sol)
end

# Loss function
function loss(θ)
    pred = predict(θ)
    if any(isnan, pred)
        return Inf
    end
    scale_factor = 100.0
    current_loss = scale_factor * mean((pred[6, :] .- y_voltage).^2)
    return current_loss
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
optprob = OptimizationProblem(optf, p_combined)

# Start optimization
println("\nStarting optimization...")
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 100)
println("\nOptimization complete!")
println("Final loss: ", losses[end])

# ----------------------------------------- Plotting -----------------------------------------

# Plot final results
println("\nGenerating final plots...")
final_pred = predict(res.u)
p25 = plot(x_time, y_voltage, label = "Original Model", xlabel = "Time (s)", ylabel = "Normalized Voltage")
plot!(p25, x_time, V2_norm, label = "Modified Model")
plot!(p25, x_time, final_pred[6,:], label = "UDE Prediction")
title!(p25, "UDE Performance")
display(p25)

p26 = plot(x_time[2500:end], y_voltage[2500:end], label = "Original Model", xlabel = "Time (s)", ylabel = "Normalized Voltage")
plot!(p26, x_time[2500:end], V2_norm[2500:end], label = "Modified Model")
plot!(p26, x_time[2500:end], final_pred[6,:][2500:end], label = "UDE Prediction")
title!(p26, "UDE Performance")
display(p26)

# State labels and actual data extraction
println("\nPlotting trajectories comparison...")
p_traj = plot(layout=(3,2), size=(1000,800))
state_labels = ["x1", "x1dot", "x2", "x2dot", "Qvar", "V"]
original_states = [normalizer([u[i] for u in sol1.u], eval(Symbol(state_labels[i], "_min")), eval(Symbol(state_labels[i], "_max"))) for i in 1:6]

# Plot each state
for i in 1:6
    plot!(p_traj[i], x_time, original_states[i], label = "Original", xlabel = "Time (s)", ylabel = "Normalized "*state_labels[i])
    plot!(p_traj[i], x_time, final_pred[i,:], label = "UDE")
end
title!(p_traj, "State Trajectories: Original vs UDE")
display(p_traj)

# Loss vs Epochs
p27 = plot(1:length(losses), losses, xlabel = "Epoch", ylabel = "Loss", label = "Training Loss")
title!(p27, "Loss vs Epochs")
display(p27)

# Predicted voltage vs actual voltage
p28 = plot(x_time, y_voltage, label = "Original Voltage", xlabel = "Time (s)", ylabel = "Normalized Voltage")
plot!(p28, x_time, final_pred[6,:], label = "UDE Prediction")
title!(p28, "Voltage: Original vs UDE Prediction")
display(p28)

p29 = plot(x_time[2500:end], y_voltage[2500:end], label = "Original Voltage", xlabel = "Time (s)", ylabel = "Normalized Voltage")
plot!(p29, x_time[2500:end], final_pred[6,:][2500:end], label = "UDE Prediction")
title!(p29, "Voltage: Original vs UDE Prediction")
display(p29)

# Compute and plot prediction error
voltage_error = final_pred[6,:] .- y_voltage
p30 = plot(x_time, voltage_error, xlabel = "Time (s)", ylabel = "Error", label = "Voltage Prediction Error")
title!(p30, "UDE Prediction Error")
display(p30)

# Compute and plot L2 norm error for all states
l2_errors = zeros(length(x_time))
for t in 1:length(x_time)
    # Compute L2 norm error across all states at each time point
    error_vector = final_pred[:,t] .- [s[t] for s in original_states]
    l2_errors[t] = sqrt(sum(error_vector.^2))
end

p31 = plot(x_time, l2_errors, xlabel = "Time (s)", ylabel = "L2 Norm Error", label = "State Space Error")
title!(p31, "L2 Norm Error Across All States")
display(p31)

# Print some error statistics
println("\nError Statistics:")
println("Mean L2 Error: ", mean(l2_errors))
println("Max L2 Error: ", maximum(l2_errors))
println("Mean Voltage Error: ", mean(abs.(voltage_error)))
println("Max Voltage Error: ", maximum(abs.(voltage_error)))

