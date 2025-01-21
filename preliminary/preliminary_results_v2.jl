# SciML Tools
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics, Interpolations

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, DelimitedFiles, Random, Parameters, SpecialFunctions

# Function to read the data from the .tmp file
function read_tmp_file(data_file)
    # Open the file and read the lines
    data = readdlm(data_file)

    # Skip the first row (the title row)
    data = data[2:end, :]

    # Extract the two columns into separate arrays
    col1 = data[:, 1]
    col2 = data[:, 2]
    
    return col1, col2
end

# Acceleration data
accel_data = "C:\\Users\\Matthew\\Desktop\\PhD\\Graduate Research\\PhD 2024-2026\\Research - Shaowu Pan Collaboration\\preliminary data\\accelT1.tmp.txt"
x_accel, y_accel = read_tmp_file(accel_data)
y_accel = 9.81 * y_accel
p1 = plot(x_accel, y_accel, xlabel="Data Points", ylabel="Amplitude", title="Acceleration Data")
display(p1)

# Voltage data
voltage_data = "C:\\Users\\Matthew\\Desktop\\PhD\\Graduate Research\\PhD 2024-2026\\Research - Shaowu Pan Collaboration\\preliminary data\\voltageT1.tmp.txt"
x_voltage, y_voltage = read_tmp_file(voltage_data)
p2 = plot(x_voltage, y_voltage, xlabel="Time (sec)", ylabel="Voltage (V)", title="Voltage Data")
display(p2)

println(size(x_accel))
println(size(y_accel))
println(size(x_voltage))
println(size(y_voltage))

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
    dz[4] = (-Fc + Fd + Fe) / m2 - Fext
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
f = 20.0 # Frequency (Hz)
alpha = 3.0 # Applied acceleration constant
g = 9.81 # Gravitational constant 
A = alpha * g
t_ramp = 0.2 # Ramp-up duration (s)
# Define the ramp function (linear ramp)
ramp(t) = t < t_ramp ? t / t_ramp : 1.0
Fext_sine = t -> A * ramp(t) * sin(2 * π * f * t)

# Step 2: Create an interpolation function for the acceleration data
x_time = range(0, stop=10.48575, length=length(y_accel))
# This will allow us to evaluate the acceleration data at any given time within the simulation
Fext_data = interpolate((x_time,), y_accel, Gridded(Constant()))

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
Fext_min, Fext_max = minimum(y_accel), maximum(y_accel)

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
# Fext_norm = normalizer(Fext_array, Fext_min, Fext_max)

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

# ------------------------------------ Setting up the UDE ------------------------------------

y_voltage = V_norm
x_time = sol.t

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(Lux.Dense(7, 32, rbf), # 7 inputs: 6 states + 1 acceleration
                    Lux.Dense(32, 32, rbf), 
                    Lux.Dense(32, 32, rbf),
                    Lux.Dense(32, 6)) # 6 outputs for state corrections

# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the UDESystem! function
function UDESystem!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(z)

    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)

    # Compute the model's derivative
    dz_model = similar(dz)
    AnalyticalModel.CoupledSystem!(dz_model, z, temp_params, t, T(current_acceleration))

    # Neural network correction
    nn_input = vcat(z, T(current_acceleration))
    nn_output = U(nn_input, p.nn, _st)[1]

    # Combine derivatives without in-place mutation
    dz_new = dz_model + nn_output

    # Update dz without in-place mutation
    for i in eachindex(dz)
        dz[i] = dz_new[i]
    end
end

# Define the problem
u0 = z0  # initial states defined earlier
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

# Define the predict function
function predict(θ, saveat = x_time)
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Tsit5(), saveat = saveat,
                abstol = abstol, reltol = reltol, maxiters = 1e7,
                sensealg = ForwardDiffSensitivity())
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(saveat))
    end
    u = Array(sol)
    return u
end

# Define the loss function
function loss(θ)
    pred = predict(θ)
    if any(isnan, pred)
        return Inf
    end
    # Ensure pred and y_voltage have the same length
    min_length = min(length(y_voltage), size(pred, 2))
    return mean((pred[6, 1:min_length] .- y_voltage[1:min_length]).^2)
end

# Training setup
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Optimization setup
using Optimization
using OptimizationOptimisers
using Optimization: OptimizationFunction, OptimizationProblem, solve

# Use automatic differentiation with Zygote and closure
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
optprob = OptimizationProblem(optf, p_combined)

res = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


## -------------------------------------------------------- CLAUDE

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Define normalization constants as a struct for easy access
struct NormalizationParams
    x1_min::Float64; x1_max::Float64
    x1dot_min::Float64; x1dot_max::Float64
    x2_min::Float64; x2_max::Float64
    x2dot_min::Float64; x2dot_max::Float64
    Qvar_min::Float64; Qvar_max::Float64
    V_min::Float64; V_max::Float64
    Fext_min::Float64; Fext_max::Float64
end

# Create normalization parameters instance
norm_params = NormalizationParams(
    -1.5e-5, 1.5e-5,      # x1
    -0.0025, 0.0025,      # x1dot
    -1.5e-5, 1.5e-5,      # x2
    -0.0025, 0.0025,      # x2dot
    -1.5e-11, 2.25e-11,   # Qvar
    -0.0025, 0.0025,      # V
    minimum(y_accel), maximum(y_accel)  # Fext
)

# Modified normalization functions to work with single values
function normalize_state(x::Real, min_val::Real, max_val::Real)
    return -1.0 + 2.0 * (x - min_val) / (max_val - min_val)
end

function denormalize_state(x_norm::Real, min_val::Real, max_val::Real)
    return min_val + (x_norm + 1.0) * (max_val - min_val) / 2.0
end

# Function to normalize the full state vector
function normalize_states(z::AbstractVector, p::NormalizationParams)
    return [
        normalize_state(z[1], p.x1_min, p.x1_max),
        normalize_state(z[2], p.x1dot_min, p.x1dot_max),
        normalize_state(z[3], p.x2_min, p.x2_max),
        normalize_state(z[4], p.x2dot_min, p.x2dot_max),
        normalize_state(z[5], p.Qvar_min, p.Qvar_max),
        normalize_state(z[6], p.V_min, p.V_max)
    ]
end

# Function to denormalize the full state vector
function denormalize_states(z_norm::AbstractVector, p::NormalizationParams)
    return [
        denormalize_state(z_norm[1], p.x1_min, p.x1_max),
        denormalize_state(z_norm[2], p.x1dot_min, p.x1dot_max),
        denormalize_state(z_norm[3], p.x2_min, p.x2_max),
        denormalize_state(z_norm[4], p.x2dot_min, p.x2dot_max),
        denormalize_state(z_norm[5], p.Qvar_min, p.Qvar_max),
        denormalize_state(z_norm[6], p.V_min, p.V_max)
    ]
end

println("Setting up Neural Network...")
# Modified neural network architecture with progress tracking
const U = Lux.Chain(
    Lux.Dense(7, 64, rbf),    
    Lux.Dense(64, 64, rbf),
    Lux.Dense(64, 64, rbf),
    Lux.Dense(64, 6)          
)

# Initialize the neural network
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st
println("Neural Network initialized successfully!")

# Modified UDE system with debugging
function UDESystem!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(z)
    
    # Create temporary parameters with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)
    
    # Get the model's derivative in original space
    dz_model = similar(dz)
    AnalyticalModel.CoupledSystem!(dz_model, z, temp_params, t, T(current_acceleration))
    
    # Normalize current state and acceleration for NN input
    z_norm = normalize_states(z, norm_params)
    acc_norm = normalize_state(current_acceleration, norm_params.Fext_min, norm_params.Fext_max)
    
    # Neural network correction in normalized space
    nn_input = vcat(z_norm, T(acc_norm))
    nn_output = U(nn_input, p.nn, _st)[1]
    
    # Denormalize the NN corrections
    correction_scales = [
        (norm_params.x1_max - norm_params.x1_min)/2,
        (norm_params.x1dot_max - norm_params.x1dot_min)/2,
        (norm_params.x2_max - norm_params.x2_min)/2,
        (norm_params.x2dot_max - norm_params.x2dot_min)/2,
        (norm_params.Qvar_max - norm_params.Qvar_min)/2,
        (norm_params.V_max - norm_params.V_min)/2
    ]
    
    nn_output_denorm = nn_output .* correction_scales
    
    # Combine derivatives
    for i in eachindex(dz)
        dz[i] = dz_model[i] + nn_output_denorm[i]
    end
end

println("Setting up problem parameters...")
# Setup the problem with combined parameters
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the initial value problem
prob_nn = ODEProblem(UDESystem!, z0, tspan, p_combined)
println("Problem parameters set up successfully!")

# Modified predict function with progress tracking
prediction_count = 0
function predict(θ, saveat = x_time)
    global prediction_count += 1
    if prediction_count % 100 == 0
        println("Making prediction #$prediction_count")
    end
    
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Tsit5(), saveat = saveat,
                abstol = abstol, reltol = reltol, maxiters = 1e7,
                sensealg = ForwardDiffSensitivity())
    
    if !SciMLBase.successful_retcode(sol)
        println("Warning: Solver failed at prediction #$prediction_count")
        return fill(NaN, 6, length(saveat))
    end
    
    return Array(sol)
end

# Modified loss function with progress tracking
loss_count = 0
function loss(θ)
    global loss_count += 1
    if loss_count % 10 == 0
        println("Computing loss #$loss_count")
    end
    
    pred = predict(θ)
    if any(isnan, pred)
        println("Warning: NaN detected in prediction at loss #$loss_count")
        return Inf
    end
    
    pred_voltage_norm = [normalize_state(v, norm_params.V_min, norm_params.V_max) 
                        for v in pred[6, :]]
    
    min_length = min(length(y_voltage), length(pred_voltage_norm))
    current_loss = mean((pred_voltage_norm[1:min_length] .- y_voltage[1:min_length]).^2)
    
    if loss_count % 10 == 0
        println("Current loss value: $current_loss")
    end
    
    return current_loss
end

println("Setting up training callback...")
# Training setup with enhanced progress tracking
losses = Float64[]
last_time = time()
callback = function (θ, l)
    push!(losses, l)
    
    current_time = time()
    elapsed = current_time - last_time
    
    if length(losses) % 10 == 0
        println("\nIteration $(length(losses)):")
        println("Current loss: $(losses[end])")
        println("Time since last checkpoint: $(round(elapsed, digits=2)) seconds")
        println("Average time per iteration: $(round(elapsed/10, digits=2)) seconds")
        global last_time = current_time
    end
    return false
end

println("\nInitializing optimization...")
# Optimization setup
adtype = Optimization.AutoForwardDiff()
println("Setting up optimization function...")
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
println("Creating optimization problem...")
optprob = OptimizationProblem(optf, p_combined)

println("\nStarting training process...")
println("This may take a while - progress will be reported every 10 iterations")
# Train the network
res = solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 1000)

println("\nTraining completed!")
println("Final loss: $(losses[end])")
println("Total iterations: $(length(losses))")

println("\nGenerating final predictions...")
# Generate predictions with the trained model
final_pred = predict(res.u)
final_pred_norm = [normalize_state(v, norm_params.V_min, norm_params.V_max) for v in final_pred[6, :]]

println("\nPlotting results...")
# Plot results
p24 = plot(x_time, y_voltage, label="Target (Experimental)")
plot!(p24, x_time[1:length(final_pred_norm)], final_pred_norm, label="UDE Prediction")
xlabel!(p24, "Time")
ylabel!(p24, "Normalized Voltage")
title!(p24, "UDE Performance Comparison")
display(p24)
println("Plotting complete!")

# Save the training history
p25 = plot(losses, yscale=:log10, xlabel="Iterations", ylabel="Loss", label="Training Loss")
title!(p25, "Training History")
display(p25)



## ------------------------------------------------------------------ ChatGPT

# ----------------------------------- Compute Derivative Min and Max -----------------------------------

println("Computing derivative min and max values...")

# Initialize arrays to store the derivatives
dz1_array = Float64[]
dz2_array = Float64[]
dz3_array = Float64[]
dz4_array = Float64[]
dz5_array = Float64[]
dz6_array = Float64[]

# Iterate over each solution point to compute derivatives
for (i, t) in enumerate(sol.t)
    if i % 100 == 0 || i == length(sol.t)
        println("Computing derivatives at time step $i / $(length(sol.t))")
    end
    z = sol.u[i]
    dz_model = zeros(size(z))
    current_acceleration = Fext_input(t)
    AnalyticalModel.CoupledSystem!(dz_model, z, p_new, t, current_acceleration)
    push!(dz1_array, dz_model[1])
    push!(dz2_array, dz_model[2])
    push!(dz3_array, dz_model[3])
    push!(dz4_array, dz_model[4])
    push!(dz5_array, dz_model[5])
    push!(dz6_array, dz_model[6])
end

println("Finished computing derivative min and max values.")

# Compute min and max values for each derivative
dz1_min, dz1_max = minimum(dz1_array), maximum(dz1_array)
dz2_min, dz2_max = minimum(dz2_array), maximum(dz2_array)
dz3_min, dz3_max = minimum(dz3_array), maximum(dz3_array)
dz4_min, dz4_max = minimum(dz4_array), maximum(dz4_array)
dz5_min, dz5_max = minimum(dz5_array), maximum(dz5_array)
dz6_min, dz6_max = minimum(dz6_array), maximum(dz6_array)

# Define min and max for the external force (acceleration)
Fext_min, Fext_max = -29.43, 29.43  # Based on A = alpha * g

# ------------------------------------ Setting up the UDE ------------------------------------

println("Setting up the UDE...")

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(Lux.Dense(7, 32, rbf), # 7 inputs: 6 states + 1 acceleration
                    Lux.Dense(32, 32, rbf), 
                    Lux.Dense(32, 32, rbf),
                    Lux.Dense(32, 6)) # 6 outputs for state corrections

# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

function UDESystem!(dz, z, p, t)
    # Print a message every 0.1 seconds
    if mod(t, 0.1) < 1e-5
        println("UDESystem! at time t = $t")
    end

    current_acceleration = Fext_input(t)
    T = typeof(t)  # Use the type of 't', which could be 'Dual'

    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)

    # Compute the model's derivative
    dz_model = similar(dz, T)
    AnalyticalModel.CoupledSystem!(dz_model, z, temp_params, t, current_acceleration)

    # Normalize the state variables 'z' and 'current_acceleration'
    z_norm = similar(z, T)
    z_norm[1] = normalizer(z[1], x1_min, x1_max)
    z_norm[2] = normalizer(z[2], x1dot_min, x1dot_max)
    z_norm[3] = normalizer(z[3], x2_min, x2_max)
    z_norm[4] = normalizer(z[4], x2dot_min, x2dot_max)
    z_norm[5] = normalizer(z[5], Qvar_min, Qvar_max)
    z_norm[6] = normalizer(z[6], V_min, V_max)

    current_accel_norm = normalizer(current_acceleration, Fext_min, Fext_max)

    # Neural network correction
    nn_input = vcat(z_norm, current_accel_norm)
    nn_output = U(nn_input, p.nn, _st)[1]

    # Denormalize the NN outputs
    nn_output_denorm = similar(nn_output, T)
    nn_output_denorm[1] = denormalizer(nn_output[1], dz1_min, dz1_max)
    nn_output_denorm[2] = denormalizer(nn_output[2], dz2_min, dz2_max)
    nn_output_denorm[3] = denormalizer(nn_output[3], dz3_min, dz3_max)
    nn_output_denorm[4] = denormalizer(nn_output[4], dz4_min, dz4_max)
    nn_output_denorm[5] = denormalizer(nn_output[5], dz5_min, dz5_max)
    nn_output_denorm[6] = denormalizer(nn_output[6], dz6_min, dz6_max)

    # Combine derivatives without in-place mutation
    dz_new = dz_model .+ nn_output_denorm

    # Update 'dz' without in-place mutation
    for i in eachindex(dz)
        dz[i] = dz_new[i]
    end
end

println("Defining the ODE problem for the UDE...")

# Define the problem
u0 = z0  # initial states defined earlier
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

# Adjust the predict function to use a stiff solver and include print statements
function predict(θ, saveat = x_time)
    println("Starting ODE solve in predict function...")
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Rosenbrock23(autodiff = false), saveat = saveat, abstol = abstol,
        reltol = reltol, maxiters = 1e7, sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
        progress = true)
    println("Finished ODE solve in predict function.")
    if !SciMLBase.successful_retcode(sol)
        println("ODE solve unsuccessful: $(sol.retcode)")
        return fill(NaN, 6, length(saveat))
    end
    u = Array(sol)
    # Return u without mutation
    return u
end


# Define the loss function with print statements
function loss(θ)
    println("Computing loss...")
    pred = predict(θ)
    if any(isnan, pred)
        println("Prediction contains NaN values.")
        return Inf
    end
    # Ensure pred and y_voltage have the same length
    min_length = min(length(y_voltage), size(pred, 2))
    # Extract the output voltage without mutating 'pred'
    v_pred = pred[6, 1:min_length]
    # Normalize 'v_pred' without in-place mutation
    v_pred_norm = normalizer(v_pred, V_min, V_max)
    # Compute the loss
    loss_value = mean((v_pred_norm .- y_voltage[1:min_length]).^2)
    println("Loss computed: $loss_value")
    return loss_value
end

println("Setting up training...")

# Training setup
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
    return false
end

# Optimization setup
using Optimization
using OptimizationOptimisers
using Optimization: OptimizationFunction, OptimizationProblem, solve

# Use automatic differentiation with Zygote and closure
adtype = Optimization.AutoZygote()
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
optprob = OptimizationProblem(optf, p_combined)

println("Starting optimization...")

res = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
















## ------------------------------------------------------------------------------
# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(
    Lux.Dense(7, 32, rbf),  # 7 inputs: 6 states + 1 acceleration
    Lux.Dense(32, 32, rbf), 
    Lux.Dense(32, 32, rbf),
    Lux.Dense(32, 6)        # 6 outputs for state corrections
)

# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the UDESystem! function
function UDESystem!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(z)

    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)

    # Compute the model's derivative
    dz_model = similar(dz)
    AnalyticalModel.CoupledSystem!(dz_model, z, temp_params, t, T(current_acceleration))

    # Normalize state and acceleration for NN input
    z_norm = [
        normalizer(z[1], x1_min, x1_max),
        normalizer(z[2], x1dot_min, x1dot_max),
        normalizer(z[3], x2_min, x2_max),
        normalizer(z[4], x2dot_min, x2dot_max),
        normalizer(z[5], Qvar_min, Qvar_max),
        normalizer(z[6], V_min, V_max)
    ]
    accel_norm = normalizer(current_acceleration, Fext_min, Fext_max)
    nn_input = vcat(z_norm, T(accel_norm))

    # Get neural network correction (in normalized space)
    nn_output = U(nn_input, p.nn, _st)[1]

    # Denormalize the correction before adding to model derivatives
    dz_correction = [
        denormalizer(nn_output[1], x1_min, x1_max),
        denormalizer(nn_output[2], x1dot_min, x1dot_max),
        denormalizer(nn_output[3], x2_min, x2_max),
        denormalizer(nn_output[4], x2dot_min, x2dot_max),
        denormalizer(nn_output[5], Qvar_min, Qvar_max),
        denormalizer(nn_output[6], V_min, V_max)
    ]

    # Update derivatives
    for i in eachindex(dz)
        dz[i] = dz_model[i] + dz_correction[i]
    end
end

# Define the problem
u0 = z0  # initial states defined earlier
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

# Define the predict function with a different solver
function predict(θ, saveat = x_time)
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Rosenbrock23(), 
                saveat = saveat,
                abstol = abstol, reltol = reltol, 
                maxiters = 1e7,
                sensealg = ForwardDiffSensitivity())
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(saveat))
    end
    u = Array(sol)
    return u
end

# Define the loss function
function loss(θ)
    pred = predict(θ)
    if any(isnan, pred)
        return Inf
    end
    
    # Normalize predicted voltage for comparison
    pred_voltage_norm = normalizer(pred[6, :], V_min, V_max)
    
    # Compare normalized values
    min_length = min(length(y_voltage), length(pred_voltage_norm))
    return mean((pred_voltage_norm[1:min_length] .- y_voltage[1:min_length]).^2)
end

# Training setup
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Optimization setup
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
optprob = OptimizationProblem(optf, p_combined)

# Solve with optimization
res = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 10)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

## -----------------------------------------------------------------------------------


# Scale factor for initial weights
const INIT_SCALE = 1e-3

# Custom weight initialization function
function scaled_init(rng::AbstractRNG, dims...) 
    return INIT_SCALE * randn(rng, dims...)
end

# Modified neural network architecture
const U = Lux.Chain(
    Lux.Dense(7, 32, rbf; init_weight=scaled_init),
    Lux.Dense(32, 32, rbf; init_weight=scaled_init),
    Lux.Dense(32, 32, rbf; init_weight=scaled_init),
    Lux.Dense(32, 6; init_weight=scaled_init)
)

# Initialize network
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Rest of the UDE system remains the same
function NormalizedUDESystem!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    
    # Get base model derivatives
    dz_model = similar(dz)
    CoupledSystem_wrapper!(dz_model, z, p.model_params, t)
    
    # Pre-normalize state and acceleration for NN input
    norm_z = [normalize_state(z[i], (:x1, :x1dot, :x2, :x2dot, :Qvar, :Vout)[i]) for i in 1:6]
    norm_accel = normalize_state(current_acceleration, :accel)
    nn_input = vcat(norm_z, norm_accel)
    
    # Get NN correction (in normalized space)
    nn_output = U(nn_input, p.nn, _st)[1]
    
    # Denormalize correction before adding to model derivatives
    denorm_correction = [denormalize_state(nn_output[i], (:x1, :x1dot, :x2, :x2dot, :Qvar, :Vout)[i]) for i in 1:6]
    
    # Scale factor for corrections (can be tuned)
    correction_scale = 1e-3
    
    # Combine derivatives with scaled correction
    for i in eachindex(dz)
        dz[i] = dz_model[i] + correction_scale * denorm_correction[i]
    end
end

# Modified predict function
function predict(θ, saveat=x_time)
    _prob = remake(prob_nn, p=θ)
    sol = solve(_prob, Rodas5(), # Changed to Rodas5 for stiff system
                saveat=saveat,
                abstol=1e-9, 
                reltol=1e-6,
                maxiters=1e7,
                save_everystep=false,
                dense=false,
                sensealg=ForwardDiffSensitivity())
    
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(saveat))
    end
    
    # Add stability check
    if any(x -> abs(x) > 1e3, sol.u)
        return fill(NaN, 6, length(saveat))
    end
    
    return Array(sol)
end

# Loss function with regularization
function loss(θ)
    pred = predict(θ)
    if any(isnan, pred)
        return Inf
    end
    
    # Extract and normalize predicted voltage
    pred_voltage = [normalize_state(v, :Vout) for v in pred[6, :]]
    
    min_length = min(length(y_voltage), length(pred_voltage))
    base_loss = mean((pred_voltage[1:min_length] .- y_voltage[1:min_length]).^2)
    
    # Add L2 regularization
    stability_penalty = sum(abs2, θ.nn) * 1e-6
    
    return base_loss + stability_penalty
end

# Optimization setup
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
optprob = OptimizationProblem(optf, p_combined)

# Solve with conservative settings
res = solve(optprob, OptimizationOptimisers.Adam(1e-4), callback=callback, maxiters=10, allow_f_increases=false)



## --------------------------------------------------------------------

# Define min and max values for normalization (from your data)
x1_min, x1_max = -1.5e-5, 1.5e-5       # For x1
x1dot_min, x1dot_max = -0.0025, 0.0025  # For x1dot
x2_min, x2_max = -1.5e-5, 1.5e-5       # For x2
x2dot_min, x2dot_max = -0.0025, 0.0025  # For x2dot
Qvar_min, Qvar_max = -1.5e-11, 2.25e-11 # For Qvar
V_min, V_max = -0.0025, 0.0025          # For Vout

# Min and max values for acceleration (from your data)
acc_min, acc_max = minimum(y_accel), maximum(y_accel)

# Normalization function
function normalizer(data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return new_min .+ (data .- min_val) .* (new_max - new_min) ./ (max_val - min_val)
end

# Denormalization function
function denormalizer(norm_data, min_val, max_val, new_min::Float64 = -1.0, new_max::Float64 = 1.0)
    return min_val .+ (norm_data .- new_min) .* (max_val - min_val) ./ (new_max - new_min)
end

# Normalize initial conditions
u0_norm = [
    normalizer(z0[1], x1_min, x1_max),
    normalizer(z0[2], x1dot_min, x1dot_max),
    normalizer(z0[3], x2_min, x2_max),
    normalizer(z0[4], x2dot_min, x2dot_max),
    normalizer(z0[5], Qvar_min, Qvar_max),
    normalizer(z0[6], V_min, V_max)
]

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(Lux.Dense(7, 32, rbf), # 7 inputs: 6 states + 1 acceleration
                    Lux.Dense(32, 32, rbf), 
                    Lux.Dense(32, 32, rbf),
                    Lux.Dense(32, 6)) # 6 outputs for state corrections

# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the UDESystem! function
function UDESystem!(dz, z, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(z)

    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)

    # ----------------- Denormalize State Variables -----------------
    z_denorm = [
        denormalizer(z[1], x1_min, x1_max),
        denormalizer(z[2], x1dot_min, x1dot_max),
        denormalizer(z[3], x2_min, x2_max),
        denormalizer(z[4], x2dot_min, x2dot_max),
        denormalizer(z[5], Qvar_min, Qvar_max),
        denormalizer(z[6], V_min, V_max)
    ]

    # Compute the model's derivative using denormalized states
    dz_model = similar(dz)
    AnalyticalModel.CoupledSystem!(dz_model, z_denorm, temp_params, t, T(current_acceleration))

    # ----------------- Normalize Inputs for Neural Network -----------------
    # State variables are already normalized (they are z)
    # Normalize current_acceleration
    current_acceleration_norm = normalizer(current_acceleration, acc_min, acc_max)

    # Create neural network input (normalized states + normalized acceleration)
    nn_input = vcat(z, T(current_acceleration_norm))

    # Neural network correction (outputs are in normalized scale)
    nn_output_norm = U(nn_input, p.nn, _st)[1]

    # ----------------- Denormalize Neural Network Outputs -----------------
    nn_output = [
        denormalizer(nn_output_norm[1], x1_min, x1_max),
        denormalizer(nn_output_norm[2], x1dot_min, x1dot_max),
        denormalizer(nn_output_norm[3], x2_min, x2_max),
        denormalizer(nn_output_norm[4], x2dot_min, x2dot_max),
        denormalizer(nn_output_norm[5], Qvar_min, Qvar_max),
        denormalizer(nn_output_norm[6], V_min, V_max)
    ]

    # ----------------- Combine Derivatives -----------------
    dz_new = dz_model .+ nn_output

    # Update dz
    for i in eachindex(dz)
        dz[i] = dz_new[i]
    end
end

# Define the problem with normalized initial conditions
prob_nn = ODEProblem(UDESystem!, u0_norm, tspan, p_combined)

# Define the predict function
function predict(θ, saveat = x_time)
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Tsit5(), saveat = saveat,
                abstol = abstol, reltol = reltol, maxiters = 1e7,
                sensealg = ForwardDiffSensitivity())
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(saveat))
    end
    u = Array(sol)
    return u
end

# Define the loss function
function loss(θ)
    pred = predict(θ)
    if any(isnan, pred)
        return Inf
    end
    # Denormalize the predicted Vout (6th state variable)
    V_pred_norm = pred[6, :]
    V_pred = denormalizer(V_pred_norm, V_min, V_max)
    # Ensure pred and y_voltage have the same length
    min_length = min(length(y_voltage), length(V_pred))
    return mean((V_pred[1:min_length] .- y_voltage[1:min_length]).^2)
end

# Training setup
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Use automatic differentiation with ForwardDiff
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
optprob = OptimizationProblem(optf, p_combined)

# Start the optimization
res = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 20)
println("Training loss after $(length(losses)) iterations: $(losses[end])")



## ---------------------------------
# 1. Plot the training loss over iterations
pl_losses = plot(1:length(losses), losses, yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "Training Loss", color = :blue)
display(pl_losses)

# 2. Get the trained parameters
p_trained = res.minimizer

# 3. Predict with the trained model
predictions = predict(p_trained)

# 4. Denormalize the predicted output voltage (6th state variable)
V_pred_norm = predictions[6, :]
V_pred = denormalizer(V_pred_norm, V_min, V_max)

# 5. Ensure the time vector matches the predictions
t_pred = x_time[1:length(V_pred)]

# 6. Plot the predicted output voltage vs. time alongside the actual data
pl_voltage = plot(t_pred, V_pred, xlabel = "Time (s)", ylabel = "Output Voltage (V)",
    label = "Model Prediction", color = :red)
plot!(t_pred, y_voltage[1:length(V_pred)], label = "Actual Data", color = :blue)
display(pl_voltage)

# 7. Compute and plot the prediction error
error = V_pred - y_voltage[1:length(V_pred)]
pl_error = plot(t_pred, error, xlabel = "Time (s)", ylabel = "Prediction Error (V)",
    label = "Error", color = :green)
display(pl_error)

# 8. Compute the absolute error (L2 norm at each time point)
error_norm = abs.(error)

# 9. Plot the error norm over time
pl_error_norm = plot(t_pred, error_norm, yaxis = :log10, xlabel = "Time (s)", ylabel = "Absolute Error",
    label = "Prediction Error Norm", color = :purple)
display(pl_error_norm)

# 10. Optionally, plot other state variables
# Denormalize other predicted state variables if needed
x1_pred_norm = predictions[1, :]
x1_pred = denormalizer(x1_pred_norm, x1_min, x1_max)

pl_x1 = plot(t_pred, x1_pred, xlabel = "Time (s)", ylabel = "x1 (m)",
    label = "Predicted x1", color = :orange)
display(pl_x1)

# 11. Optionally, plot the neural network's correction to Vout
# Compute the neural network outputs over time
nn_outputs = []

for i in 1:length(t_pred)
    # Get the normalized state at time t_pred[i]
    z_norm = predictions[:, i]
    # Get the current acceleration
    current_acceleration = Fext_input(t_pred[i])
    current_acceleration_norm = normalizer(current_acceleration, acc_min, acc_max)
    # Neural network input
    nn_input = vcat(z_norm, current_acceleration_norm)
    # Get the neural network output
    nn_output_norm = U(nn_input, p_trained.nn, _st)[1]
    # Denormalize the output
    nn_output = [
        denormalizer(nn_output_norm[1], x1_min, x1_max),
        denormalizer(nn_output_norm[2], x1dot_min, x1dot_max),
        denormalizer(nn_output_norm[3], x2_min, x2_max),
        denormalizer(nn_output_norm[4], x2dot_min, x2dot_max),
        denormalizer(nn_output_norm[5], Qvar_min, Qvar_max),
        denormalizer(nn_output_norm[6], V_min, V_max)
    ]
    push!(nn_outputs, nn_output)
end

# Convert nn_outputs to an array
nn_outputs_array = hcat(nn_outputs...)

# Plot the neural network's correction to the output voltage
pl_nn_correction = plot(t_pred, nn_outputs_array[6, :], xlabel = "Time (s)", ylabel = "NN Correction to Vout (V)",
    label = "NN Correction", color = :magenta)
display(pl_nn_correction)
