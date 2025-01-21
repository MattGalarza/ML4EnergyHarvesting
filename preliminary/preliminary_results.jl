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
    N::Int = 160             # Number of electrodes
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
    z1, z2, z3, z4, z5 = z

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
z0 = [x10, x10dot, x20, x20dot, Q0]
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

# Generate forces during the simulation
# Initialize arrays to store forces
Fs_array = Float64[] # Suspension spring force
Fc_array = Float64[] # Collision force
Fd_array = Float64[] # Damping force
Fe_array = Float64[] # Electrostatic force
Vout_array = Float64[] # Output voltage

# Iterate over each solution point to compute forces
for (i, t) in enumerate(sol.t)
    # Extract state variables at time t
    z = sol.u[i]
    z1, z2, z3, z4, z5 = z
    
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
    
    # Compute Vout (Output voltage)
    Vout = p_new.Vbias - z5 / Ctotalx
    push!(Vout_array, Vout)
end

# Plotting respective forces
p8 = plot(sol.t, Fs_array, xlabel = "Time (s)", ylabel = "Fs (N)", title = "Suspension + Soft-stopper Spring Force")
display(p8)
p9 = plot(sol.t, Fc_array, xlabel = "Time (s)", ylabel = "Fc (N)", title = "Mobile Electrode Collision Force")
display(p9)
p10 = plot(sol.t, Fd_array, xlabel = "Time (s)", ylabel = "Fd (N)", title = "Viscous Damping Force")
display(p10)
p11 = plot(sol.t, Fe_array, xlabel = "Time (s)", ylabel = "Fe (N)", title = "Electrostatic Force")
display(p11)
p12 = plot(sol.t, Fext_input, xlabel = "Time (s)", ylabel = "Fext (N)", title = "Applied External Force")
display(p12)
p13 = plot(sol.t, Vout_array, xlabel = "Time (s)", ylabel = "Vout (V)", title = "Output Voltage")
display(p13)

# ------------------------------------ Setting up the UDE ------------------------------------

# Radial Basis Function (RBF)
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf),
                    Lux.Dense(5, 5, rbf), 
                    Lux.Dense(5, 5, rbf),
                    Lux.Dense(5, 2))

# Get the initial parameters and state variables of the model
rng = Random.default_rng() # random value used to initialize the weights
p, st = Lux.setup(rng, U) # p = dictionary of NN weights and bias, st = initial state of NN
const _st = st

# Set the data for the UDE based on model and collected data above
X = Vout_array # analytical model voltage output
t = sol.t # analytical model time 
Xₙ = y_voltage # collected voltage output

size(X)
size(Xₙ)

# Define parameters as a ComponentArray for automatic differentiation
p_ude = ComponentArray(p_new)

# Setting up the hybrid model
# Define the hybrid model, u' = known(u) + NN(u)
function ude_dynamics!(du, u, p_ude, t, p_true)
    û = U(u, p_ude, _st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p_ude, t) = ude_dynamics!(du, u, p_ude, t, p_ude)

# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p_ude)

# --------------------------------- Setting up training loop ---------------------------------

function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Rosenbrock23(autodiff=false), saveat = T, 
                abstol=abstol, reltol=reltol, 
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end

losses = Float64[]

callback = function (p_ude, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# ----------------------------------------- Training -----------------------------------------

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((θ, p_ude) -> loss(θ), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_ude))

res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")





# -------------------------------------------- Claude NN WITHOUT dynamics ----------------------

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Modify the neural network to output a scalar
const U = Lux.Chain(Lux.Dense(2, 10, rbf),
                    Lux.Dense(10, 10, rbf),
                    Lux.Dense(10, 1, identity))

# Initialize the neural network parameters
rng = Random.default_rng()
p, st = Lux.setup(rng, U)
const _st = st

function UDESystem!(du, u, p, t)
    # u is the current voltage
    # du will be the rate of change of voltage
    
    # Get the current acceleration
    current_acceleration = Fext_input(t)
    
    # Neural network prediction (now a scalar)
    nn_output = U([current_acceleration, u[1]], p.nn, _st)[1][1]  # Note the [1][1] to get a scalar
    
    # Set the rate of change
    du[1] = nn_output
end

# Assuming x_voltage is your time array from experimental data
tspan = (0.0, 5)

# Initial condition (initial voltage)
u0 = [y_voltage[1]]  # Assuming y_voltage is your experimental voltage data

# Combine neural network parameters
p_combined = ComponentArray(nn = p)

# Define the UDE problem
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

function predict(θ, saveat=x_voltage)
    _prob = remake(prob_nn, p=θ)
    solve(_prob, Tsit5(), saveat=saveat, 
          abstol=1e-6, reltol=1e-6)
end

function loss(θ, p)
    sol = predict(θ)
    if !SciMLBase.successful_retcode(sol.retcode)
        return Inf
    end
    chunk_size = 1000  # Adjust this based on your available memory
    total_loss = 0.0
    total_points = 0
    for i in 1:chunk_size:length(y_voltage)
        end_idx = min(i + chunk_size - 1, length(y_voltage))
        chunk = i:end_idx
        pred_chunk = sol[1, chunk]  # Extract the first state variable
        total_loss += sum(abs2, pred_chunk .- y_voltage[chunk])
        total_points += length(chunk)
    end
    total_loss / total_points
end

losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Convert p_combined to a flat vector
p_vec = ComponentVector{Float64}(p_combined)

# Use a gradient-free optimization method
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, p_vec)

res1 = solve(optprob, NelderMead(), callback = callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Extract the optimized parameters
p_optimized = res1.u


# -------------------------------------------- Claude WITH dynamics -----------------------------

# Truncate x_voltage and y_voltage to match tspan
indices_within_tspan = findall(x -> x >= tspan[1] && x <= tspan[2], x_voltage)
x_voltage = x_voltage[indices_within_tspan]
y_voltage = y_voltage[indices_within_tspan]


# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(Lux.Dense(6, 10, rbf), # 6 inputs: 5 states + 1 acceleration
                    Lux.Dense(10, 10, rbf), 
                    Lux.Dense(10, 10, rbf),
                    Lux.Dense(10, 5)) # 5 outputs for state corrections


# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the UDESystem! function
function UDESystem!(du, u, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(u)
    
    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)
    
    # Call the analytical model's ODE function
    AnalyticalModel.CoupledSystem!(du, u, temp_params, t, T(current_acceleration))
    
    # Neural network correction
    nn_input = vcat(u, T(current_acceleration))
    nn_output = U(nn_input, p.nn, _st)[1]
    du .+= nn_output
end

# Define the problem
u0 = z0  # initial states defined earlier
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

function predict(θ, saveat = x_voltage)
    _prob = remake(prob_nn, p = θ)
    sol = solve(_prob, Rosenbrock23(), saveat = saveat, 
                abstol = 1e-8, reltol = 1e-6, maxiters = 1e7)
    
    if !SciMLBase.successful_retcode(sol)
        return fill(NaN, 6, length(saveat))
    end
    
    # Ensure the output has the same number of time points as saveat
    t = sol.t
    u = sol(t)
    
    vout = map(1:length(t)) do i
        _, Ctotal = AnalyticalModel.electrostatic(
            u[1,i], u[3,i], u[5,i], 
            θ.model_params.g0, θ.model_params.gp, 
            θ.model_params.a, θ.model_params.e, θ.model_params.ep, 
            θ.model_params.cp, θ.model_params.wt, θ.model_params.wb, 
            θ.model_params.ke, θ.model_params.E, θ.model_params.I, 
            θ.model_params.Leff, θ.model_params.Tf, θ.model_params.Tp, θ.model_params.N
        )
        θ.model_params.Vbias - (u[5,i] / Ctotal)
    end
    
    return vcat(u, vout')
end

function loss(θ, p)
    pred = predict(θ)
    if size(pred, 1) != 6 || any(isnan, pred)
        return Inf
    end
    # Ensure pred and y_voltage have the same length
    min_length = min(length(y_voltage), size(pred, 2))
    return mean((pred[6, 1:min_length] .- y_voltage[1:min_length]).^2)
end

function loss(θ, p)
    println("Evaluating loss...")
    pred = predict(θ)
    if size(pred, 1) != 6 || any(isnan, pred)
        println("Invalid prediction")
        return Inf
    end
    # Ensure pred and y_voltage have the same length
    min_length = min(length(y_voltage), size(pred, 2))
    l = mean((pred[6, 1:min_length] .- y_voltage[1:min_length]).^2)
    println("Loss: $l")
    return l
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

# Use finite differences for gradient computation
optf = OptimizationFunction(loss, Optimization.AutoFiniteDiff())
optprob = OptimizationProblem(optf, p_combined)

# Solve the optimization problem
res1 = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")








# ------------------------- TESTING THE CODE 

Vout_sim1 = Vout_array
x_time = sol.t

# Define the activation function
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward Neural Network
const U = Lux.Chain(Lux.Dense(6, 10, rbf), # 6 inputs: 5 states + 1 acceleration
                    Lux.Dense(10, 10, rbf), 
                    Lux.Dense(10, 10, rbf),
                    Lux.Dense(10, 5)) # 5 outputs for state corrections


# Initialize the neural network parameters
rng = Random.default_rng()
nn_params, st = Lux.setup(rng, U)
const _st = st

# Extract parameters from AnalyticalModel.p using params_to_namedtuple
model_params_tuple = params_to_namedtuple(AnalyticalModel.p)

# Create p_combined
p_combined = ComponentArray(nn = nn_params, model_params = model_params_tuple)

# Define the UDESystem! function
function UDESystem!(du, u, p, t)
    current_acceleration = Fext_input(t)
    T = eltype(u)
    
    # Create temp_params with the correct type
    temp_params = AnalyticalModel.Params{T}(; p.model_params...)
    
    # Call the analytical model's ODE function
    AnalyticalModel.CoupledSystem!(du, u, temp_params, t, T(current_acceleration))
    
    # Neural network correction
    nn_input = T[u; current_acceleration]
    nn_output = U(nn_input, p.nn, _st)[1]
    du .+= T.(nn_output)
    
    if any(isnan, du) || any(isinf, du)
        println("NaN or Inf detected in du at t=$t")
        println("u = ", u)
        println("du = ", du)
        println("nn_input = ", nn_input)
        println("nn_output = ", nn_output)
        error("NaN or Inf detected in ODE system")
    end
end

# Define the problem
u0 = z0  # initial states defined earlier
tspan = (0.0, 0.5)
prob_nn = ODEProblem(UDESystem!, u0, tspan, p_combined)

function predict(θ, saveat = x_time)
    _prob = remake(prob_nn, p = θ)
    sol = try
        solve(_prob, QNDF(), saveat = saveat, 
              abstol = 1e-8, reltol = 1e-6, maxiters = 1e9)
    catch e
        println("Solver error: ", e)
        return fill(NaN, 6, length(saveat))
    end
    
    if !SciMLBase.successful_retcode(sol)
        println("Solver failed with retcode: $(sol.retcode)")
        return fill(NaN, 6, length(saveat))
    end
    
    println("Solution shape: ", size(sol))
    println("First few timepoints: ", sol.t[1:min(5, length(sol.t))])
    println("First few u values: ", sol.u[1:min(5, length(sol.u))])
    
    t = sol.t
    u = sol(t)
    
    vout = map(1:length(t)) do i
        _, Ctotal = AnalyticalModel.electrostatic(
            u[1,i], u[3,i], u[5,i], 
            θ.model_params.g0, θ.model_params.gp, 
            θ.model_params.a, θ.model_params.e, θ.model_params.ep, 
            θ.model_params.cp, θ.model_params.wt, θ.model_params.wb, 
            θ.model_params.ke, θ.model_params.E, θ.model_params.I, 
            θ.model_params.Leff, θ.model_params.Tf, θ.model_params.Tp, θ.model_params.N
        )
        vout_val = θ.model_params.Vbias - (u[5,i] / Ctotal)
        if i <= 5
            println("i=$i, u[5,i]=$(u[5,i]), Ctotal=$Ctotal, vout_val=$vout_val")
        end
        vout_val
    end
    
    println("vout shape: ", size(vout))
    println("First few vout values: ", vout[1:min(5, length(vout))])
    
    return vcat(u, vout')
end

function loss(θ, p)
    println("Evaluating loss...")
    pred = predict(θ)
    println("Prediction shape: ", size(pred))
    if size(pred, 1) != 6
        println("Invalid prediction shape")
        error("Stopping execution due to invalid prediction shape")
    end
    if any(isnan, pred)
        println("Prediction contains NaN values")
        error("Stopping execution due to NaN values in prediction")
    end
    
    warm_up = 1000  # Adjust this value as needed
    vout_pred = pred[6, warm_up:end]
    vout_sim = Vout_sim1[warm_up:end]
    
    println("vout_pred shape: ", size(vout_pred))
    println("vout_sim shape: ", size(vout_sim))
    
    if length(vout_pred) != length(vout_sim)
        println("Length mismatch: vout_pred ($(length(vout_pred))) vs vout_sim ($(length(vout_sim)))")
        error("Stopping execution due to length mismatch")
    end
    
    l = mean((vout_pred .- vout_sim).^2)
    println("Loss: $l")
    return l
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

function plot_comparison(θ)
    pred = predict(θ)
    if any(isnan, pred)
        println("Prediction contains NaN values. Unable to plot.")
        return
    end
    p = plot(x_time, [pred[6,:], Vout_sim1], label=["UDE Prediction" "Simulation 1"],
             title="Voltage Comparison", xlabel="Time", ylabel="Voltage")
    display(p)
end

# Solve the optimization problem
# Use finite differences for gradient computation
optf = OptimizationFunction(loss, Optimization.AutoFiniteDiff())
optprob = OptimizationProblem(optf, p_combined)

# Solve the optimization problem
try
    res1 = solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 100)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
    
    # Plot the final comparison
    plot_comparison(res1.u)
catch e
    println("Optimization stopped due to error: ", e)
end



