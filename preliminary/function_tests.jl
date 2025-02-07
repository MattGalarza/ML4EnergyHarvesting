# Import helpful libraries
using Sundials, ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics, Interpolations, Plots

# --------------------------------------- Model Parameters ----------------------------------

# Independent parameters
m1 = 2.0933e-6        # Shuttle mass (kg)
E = 180e9             # Young's modulus (Pa)
eta = 1.849e-5        # Dynamic viscosity of air (Pa·s)
c = 0.015             # Damping scaling factor
g0 = 14e-6            # Electrode gap (m)
Tp = 120e-9           # Thickness of parylene layer (m)
Tf = 25e-6            # Device thickness (m)
gss = 14e-6           # Soft-stopper initial gap (m)
rho = 2333.0          # Density of silicon (kg/m³)
cp = 5e-12            # Capacitance of parylene (F)
wt = 9e-6             # Electrode width, top (m)
wb = 30e-6            # Electrode width, bottom (m)
ws = 14.7e-6          # Suspension spring width (m)
Lss = 1400e-6         # Suspension spring length (m)
Lff = 450e-6          # Electrode length (m)
Leff = 400e-6         # Electrode effective overlap length (m)
e = 8.85e-12          # Permittivity of free space (F/m)
ep = 3.2              # Permittivity of parylene
Vbias = 3.0           # Bias voltage (V)
Rload = 0.42e6        # Load resistance (Ω)
N = 160               # Number of electrodes
kss = 6.0             # Soft-stopper spring force (N/m)

# Dependent parameters
gp = ((g0 - 2 * Tp))                                  # Initial gap including parylene layer (m)
wavg = (((wt + wb) / 2))                              # Average electrode width (m)
a = (((wb - wt) / Leff))                              # Tilt factor
k1 = (((4 / 6) * ((E * Tf * (ws^3)) / (Lss^3))))      # Spring constant (N/m)
k3 = (((18 / 25) * ((E * Tf * ws) / (Lss^3))))        # Cubic spring constant (N/m³)
I = (((1 / 12) * Tf * (wavg^3)))                      # Electrode moment of inertia (m⁴)
m2 = (((33 / 140) * rho * Tf * Lff * wavg))           # Mass of electrode (kg)
ke = (((1 / 4) * E / Lff^3 * I))                      # Electrode spring constant (N/m)

# ------------------------------- Suspension Spring + Soft Stopper --------------------------

# Suspension spring force, Fsp
function spring(x1, k1, k3, gss, kss)
    # Fsp = - k1 * x1 - k3 * (x1^3) # Suspension beam force
    Fsp = -k1 * x1 # Suspension beam force
    if abs(x1) < gss
        Fss = 0.0
    else
        Fss = -kss * (abs(x1) - gss) # Soft-stopper force
    end
    Fs = Fsp + Fss
    return Fs
end



# Define the spring function (your original function)
function spring(x1, k1, k3, gss, kss)
    Fsp = -k1 * x1 # Suspension beam force
    if abs(x1) < gss
        Fss = 0.0
    else
        Fss = -kss * (abs(x1) - gss) # Soft-stopper force
    end
    Fs = Fsp + Fss
    return Fs
end

# Parameters (from your original code)
E = 180e9             # Young's modulus (Pa)
Tf = 25e-6            # Device thickness (m)
ws = 14.7e-6          # Suspension spring width (m)
Lss = 1400e-6         # Suspension spring length (m)
gss = 14e-6           # Soft-stopper initial gap (m)
kss = 6.0             # Soft-stopper spring force (N/m)

# Calculate k1 (suspension beam spring constant)
k1 = (4/6) * ((E * Tf * ws^3) / Lss^3)

# Calculate k3 (not used in current version but included for completeness)
k3 = (18/25) * ((E * Tf * ws) / Lss^3)

# Create range of x1 values
x1_range = range(-1.5e-5, 1.5e-5, length=1000)

# Calculate spring force for each x1
forces = [spring(x1, k1, k3, gss, kss) for x1 in x1_range]

# Create plot
p = plot(x1_range .* 1e6, forces,  # Convert x1 to micrometers for better readability
    xlabel="Displacement (μm)",
    ylabel="Force (N)",
    title="Spring Force vs Displacement",
    label="Spring Force",
    linewidth=2,
    grid=true)

# Add vertical lines at soft-stopper engagement points
vline!([-gss * 1e6, gss * 1e6], label="Soft-stopper engagement", linestyle=:dash, color=:red)

display(p)

# Print some key values
println("\nKey Parameters:")
println("k1 = $(k1) N/m")
println("k3 = $(k3) N/m³")
println("gss = $(gss * 1e6) μm")
println("kss = $(kss) N/m")

# Print force values at specific points
test_points = [-1.5e-5, -gss, 0.0, gss, 1.5e-5]
println("\nForce values at key points:")
for x in test_points
    f = spring(x, k1, k3, gss, kss)
    println("x = $(x * 1e6) μm: F = $f N")
end

# --------------------------------------- Analytical Model ----------------------------------
