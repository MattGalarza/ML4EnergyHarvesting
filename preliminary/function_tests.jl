# Import helpful libraries
using ForwardDiff, DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq
using LineSearches, LinearAlgebra, Statistics, Interpolations, Plots

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

# Create range for state values
x1_range = range(-1.5e-5, 1.5e-5, length = 10000)
x2_range = range(-1.5e-5, 1.5e-5, length = 10000)

# ------------------------------- Suspension Spring + Soft Stopper --------------------------

# Suspension spring force, Fsp
function spring(x1, k1, k3, gss, kss)
    Fsp = -k1 * x1 # Suspension beam force
    if abs(x1) < gss
        Fss = 0.0
    else
        Fss = -kss * (abs(x1) - gss) * sign(x1) # Soft-stopper force
    end
    Fs = Fsp + Fss
    return Fs
end

# Smoothed suspension spring force, Fsp
function spring_smoothed(x1, k1, k3, gss, kss, alpha=1e7)
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

# Calculate spring force for smoothed and non-smoothed
Fs = [spring(x1, k1, k3, gss, kss) for x1 in x1_range]
Fs_smoothed = [spring_smoothed(x1, k1, k3, gss, kss) for x1 in x1_range]

# Plot both smoothed and non-smooth forces
p1 = plot(x1_range, Fs, xlabel = "Displacement (m)", ylabel = "Force (N)", title = "Fs vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(x1_rang1e, Fs_smoothed, xlabel = "Displacement (m)", ylabel = "Force (N)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([-gss, gss], label = "Transition point", linestyle=:dash, color=:red)
display(p1)

# Plot both smoothed and non-smooth forces (close up)
zoom_range = range(-gss-0.5e-6, -gss+0.5e-6, length = 500)
Fs_zoom = [spring(x1, k1, k3, gss, kss) for x1 in zoom_range]
Fs_smoothed_zoom = [spring_smoothed(x1, k1, k3, gss, kss) for x1 in zoom_range]
p2 = plot(zoom_range, Fs_zoom, xlabel = "Displacement (m)", ylabel = "Force (N)", title = "Fs vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(zoom_range, Fs_smoothed_zoom, xlabel = "Displacement (m)", ylabel = "Force (N)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([-gss], label = "Transition point", linestyle=:dash, color=:red)
display(p2)

# ------------------------------- Electrode Restoring + Collision ---------------------------

# Electrode collision force, Fc
function collision(x1, x2, m2, ke, gp)
    if abs(x2) < gp
        m2_out = m2
        Fc = -ke * (x1 - x2)
    else
        m2_out = 2 * m2
        Fc = -ke * (x1 - x2) + ke * (abs(x2) - gp) * sign(x2)
    end
    return m2_out, Fc
end

# Smoothed electrode collision force, Fc
function collision_smoothed(x1, x2, m2, ke, gp, alpha=5e7)
    # Non-collision restoring force
    Fnc = -ke * (x1 - x2)

    # Contact collision force
    S = 0.5 * (1 + tanh(alpha * (abs(x2) - gp))) # Smooth transition function
    m2_effective = m2 * (1 + S) # Calculation of mass
    Fcc = ke * (abs(x2) - gp) * sign(x2) * S

    # Total force
    Fc = Fnc + Fcc
    return m2_effective, Fc
end

# Create a range difference for collision
x1_collision = range(-1.5e-5, 1.5e-5, length = 10000)

# Calculate forces and masses for both smoothed and non-smoothed
Fc = Float64[]
m = Float64[]
Fc_smooth = Float64[]
m_smooth = Float64[]

for i in 1:length(x2_range)
    mass, force = collision(x1_collision[i], x2_range[i], m2, ke, gp)
    mass_smooth, force_smooth = collision_smoothed(x1_collision[i], x2_range[i], m2, ke, gp)
    
    push!(Fc, force)
    push!(m, mass)
    push!(Fc_smooth, force_smooth)
    push!(m_smooth, mass_smooth)
end

# Calculate forces and masses for both smoothed and non-smoothed (close up)
zoom_points = 500
zoom_width = 0.5e-6  # Width of zoom window on each side
zoom_range = range(gp - zoom_width, gp + zoom_width, length = zoom_points)

# Calculate zoom range forces
Fc_zoom = Float64[]
m_zoom = Float64[]
Fc_smooth_zoom = Float64[]
m_smooth_zoom = Float64[]

for i in 1:length(zoom_range)
    mass, force = collision(zoom_range[i], zoom_range[i], m2, ke, gp)
    mass_smooth, force_smooth = collision_smoothed(zoom_range[i], zoom_range[i], m2, ke, gp)
    
    push!(Fc_zoom, force)
    push!(m_zoom, mass)
    push!(Fc_smooth_zoom, force_smooth)
    push!(m_smooth_zoom, mass_smooth)
end

# Plot both smoothed and non-smooth forces
p3 = plot(x2_range, Fc, xlabel = "Displacement (m)", ylabel = "Force (N)", title = "Fc vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(x2_range, Fc_smooth, xlabel = "Displacement (m)", ylabel = "Force (N)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([-gss, gss], label = "Transition point", linestyle=:dash, color=:red)
display(p3)

# Plot both smoothed and non-smooth forces (close up)
p4 = plot(zoom_range, Fc_zoom, xlabel = "Displacement (m)", ylabel = "Force (N)", title = "Fc vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(zoom_range, Fc_smooth_zoom, xlabel = "Displacement (m)", ylabel = "Force (N)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([gp], label = "Transition point", linestyle=:dash, color=:red)
display(p4)

# Plot both smoothed and non-smooth masses
p5 = plot(x2_range, m, xlabel = "Displacement (m)", ylabel = "Mass (kg)", title = "Mass vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(x2_range, m_smooth, xlabel = "Displacement (m)", ylabel = "Mass (kg)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([-gss, gss], label = "Transition point", linestyle=:dash, color=:red)
display(p5)

# Plot both smoothed and non-smooth masses (close up)
p4 = plot(zoom_range, m_zoom, xlabel = "Displacement (m)", ylabel = "Mass (kg)", title = "Mass vs Displacement: Original vs Smoothed", label = "Original", linewidth=2)
plot!(zoom_range, m_smooth_zoom, xlabel = "Displacement (m)", ylabel = "Mass (kg)", label = "Smoothed", linewidth=2) # linestyle=:dash
vline!([gp], label = "Transition point", linestyle=:dash, color=:red)
display(p4)

# ----------------------------------- Electrostatic Coupling --------------------------------

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

# Smoothed electrostatic coupling, Fe
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

