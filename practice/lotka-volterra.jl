# Required imports
using DifferentialEquations, Lux, LuxCore, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Zygote, Plots, Random, Statistics, SciMLSensitivity, LineSearches

# Initialize random number generator
rng = Random.default_rng()

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

rbf(x) = exp.(-(x .^ 2))
# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# Convert parameters to vector format for optimization
θ, reconstruct = LuxCore.vectorize(p)

# Define the hybrid model
function ude_dynamics!(du, u, p_vec, t, p_true)
    # Reconstruct parameters from vector
    p_nested = reconstruct(p_vec)
    û = U(u, p_nested, _st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, θ)

function predict(θ_vec, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ_vec)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

function loss(θ_vec)
    X̂ = predict(θ_vec)
    mean(abs2, Xₙ .- X̂)
end

losses = Float64[]
callback = function (state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

res1 = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(
    optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

# Plot results
X̂_final = predict(p_trained)
plot(t, transpose(X), alpha = 0.75, color = :black, label = ["True x" "True y"])
scatter!(t, transpose(Xₙ), color = :red, alpha = 0.5, label = ["Noisy x" "Noisy y"])
plot!(t, transpose(X̂_final), color = :blue, linewidth = 2, linestyle = :dash, 
      label = ["UDE x" "UDE y"])
xlabel!("Time")
ylabel!("Population")
title!("Lotka-Volterra UDE Results")