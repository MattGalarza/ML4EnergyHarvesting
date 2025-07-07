# ============================================================================
# COMPLETE DELAY AUTOENCODER IMPLEMENTATION
# Two Methods: Deep Delay vs Forced Delay
# Robust for any dynamical system data
# ============================================================================

using Flux
using DifferentialEquations
using LinearAlgebra
using Statistics
using Random
using Plots
using BSON
using Zygote
using MLJ
using GLM

# ============================================================================
# COMMON UTILITIES
# ============================================================================

"""
Create time-delay embedding from time series data
"""
function create_delay_embedding(data::AbstractMatrix, n_delays::Int, τ::Int)
    """
    data: (n_samples, n_variables) 
    n_delays: number of delay coordinates
    τ: delay time step
    Returns: (n_samples - (n_delays-1)*τ, n_variables * n_delays)
    """
    n_samples, n_vars = size(data)
    n_output_samples = n_samples - (n_delays - 1) * τ
    
    if n_output_samples <= 0
        error("Not enough data points for delay embedding. Need at least $((n_delays-1)*τ + 1) samples.")
    end
    
    embedded = zeros(Float32, n_output_samples, n_vars * n_delays)
    
    for i in 1:n_output_samples
        for j in 1:n_delays
            idx = i + (j - 1) * τ
            start_col = (j - 1) * n_vars + 1
            end_col = j * n_vars
            embedded[i, start_col:end_col] = data[idx, :]
        end
    end
    
    return embedded
end

"""
Compute time derivatives using finite differences
"""
function compute_derivatives(data::AbstractMatrix, dt::Float32)
    """
    data: (n_samples, n_variables)
    dt: time step
    Returns: (n_samples-1, n_variables) derivatives
    """
    n_samples, n_vars = size(data)
    derivatives = zeros(Float32, n_samples - 1, n_vars)
    
    for i in 1:(n_samples-1)
        derivatives[i, :] = (data[i+1, :] - data[i, :]) / dt
    end
    
    return derivatives
end

"""
Create polynomial feature library for SINDy
"""
function create_polynomial_library(data::AbstractMatrix, max_degree::Int=3, 
                                 include_constant::Bool=true, force_data::Union{AbstractMatrix,Nothing}=nothing)
    """
    data: (n_samples, n_state_vars)
    max_degree: maximum polynomial degree
    include_constant: whether to include constant term
    force_data: (n_samples, n_force_vars) external forcing (optional)
    Returns: (n_samples, n_features) feature matrix
    """
    n_samples, n_vars = size(data)
    features = Vector{Matrix{Float32}}()
    feature_names = Vector{String}()
    
    # Constant term
    if include_constant
        push!(features, ones(Float32, n_samples, 1))
        push!(feature_names, "1")
    end
    
    # Linear terms (states)
    push!(features, data)
    for i in 1:n_vars
        push!(feature_names, "z$i")
    end
    
    # Force terms (if provided)
    if force_data !== nothing
        n_force_vars = size(force_data, 2)
        push!(features, force_data)
        for i in 1:n_force_vars
            push!(feature_names, "F$i")
        end
    end
    
    # Polynomial terms up to max_degree
    if max_degree >= 2
        # Quadratic terms
        for i in 1:n_vars
            for j in i:n_vars
                if i == j
                    new_feature = data[:, i:i] .^ 2
                    push!(features, new_feature)
                    push!(feature_names, "z$(i)^2")
                else
                    new_feature = data[:, i:i] .* data[:, j:j]
                    push!(features, new_feature)
                    push!(feature_names, "z$(i)*z$(j)")
                end
            end
        end
        
        # Force-state interaction terms
        if force_data !== nothing
            n_force_vars = size(force_data, 2)
            for i in 1:n_vars
                for j in 1:n_force_vars
                    new_feature = data[:, i:i] .* force_data[:, j:j]
                    push!(features, new_feature)
                    push!(feature_names, "z$(i)*F$(j)")
                end
            end
        end
    end
    
    # Higher degree terms (cubic, etc.)
    if max_degree >= 3
        for i in 1:n_vars
            for j in i:n_vars
                for k in j:n_vars
                    new_feature = data[:, i:i] .* data[:, j:j] .* data[:, k:k]
                    push!(features, new_feature)
                    if i == j == k
                        push!(feature_names, "z$(i)^3")
                    elseif i == j
                        push!(feature_names, "z$(i)^2*z$(k)")
                    elseif j == k
                        push!(feature_names, "z$(i)*z$(j)^2")
                    else
                        push!(feature_names, "z$(i)*z$(j)*z$(k)")
                    end
                end
            end
        end
    end
    
    return hcat(features...), feature_names
end

"""
Sparse regression using LASSO (L1 regularization)
"""
function sparse_regression(X::AbstractMatrix, y::AbstractVector, λ::Float32=0.01f0)
    """
    X: (n_samples, n_features) feature matrix
    y: (n_samples,) target vector
    λ: L1 regularization parameter
    Returns: coefficient vector
    """
    # Simple iterative soft thresholding for LASSO
    n_features = size(X, 2)
    β = zeros(Float32, n_features)
    
    # Normalize features
    X_norm = X ./ (sqrt.(sum(X.^2, dims=1)) .+ 1e-8)
    
    # Iterative soft thresholding
    for iter in 1:1000
        β_old = copy(β)
        
        for j in 1:n_features
            # Partial residual
            r = y - X_norm * β + X_norm[:, j] * β[j]
            
            # Soft thresholding
            ρ = dot(X_norm[:, j], r)
            
            if ρ > λ
                β[j] = ρ - λ
            elseif ρ < -λ
                β[j] = ρ + λ
            else
                β[j] = 0.0f0
            end
        end
        
        # Check convergence
        if norm(β - β_old) < 1e-6
            break
        end
    end
    
    # Denormalize coefficients
    β_denorm = β ./ (sqrt.(sum(X.^2, dims=1))' .+ 1e-8)
    
    return β_denorm
end

"""
Apply SINDy to discover sparse dynamics
"""
function apply_sindy(Z::AbstractMatrix, Z_dot::AbstractMatrix, λ::Float32=0.01f0, 
                    max_degree::Int=3, force_data::Union{AbstractMatrix,Nothing}=nothing)
    """
    Z: (n_samples, n_states) state data
    Z_dot: (n_samples, n_states) derivative data
    λ: sparsity parameter
    max_degree: maximum polynomial degree
    force_data: external forcing data (optional)
    Returns: (coefficients, feature_names, discovered_equations)
    """
    # Create feature library
    Θ, feature_names = create_polynomial_library(Z, max_degree, true, force_data)
    
    n_states = size(Z, 2)
    coefficients = zeros(Float32, length(feature_names), n_states)
    discovered_equations = Vector{String}()
    
    # Discover equation for each state
    for i in 1:n_states
        coeffs = sparse_regression(Θ, Z_dot[:, i], λ)
        coefficients[:, i] = coeffs
        
        # Create equation string
        equation_terms = String[]
        for (j, coeff) in enumerate(coeffs)
            if abs(coeff) > 1e-6  # Threshold for significance
                sign = coeff >= 0 ? "+" : "-"
                if length(equation_terms) == 0 && sign == "+"
                    sign = ""
                end
                push!(equation_terms, "$(sign)$(round(abs(coeff), digits=4))*$(feature_names[j])")
            end
        end
        
        if isempty(equation_terms)
            equation = "0"
        else
            equation = join(equation_terms, " ")
            equation = replace(equation, "*1" => "")  # Clean up constant terms
        end
        
        push!(discovered_equations, "dz$(i)/dt = $equation")
    end
    
    return coefficients, feature_names, discovered_equations
end

# ============================================================================
# METHOD 1: DEEP DELAY AUTOENCODER (from paper)
# For autonomous systems only
# ============================================================================

struct DeepDelayAutoencoder
    encoder::Chain
    decoder::Chain
    config::NamedTuple
end

function DeepDelayAutoencoder(input_dim::Int, latent_dim::Int, hidden_dims::Vector{Int}=[64, 32])
    """
    input_dim: dimension of delay embedding
    latent_dim: dimension of latent space
    hidden_dims: hidden layer dimensions
    """
    
    # Encoder
    encoder_layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims
        push!(encoder_layers, Dense(prev_dim, hidden_dim, tanh))
        prev_dim = hidden_dim
    end
    push!(encoder_layers, Dense(prev_dim, latent_dim))
    encoder = Chain(encoder_layers...)
    
    # Decoder
    decoder_layers = []
    prev_dim = latent_dim
    for hidden_dim in reverse(hidden_dims)
        push!(decoder_layers, Dense(prev_dim, hidden_dim, tanh))
        prev_dim = hidden_dim
    end
    push!(decoder_layers, Dense(prev_dim, input_dim))
    decoder = Chain(decoder_layers...)
    
    config = (
        input_dim = input_dim,
        latent_dim = latent_dim,
        hidden_dims = hidden_dims
    )
    
    return DeepDelayAutoencoder(encoder, decoder, config)
end

function (model::DeepDelayAutoencoder)(h::AbstractMatrix)
    """Forward pass through autoencoder"""
    z = model.encoder(h')  # (latent_dim, batch)
    h_reconstructed = model.decoder(z)  # (input_dim, batch)
    return z', h_reconstructed'  # Return (batch, dim) format
end

"""
Multi-objective loss for Deep Delay Autoencoder
"""
function deep_delay_loss(model::DeepDelayAutoencoder, h_batch::AbstractMatrix, h_dot_batch::AbstractMatrix, 
                        y_batch::AbstractMatrix, sindy_coeffs::AbstractMatrix, feature_matrix::AbstractMatrix;
                        λ_recon::Float32=1.0f0, λ_sindy_z::Float32=1.0f0, λ_sindy_h::Float32=1.0f0, 
                        λ_z1::Float32=10.0f0, λ_sparse::Float32=0.001f0)
    """
    h_batch: (batch, delay_embedding_dim) delay embedded data
    h_dot_batch: (batch, delay_embedding_dim) derivatives of delay embedding
    y_batch: (batch, 1) original observable
    sindy_coeffs: (n_features, latent_dim) SINDy coefficients
    feature_matrix: (batch, n_features) polynomial features of latent variables
    """
    
    batch_size = size(h_batch, 1)
    latent_dim = model.config.latent_dim
    
    # Forward pass
    z, h_recon = model(h_batch)
    
    # Reconstruction loss
    L_recon = mean((h_batch - h_recon).^2)
    
    # Compute z_dot using automatic differentiation
    z_dot = similar(z)
    for i in 1:batch_size
        h_i = h_batch[i:i, :]
        z_dot[i:i, :] = Zygote.jacobian(x -> model.encoder(x'), h_i')[1] * h_dot_batch[i:i, :]'
    end
    
    # SINDy loss in z space
    sindy_pred = feature_matrix * sindy_coeffs  # (batch, latent_dim)
    L_sindy_z = mean((z_dot - sindy_pred).^2)
    
    # SINDy loss in h space (consistency)
    h_dot_pred = similar(h_dot_batch)
    for i in 1:batch_size
        z_i = z[i:i, :]
        decoder_jacobian = Zygote.jacobian(x -> model.decoder(x'), z_i')[1]
        h_dot_pred[i:i, :] = (decoder_jacobian * sindy_pred[i:i, :]')'
    end
    L_sindy_h = mean((h_dot_batch - h_dot_pred).^2)
    
    # First component constraint (z1 should equal original observable y)
    L_z1 = mean((z[:, 1:1] - y_batch).^2)
    
    # Sparsity loss
    L_sparse = sum(abs.(sindy_coeffs))
    
    # Total loss
    total_loss = λ_recon * L_recon + λ_sindy_z * L_sindy_z + λ_sindy_h * L_sindy_h + 
                λ_z1 * L_z1 + λ_sparse * L_sparse
    
    return total_loss, (
        recon = L_recon,
        sindy_z = L_sindy_z,
        sindy_h = L_sindy_h,
        z1 = L_z1,
        sparse = L_sparse,
        total = total_loss
    )
end

# ============================================================================
# METHOD 2: FORCED DELAY AUTOENCODER (our adaptation)
# For systems with external forcing
# ============================================================================

struct ForcedDelayAutoencoder
    encoder::Chain
    decoder::Chain
    config::NamedTuple
end

function ForcedDelayAutoencoder(observable_dim::Int, force_dim::Int, n_delays::Int, 
                               latent_dim::Int, hidden_dims::Vector{Int}=[64, 32])
    """
    observable_dim: dimension of observable (usually 1)
    force_dim: dimension of forcing (usually 1)
    n_delays: number of delay coordinates
    latent_dim: dimension of latent space
    hidden_dims: hidden layer dimensions
    """
    
    input_dim = (observable_dim + force_dim) * n_delays
    
    # Encoder
    encoder_layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims
        push!(encoder_layers, Dense(prev_dim, hidden_dim, tanh))
        prev_dim = hidden_dim
    end
    push!(encoder_layers, Dense(prev_dim, latent_dim))
    encoder = Chain(encoder_layers...)
    
    # Decoder (reconstructs combined [observable, force] delay embedding)
    decoder_layers = []
    prev_dim = latent_dim
    for hidden_dim in reverse(hidden_dims)
        push!(decoder_layers, Dense(prev_dim, hidden_dim, tanh))
        prev_dim = hidden_dim
    end
    push!(decoder_layers, Dense(prev_dim, input_dim))
    decoder = Chain(decoder_layers...)
    
    config = (
        observable_dim = observable_dim,
        force_dim = force_dim,
        n_delays = n_delays,
        input_dim = input_dim,
        latent_dim = latent_dim,
        hidden_dims = hidden_dims
    )
    
    return ForcedDelayAutoencoder(encoder, decoder, config)
end

function (model::ForcedDelayAutoencoder)(combined_embedding::AbstractMatrix)
    """Forward pass through forced autoencoder"""
    z = model.encoder(combined_embedding')  # (latent_dim, batch)
    embedding_recon = model.decoder(z)  # (input_dim, batch)
    return z', embedding_recon'  # Return (batch, dim) format
end

"""
Multi-objective loss for Forced Delay Autoencoder
"""
function forced_delay_loss(model::ForcedDelayAutoencoder, combined_embedding::AbstractMatrix, 
                          combined_embedding_dot::AbstractMatrix, y_batch::AbstractMatrix, 
                          force_batch::AbstractMatrix, sindy_coeffs::AbstractMatrix, 
                          feature_matrix::AbstractMatrix;
                          λ_recon::Float32=1.0f0, λ_sindy_z::Float32=1.0f0, λ_sindy_combined::Float32=1.0f0,
                          λ_y1::Float32=10.0f0, λ_sparse::Float32=0.001f0)
    """
    Similar to deep_delay_loss but handles combined [observable, force] embedding
    """
    
    batch_size = size(combined_embedding, 1)
    latent_dim = model.config.latent_dim
    observable_dim = model.config.observable_dim
    
    # Forward pass
    z, combined_recon = model(combined_embedding)
    
    # Reconstruction loss
    L_recon = mean((combined_embedding - combined_recon).^2)
    
    # Compute z_dot using automatic differentiation
    z_dot = similar(z)
    for i in 1:batch_size
        embedding_i = combined_embedding[i:i, :]
        z_dot[i:i, :] = Zygote.jacobian(x -> model.encoder(x'), embedding_i')[1] * combined_embedding_dot[i:i, :]'
    end
    
    # SINDy loss in z space
    sindy_pred = feature_matrix * sindy_coeffs  # (batch, latent_dim)
    L_sindy_z = mean((z_dot - sindy_pred).^2)
    
    # SINDy loss in combined embedding space (consistency)
    combined_dot_pred = similar(combined_embedding_dot)
    for i in 1:batch_size
        z_i = z[i:i, :]
        decoder_jacobian = Zygote.jacobian(x -> model.decoder(x'), z_i')[1]
        combined_dot_pred[i:i, :] = (decoder_jacobian * sindy_pred[i:i, :]')'
    end
    L_sindy_combined = mean((combined_embedding_dot - combined_dot_pred).^2)
    
    # First observable constraint (first observable component should match)
    # Extract first observable from reconstructed embedding
    obs_recon_first = combined_recon[:, 1:observable_dim]  # First observable component
    L_y1 = mean((obs_recon_first - y_batch[:, 1:observable_dim]).^2)
    
    # Sparsity loss
    L_sparse = sum(abs.(sindy_coeffs))
    
    # Total loss
    total_loss = λ_recon * L_recon + λ_sindy_z * L_sindy_z + λ_sindy_combined * L_sindy_combined + 
                λ_y1 * L_y1 + λ_sparse * L_sparse
    
    return total_loss, (
        recon = L_recon,
        sindy_z = L_sindy_z,
        sindy_combined = L_sindy_combined,
        y1 = L_y1,
        sparse = L_sparse,
        total = total_loss
    )
end

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

"""
Training function for Deep Delay Autoencoder (Method 1)
"""
function train_deep_delay_autoencoder(y_data::AbstractVector, dt::Float32; 
                                     n_delays::Int=20, τ::Int=1, latent_dim::Int=3,
                                     hidden_dims::Vector{Int}=[64, 32], n_epochs::Int=1000,
                                     learning_rate::Float32=1e-3, batch_size::Int=64,
                                     λ_recon::Float32=1.0f0, λ_sindy_z::Float32=1.0f0, 
                                     λ_sindy_h::Float32=1.0f0, λ_z1::Float32=10.0f0, 
                                     λ_sparse::Float32=0.001f0, sindy_λ::Float32=0.01f0)
    """
    Train Deep Delay Autoencoder on single observable time series
    
    y_data: (n_samples,) single observable time series
    dt: time step
    """
    
    println("Training Deep Delay Autoencoder...")
    println("Data length: $(length(y_data)), n_delays: $n_delays, τ: $τ")
    
    # Create delay embedding
    y_matrix = reshape(y_data, :, 1)  # Convert to matrix format
    h = create_delay_embedding(y_matrix, n_delays, τ)
    println("Delay embedding shape: $(size(h))")
    
    # Compute derivatives
    h_dot = compute_derivatives(h, dt * τ)
    h = h[1:end-1, :]  # Match derivative dimensions
    
    # Extract original observable for constraint
    y_constrained = y_matrix[1:size(h, 1), :]
    
    # Create model
    input_dim = size(h, 2)
    model = DeepDelayAutoencoder(input_dim, latent_dim, hidden_dims)
    
    # Setup optimizer
    optimizer = ADAM(learning_rate)
    
    # Training data
    n_samples = size(h, 1)
    n_batches = ceil(Int, n_samples / batch_size)
    
    # Training loop
    losses = []
    
    for epoch in 1:n_epochs
        epoch_loss = 0.0f0
        
        # Shuffle data
        indices = randperm(n_samples)
        
        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            h_batch = h[batch_indices, :]
            h_dot_batch = h_dot[batch_indices, :]
            y_batch = y_constrained[batch_indices, :]
            
            # Get current latent representation for SINDy
            z_current, _ = model(h_batch)
            
            # Create feature matrix and update SINDy coefficients
            feature_matrix, feature_names = create_polynomial_library(z_current, 3, true, nothing)
            
            # Update SINDy coefficients
            sindy_coeffs = zeros(Float32, size(feature_matrix, 2), latent_dim)
            for i in 1:latent_dim
                z_dot_approx = compute_derivatives(z_current[:, i:i], dt * τ)
                if size(z_dot_approx, 1) > 0
                    coeffs = sparse_regression(feature_matrix[1:end-1, :], z_dot_approx[:, 1], sindy_λ)
                    sindy_coeffs[:, i] = coeffs
                end
            end
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(Flux.params(model.encoder, model.decoder)) do
                total_loss, loss_components = deep_delay_loss(
                    model, h_batch, h_dot_batch, y_batch, sindy_coeffs, feature_matrix,
                    λ_recon=λ_recon, λ_sindy_z=λ_sindy_z, λ_sindy_h=λ_sindy_h, 
                    λ_z1=λ_z1, λ_sparse=λ_sparse
                )
                return total_loss
            end
            
            # Update parameters
            Flux.Optimise.update!(optimizer, Flux.params(model.encoder, model.decoder), grads)
            
            epoch_loss += loss
        end
        
        avg_loss = epoch_loss / n_batches
        push!(losses, avg_loss)
        
        if epoch % 100 == 0
            println("Epoch $epoch: Average Loss = $(round(avg_loss, digits=6))")
        end
    end
    
    # Final SINDy discovery
    z_final, _ = model(h)
    coefficients, feature_names, equations = apply_sindy(z_final, compute_derivatives(z_final, dt * τ), sindy_λ, 3, nothing)
    
    println("\nDiscovered Equations:")
    for eq in equations
        println(eq)
    end
    
    return model, coefficients, feature_names, equations, losses
end

"""
Training function for Forced Delay Autoencoder (Method 2)
"""
function train_forced_delay_autoencoder(y_data::AbstractVector, force_data::AbstractVector, dt::Float32;
                                      n_delays::Int=20, τ::Int=1, latent_dim::Int=3,
                                      hidden_dims::Vector{Int}=[64, 32], n_epochs::Int=1000,
                                      learning_rate::Float32=1e-3, batch_size::Int=64,
                                      λ_recon::Float32=1.0f0, λ_sindy_z::Float32=1.0f0,
                                      λ_sindy_combined::Float32=1.0f0, λ_y1::Float32=10.0f0,
                                      λ_sparse::Float32=0.001f0, sindy_λ::Float32=0.01f0)
    """
    Train Forced Delay Autoencoder on observable + forcing time series
    
    y_data: (n_samples,) observable time series
    force_data: (n_samples,) forcing time series
    dt: time step
    """
    
    println("Training Forced Delay Autoencoder...")
    println("Data length: $(length(y_data)), n_delays: $n_delays, τ: $τ")
    
    # Combine observable and force data
    combined_data = hcat(reshape(y_data, :, 1), reshape(force_data, :, 1))
    
    # Create combined delay embedding
    combined_embedding = create_delay_embedding(combined_data, n_delays, τ)
    println("Combined embedding shape: $(size(combined_embedding))")
    
    # Compute derivatives
    combined_embedding_dot = compute_derivatives(combined_embedding, dt * τ)
    combined_embedding = combined_embedding[1:end-1, :]  # Match derivative dimensions
    
    # Extract original data for constraints
    y_constrained = reshape(y_data[1:size(combined_embedding, 1)], :, 1)
    force_constrained = reshape(force_data[1:size(combined_embedding, 1)], :, 1)
    
    # Create model
    model = ForcedDelayAutoencoder(1, 1, n_delays, latent_dim, hidden_dims)
    
    # Setup optimizer
    optimizer = ADAM(learning_rate)
    
    # Training data
    n_samples = size(combined_embedding, 1)
    n_batches = ceil(Int, n_samples / batch_size)
    
    # Training loop
    losses = []
    
    for epoch in 1:n_epochs
        epoch_loss = 0.0f0
        
        # Shuffle data
        indices = randperm(n_samples)
        
        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            combined_batch = combined_embedding[batch_indices, :]
            combined_dot_batch = combined_embedding_dot[batch_indices, :]
            y_batch = y_constrained[batch_indices, :]
            force_batch = force_constrained[batch_indices, :]
            
            # Get current latent representation for SINDy
            z_current, _ = model(combined_batch)
            
            # Create feature matrix including force terms
            feature_matrix, feature_names = create_polynomial_library(z_current, 3, true, force_batch)
            
            # Update SINDy coefficients
            sindy_coeffs = zeros(Float32, size(feature_matrix, 2), latent_dim)
            for i in 1:latent_dim
                z_dot_approx = compute_derivatives(z_current[:, i:i], dt * τ)
                if size(z_dot_approx, 1) > 0 && size(feature_matrix[1:end-1, :], 1) > 0
                    coeffs = sparse_regression(feature_matrix[1:end-1, :], z_dot_approx[:, 1], sindy_λ)
                    sindy_coeffs[:, i] = coeffs
                end
            end
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(Flux.params(model.encoder, model.decoder)) do
                total_loss, loss_components = forced_delay_loss(
                    model, combined_batch, combined_dot_batch, y_batch, force_batch, 
                    sindy_coeffs, feature_matrix,
                    λ_recon=λ_recon, λ_sindy_z=λ_sindy_z, λ_sindy_combined=λ_sindy_combined,
                    λ_y1=λ_y1, λ_sparse=λ_sparse
                )
                return total_loss
            end
            
            # Update parameters
            Flux.Optimise.update!(optimizer, Flux.params(model.encoder, model.decoder), grads)
            
            epoch_loss += loss
        end
        
        avg_loss = epoch_loss / n_batches
        push!(losses, avg_loss)
        
        if epoch % 100 == 0
            println("Epoch $epoch: Average Loss = $(round(avg_loss, digits=6))")
        end
    end
    
    # Final SINDy discovery
    z_final, _ = model(combined_embedding)
    force_for_sindy = force_constrained[1:size(z_final, 1), :]
    coefficients, feature_names, equations = apply_sindy(z_final, compute_derivatives(z_final, dt * τ), sindy_λ, 3, force_for_sindy)
    
    println("\nDiscovered Equations:")
    for eq in equations
        println(eq)
    end
    
    return model, coefficients, feature_names, equations, losses
end

# ============================================================================
# TEST SYSTEM: LORENZ ATTRACTOR
# ============================================================================

"""
Generate Lorenz attractor data
"""
function generate_lorenz_data(; σ=10.0, ρ=28.0, β=8.0/3.0, dt=0.01, T=50.0, 
                             initial_condition=[1.0, 1.0, 1.0], add_noise=false, noise_level=0.01)
    """
    Generate Lorenz attractor data for testing
    Returns: (t, x, y, z, F) where F is zero (autonomous system)
    """
    
    function lorenz!(du, u, p, t)
        σ, ρ, β = p
        du[1] = σ * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
    end
    
    tspan = (0.0, T)
    prob = ODEProblem(lorenz!, initial_condition, tspan, [σ, ρ, β])
    sol = solve(prob, Tsit5(), saveat=dt)
    
    t = sol.t
    states = hcat(sol.u...)'  # (n_samples, 3)
    
    if add_noise
        states += noise_level * randn(size(states))
    end
    
    # For Lorenz (autonomous), forcing is zero
    F = zeros(length(t))
    
    return Float32.(t), Float32.(states[:, 1]), Float32.(states[:, 2]), Float32.(states[:, 3]), Float32.(F)
end

"""
Test both methods on Lorenz data
"""
function test_both_methods_lorenz()
    """
    Test both Deep Delay and Forced Delay methods on Lorenz attractor
    """
    
    println("="^70)
    println("TESTING BOTH METHODS ON LORENZ ATTRACTOR")
    println("="^70)
    
    # Generate Lorenz data
    println("Generating Lorenz attractor data...")
    t, x, y, z, F = generate_lorenz_data(dt=0.01, T=20.0)
    
    # Use x (first component) as observable
    y_obs = x
    dt = Float32(0.01)
    
    println("Generated $(length(t)) data points")
    println("Observable range: [$(minimum(y_obs)), $(maximum(y_obs))]")
    
    # Method 1: Deep Delay Autoencoder (autonomous)
    println("\n" * "="^50)
    println("METHOD 1: DEEP DELAY AUTOENCODER")
    println("="^50)
    
    model1, coeffs1, names1, eqs1, losses1 = train_deep_delay_autoencoder(
        y_obs, dt,
        n_delays=20, τ=1, latent_dim=3, n_epochs=500,
        λ_z1=1.0f0, λ_sparse=0.01f0
    )
    
    # Method 2: Forced Delay Autoencoder (with F=0 for fair comparison)
    println("\n" * "="^50)
    println("METHOD 2: FORCED DELAY AUTOENCODER (F=0)")
    println("="^50)
    
    model2, coeffs2, names2, eqs2, losses2 = train_forced_delay_autoencoder(
        y_obs, F, dt,  # F is zero for Lorenz
        n_delays=20, τ=1, latent_dim=3, n_epochs=500,
        λ_y1=1.0f0, λ_sparse=0.01f0
    )
    
    # Compare results
    println("\n" * "="^70)
    println("COMPARISON OF RESULTS")
    println("="^70)
    
    println("\nMethod 1 (Deep Delay) Equations:")
    for eq in eqs1
        println("  $eq")
    end
    
    println("\nMethod 2 (Forced Delay) Equations:")
    for eq in eqs2
        println("  $eq")
    end
    
    # Plot training curves
    p1 = plot(losses1, title="Method 1: Deep Delay", xlabel="Epoch", ylabel="Loss", yscale=:log10)
    p2 = plot(losses2, title="Method 2: Forced Delay", xlabel="Epoch", ylabel="Loss", yscale=:log10)
    combined_plot = plot(p1, p2, layout=(1, 2), size=(800, 300))
    
    display(combined_plot)
    
    return (model1, coeffs1, names1, eqs1, losses1), (model2, coeffs2, names2, eqs2, losses2)
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running comparative test on Lorenz attractor...")
    Random.seed!(42)  # For reproducibility
    
    result1, result2 = test_both_methods_lorenz()
    
    println("\nTest completed! Check the discovered equations above.")
    println("Both methods should discover similar dynamics for the autonomous Lorenz system.")
end