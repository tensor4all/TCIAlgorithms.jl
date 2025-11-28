"""
    rank_cost(P; τ, method=:svd)

Rank-like cost of pivot matrix `P`:

- `method = :svd`      → numerical rank Rank_τ(P)
- `method = :nuclear`  → nuclear norm ‖P‖_* (optionally with thresholding)
"""
function rank_cost(P::AbstractMatrix; τ::Real = 1e-10, method::Symbol = :svd)
    s = svdvals(P)

    if method === :svd
        return count(>(τ), s)          # Rank_τ
    elseif method === :nuclear
        # if you want thresholding:
        # s_eff = s[s .> τ]
        # return sum(s_eff)
        return sum(s)                  # nuclear norm
    else
        error("Unknown rank method: $method (expected :svd or :nuclear)")
    end
end

"""
    fullindex_from_ij(i, j)

Reconstruct full index σ = i ⊕ j from left and right MultiIndices.
Adjust this if your `MultiIndex` isn’t just `Vector{Int}`.
"""
fullindex_from_ij(i::MultiIndex, j::MultiIndex) = vcat(i, j)

"""
    pivot_matrix_for_site(f, tci, ℓ, v, ℓ_max)

Construct the pivot matrix P(ℓ, v) using the pivot multi-indices
at the maximally entangled bond ℓ_max.
- `ℓ` is a site index in the *TCI* (so between 1 and length(tci.localdims)).
- `v` is the fixed value at site ℓ (in 1:localdims[ℓ]).
"""
function pivot_matrix_for_site(
    f,
    tci::TensorCI2{T},
    ℓ::Int,
    v::Int,
    ℓ_max::Int,
) where {T}
    Iℓmax  = tci.Iset[ℓ_max+1]
    Jℓmax1 = tci.Jset[ℓ_max]

    χ_max = length(Iℓmax)
    length(Jℓmax1) == χ_max ||
        error("Inconsistent Iset/Jset at ℓ_max: $(χ_max) vs $(length(Jℓmax1))")

    P = Matrix{T}(undef, χ_max, χ_max)

    @inbounds for (r, i) in enumerate(Iℓmax)
        for (c, j) in enumerate(Jℓmax1)
            σ  = fullindex_from_ij(i, j)
            σ̃ = copy(σ)
            σ̃[ℓ] = v
            P[r, c] = f(σ̃)
        end
    end

    return P
end

"""
    reference_pivot_matrix(f, tci, ℓ_max)

Construct the reference pivot matrix P_{ℓ_max} (no overwrite).
"""
function reference_pivot_matrix(f, tci::TensorCI2{T}, ℓ_max::Int) where {T}
    Iℓmax  = tci.Iset[ℓ_max+1]
    Jℓmax1 = tci.Jset[ℓ_max]
    χ_max  = length(Iℓmax)

    length(Jℓmax1) == χ_max ||
        error("Inconsistent Iset/Jset at ℓ_max")

    P = Matrix{T}(undef, χ_max, χ_max)
    @inbounds for (r, i) in enumerate(Iℓmax)
        for (c, j) in enumerate(Jℓmax1)
            σ = fullindex_from_ij(i, j)
            P[r, c] = f(σ)
        end
    end
    return P
end

"""
    choose_optimal_patching_site(tci, f; τ, rank_method)

Implement Algorithm 1 on a TensorCI2.

- `tci`: TensorCI2 object after optimization
- `f`:   black-box evaluator used to build `tci`
Returns ℓ* in 1:length(tci.localdims), i.e. a TCI *site index*.
"""
function choose_optimal_patching_site(
    tci::TensorCI2{T},
    f;
    τ::Real = 1e-10,
    rank_method::Symbol = :svd,
) where {T}

    χ = TCI.linkdims(tci)         # bond dims χ_ℓ, ℓ = 1..L-1
    L = length(tci.localdims)

    χ_max, ℓ_max = findmax(χ)     # ℓ_max = arg max bond dimension

    Pref  = reference_pivot_matrix(f, tci, ℓ_max)
    r_ref = rank_cost(Pref; τ, method = rank_method)
    r_ref > 0 || error("Reference pivot rank is zero, cannot normalize")

    S = zeros(Float64, L)

    for ℓ in 1:L
        dℓ = tci.localdims[ℓ]
        Sℓ = 0.0
        for v in 1:dℓ
            Pℓv = pivot_matrix_for_site(f, tci, ℓ, v, ℓ_max)
            rℓv = rank_cost(Pℓv; τ, method = rank_method)
            Sℓ += rℓv^2
        end
        S[ℓ] = Sℓ / (r_ref^2) 
    end

    _, ℓ_star = findmin(S)
    return ℓ_star
end
