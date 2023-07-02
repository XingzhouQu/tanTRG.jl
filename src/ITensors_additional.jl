# ---------------------------------------------------------------------------------------- #
#=
    modifications for `src/mps/abstractprojmpo.jl` [ITensors v0.3.34]
=#

function ITensors.prime(::ITensors.OneITensor, ::String) end

function ITensors.:*(it::ITensor, ::Nothing)
  return it
end

"""
    product(P::ProjMPO,v::ITensor)::ITensor

    (P::ProjMPO)(v::ITensor)

Efficiently multiply the ProjMPO `P`
by an ITensor `v` in the sense that the
ProjMPO is a generalized square matrix
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.

If v comes from an MPO like density matrix,
one firstly prime site indices of MPO in P,
after Pv, replace link 1st level to 0th level
then site indices 2nd level to 1st level
"""
function ITensors.product(P::AbstractProjMPO, v::ITensor)::ITensor
  Pv = contract(P, v)
  if order(Pv) != order(v)
    error(
      string(
        "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
        "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
        "(1) You are trying to multiply the ProjMPO with the $(nsite(P))-site wave-function at the wrong position.\n",
        "(2) `orthogonalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
        "P*v inds: $(inds(Pv)) \n\n",
        "v inds: $(inds(v))",
      ),
    )
  end
  # return noprime(Pv)
  return replaceprime(Pv, 2, 1, "Site")
end

function ITensors._makeL!(P::AbstractProjMPO, psi::MPO, k::Int)::Union{ITensor,Nothing}
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  ll = P.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    P.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(P)
  while ll < k
    # Prime level of P.H is already handled. See Func tdvp_sweep in file tdvp_setp.jl. Edited by XZ.Q
    L = L * psi[ll + 1] * P.H[ll + 1] * dag(replaceprime(prime(psi[ll + 1]), 1, 0, "Site"))
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return L
end

function ITensors.makeL!(P::AbstractProjMPO, psi::MPO, k::Int)
  ITensors._makeL!(P, psi, k)
  return P
end

function ITensors._makeR!(P::AbstractProjMPO, psi::MPO, k::Int)::Union{ITensor,Nothing}
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  rl = P.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    P.rpos = k
    return nothing
  end
  N = length(P.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    # Prime level of P.H is already handled. See Func tdvp_sweep in file tdvp_setp.jl. Edited by XZ.Q
    R = R * psi[rl - 1] * P.H[rl - 1] * dag(replaceprime(prime(psi[rl - 1]), 1, 0, "Site"))
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return R
end

function ITensors.makeR!(P::AbstractProjMPO, psi::MPO, k::Int)
  ITensors._makeR!(P, psi, k)
  return P
end

"""
    position!(P::ProjMPO, psi, pos::Int)

Given an MPO `psi` (with additional doubly-primed legs), shift the projection of the
MPO represented by the ProjMPO `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPO on sites that have already been projected.
The MPO `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function ITensors.position!(P::AbstractProjMPO, psi::MPO, pos::Int)
  makeL!(P, psi, pos - 1)
  makeR!(P, psi, pos + nsite(P))
  return P
end

# ---------------------------------------------------------------------------------------- #
#=
    modifications for `src/mps/mps.jl` [ITensors v0.3.34]
=#
"""
    replacebond!(M::MPO, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b` and `b+1` of MPO `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""
function ITensors.replacebond!(M::MPO, b::Int, phi::ITensor; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  swapsites::Bool = get(kwargs, :swapsites, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error(
      """dir keyword in replacebond! has been replaced by ortho.
      Note that the options are now the same as factorize, so use `left` instead of `fromleft` and `right` instead of `fromright`.""",
    )
  end

  indsMb = inds(M[b])
  if swapsites
    sb = siteind(M, b)
    noprime!(sb)
    sbp1 = siteind(M, b + 1)
    noprime!(sbp1)
    indsMb = replaceind(indsMb, sb, sbp1)
    indsMb = replaceind(indsMb, sb'', sbp1'')
  end

  L, R, spec = factorize(
    phi, indsMb; which_decomp=which_decomp, tags=tags(linkind(M, b)), kwargs...
  )

  M[b] = L
  M[b + 1] = R
  if ortho == "left"
    leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
    rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
    normalize && (M[b + 1] ./= norm(M[b + 1]))
  elseif ortho == "right"
    leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
    rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
    normalize && (M[b] ./= norm(M[b]))
  else
    error(
      "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
    )
  end
  return spec
end

"""
    replacebond(M::MPO, b::Int, phi::ITensor; kwargs...)

Like `replacebond!`, but returns the new MPO.
"""
function ITensors.replacebond(M0::MPO, b::Int, phi::ITensor; kwargs...)
  M = copy(M0)
  replacebond!(M, b, phi; kwargs...)
  return M
end

# Allows overloading `replacebond!` based on the projected
# MPO type. By default just calls `replacebond!` on the MPO.
function ITensors.replacebond!(PH, M::MPO, b::Int, phi::ITensor; kwargs...)
  return replacebond!(M, b, phi; kwargs...)
end
