using ITensors:
  promote_itensor_eltype, _op_prod, has_fermion_string, datatype, adapt, using_auto_fermion
import ITensors: expect
"""
    thermal_corr(psi::MPO, s,
                       Op1::AbstractString,
                       Op2::AbstractString;
                       kwargs...)

    thermal_corr(psi::MPO, s,
                       Op1::Matrix{<:Number},
                       Op2::Matrix{<:Number};
                       kwargs...)

Given an MPO, the site s, and two strings denoting
operators (as recognized by the `op` function),
computes the two-point correlation function matrix
C[i,j] = Tr(ρ(β/2) Op1i Op2j ρ(β/2)) / Tr(ρ(β/2)ρ(β/2))
using efficient MPO techniques. Returns the matrix C.

# Optional Keyword Arguments

  - `site_range = 1:length(psi)`: compute correlations only
     for sites in the given range
  - `ishermitian = false` : if `false`, force independent calculations of the
     matrix elements above and below the diagonal, while if `true` assume they are complex conjugates.

Input must follow this convention:
              |
              ↑'                    ↑'
        -->---|--->--- ρ            |  operator
              ↑                     ↑
              |

Follow this convention during contraction:
              _________
              |       |
              ↓''     |
  -<'--  -<'--|--<-   |         ρ^† 
  |           ↓       ↑''
  |                   |         o
  |           ↑'——————↑'    
  ->---  ->---|-->-             ρ
              ↑
              |
where the in and out physical site indexs are also contracted.

# Notice that commonind(rho[i], H[i]) returns the two Site inds.
# However siteind(rho[i], i) returns only one site ind with in arrow.
# s must be given.!!!
# Modified from MPS version by (1) change the prime level of local op, from (Out',In) to (Out'', In')
# (2) change the prime levle of dag(psi), from dag(psi[ii]) to prime(dag(psi[ii]), plev=1).

# Examples

```julia
Cuu = thermal_corr(psi, sites, "Cdagup", "Cup"; site_range=2:8, ishermitian=true)
```
"""
function thermal_corr(
  psi::MPO, s, _Op1, _Op2; site_range=1:length(psi), ishermitian=nothing
)
  if site_range isa AbstractRange
    sites = collect(site_range)
  end

  start_site = first(sites)
  end_site = last(sites)

  N = length(psi)
  ElT = promote_itensor_eltype(psi)

  Op1 = _Op1 #make copies into which we can insert "F" string operators, and then restore.
  Op2 = _Op2
  onsiteOp = _op_prod(Op1, Op2)
  fermionic1 = has_fermion_string(Op1, s[start_site])
  fermionic2 = has_fermion_string(Op2, s[end_site])
  if fermionic1 != fermionic2
    error(
      "correlation_matrix: Mixed fermionic and bosonic operators are not supported yet."
    )
  end

  # Decide if we need to calculate a non-hermitian corr. matrix, which is roughly double the work.
  is_cm_hermitian = ishermitian
  if isnothing(is_cm_hermitian)
    # Assume correlation matrix is non-hermitian
    is_cm_hermitian = false
    O1 = op(Op1, s, start_site)  # 这里的convention是，Out' 和 In.
    O2 = op(Op2, s, start_site)
    O1 /= norm(O1)
    O2 /= norm(O2)
    #We need to decide if O1 ∝ O2 or O1 ∝ O2^dagger allowing for some round off errors.
    eps = 1e-10
    is_op_proportional = norm(O1 - O2) < eps
    is_op_hermitian = norm(O1 - dag(swapprime(O2, 0, 1))) < eps
    if is_op_proportional || is_op_hermitian
      is_cm_hermitian = true
    end
    # finally if they are both fermionic and proportional then the corr matrix will
    # be anti symmetric insterad of Hermitian. Handle things like <C_i*C_j>
    # at this point we know fermionic2=fermionic1, but we put them both in the if
    # to clarify the meaning of what we are doing.
    if is_op_proportional && fermionic1 && fermionic2
      is_cm_hermitian = false
    end
  end

  psi = copy(psi)
  ITensors.orthogonalize!(psi, start_site)
  norm2_psi = norm(psi[start_site])^2

  # Nb = size of block of correlation matrix
  Nb = length(sites)

  C = zeros(ElT, Nb, Nb)

  if start_site == 1
    L = ITensor(1.0)
  else
    lind = commonind(psi[start_site], psi[start_site - 1])
    L = delta(dag(lind), lind')
  end
  pL = start_site - 1

  for (ni, i) in enumerate(sites[1:(end - 1)])
    while pL < i - 1
      pL += 1
      # sᵢ = siteind(psi, pL)  # 返回一个site指标！！！！
      L = (L * psi[pL]) * prime(dag(psi[pL]), "Link")
    end

    Li = L * psi[i]

    # Get j == i diagonal correlations
    if i > 1
      lind = commonind(psi[i], psi[i - 1])  # 留下rind不加prime, 可以在后面直接和psi缩并掉
      oᵢ = prime(adapt(datatype(Li), op(onsiteOp, s, i)))   # add this prime for thermal_corr. Out'' 和 In'.
      C[ni, ni] = ((Li * oᵢ) * prime(prime(dag(psi[i]); plev=1), lind))[] / norm2_psi
    else  #i == 1
      @assert i == 1
      oᵢ = prime(adapt(datatype(Li), op(onsiteOp, s, i)))   # add this prime for thermal_corr. Out'' 和 In'.
      C[ni, ni] = ((Li * oᵢ) * (prime(dag(psi[i]); plev=1)))[] / norm2_psi
    end

    # Get j > i correlations
    if !using_auto_fermion() && fermionic2
      Op1 = "$Op1 * F"
    end

    oᵢ = prime(adapt(datatype(Li), op(Op1, s, i)))  # add this prime for thermal_corr. Out'' 和 In'.

    Li12 = (replaceprime(dag(psi[i])', 1, 0, "Site") * oᵢ) * Li
    pL12 = i

    for (n, j) in enumerate(sites[(ni + 1):end])
      nj = ni + n

      while pL12 < j - 1
        pL12 += 1
        if !using_auto_fermion() && fermionic2
          oᵢ = prime(adapt(datatype(psi[pL12]), op("F", s[pL12])))  # add this prime for thermal_corr. Out'' 和 In'.
          Li12 *= (oᵢ * replaceprime(dag(psi[pL12])', 1, 0, "Site"))
        else
          Li12 *= prime(dag(psi[pL12]), "Link")
        end
        Li12 *= psi[pL12]
      end

      # rind = linkind(psi, j)  # 返回右侧的 Link inds. “Link,l=j”. 如无右侧则返回nothing.
      lind = commonind(psi[j], Li12)
      Li12 *= psi[j]

      oⱼ = prime(adapt(datatype(Li12), op(Op2, s, j)))  # add this prime for thermal_corr. Out'' 和 In'.
      # sⱼ = siteind(dag(psi), j)  # 这个函数对于 rho 只返回一个 In index. 此函数很奇怪，对rho不好用
      val = (Li12 * oⱼ) * prime(prime(dag(psi[j]); plev=1), lind)  # Modify this line for thermal_corr

      C[ni, nj] = scalar(val) / norm2_psi
      if is_cm_hermitian
        C[nj, ni] = conj(C[ni, nj])
      end

      pL12 += 1
      if !using_auto_fermion() && fermionic2
        oᵢ = prime(adapt(datatype(psi[pL12]), op("F", s[pL12])))  # add this prime for thermal_corr. Out'' 和 In'.
        Li12 *= (oᵢ * replaceprime(dag(psi[pL12])', 1, 0, "Site"))
      else
        Li12 *= prime(dag(psi[pL12]), "Link")
      end
      @assert pL12 == j
    end #for j
    Op1 = _Op1 #"Restore Op1 with no Fs"

    if !is_cm_hermitian #If isHermitian=false the we must calculate the below diag elements explicitly.

      #  Get j < i correlations by swapping the operators
      if !using_auto_fermion() && fermionic1
        Op2 = "$Op2 * F"
      end
      oᵢ = prime(adapt(datatype(psi[i]), op(Op2, s, i)))  # add this prime for thermal_corr. Out'' 和 In'.
      Li21 = (Li * oᵢ) * replaceprime(dag(psi[i])', 1, 0, "Site")
      pL21 = i
      if !using_auto_fermion() && fermionic1
        Li21 = -Li21 #Required because we swapped fermionic ops, instead of sweeping right to left.
      end

      for (n, j) in enumerate(sites[(ni + 1):end])
        nj = ni + n

        while pL21 < j - 1
          pL21 += 1
          if !using_auto_fermion() && fermionic1
            oᵢ = prime(adapt(datatype(psi[pL21]), op("F", s[pL21]))) # add this prime for thermal_corr. Out'' 和 In'.
            Li21 *= oᵢ * replaceprime(dag(psi[pL21])', 1, 0, "Site")
          else
            Li21 *= prime(dag(si[pL21]), "Link")
          end
          Li21 *= prime(psi[pL21], "Site")
        end

        lind = commonind(psi[j], Li21)
        Li21 *= psi[j]

        oⱼ = prime(adapt(datatype(psi[j]), op(Op1, s, j)))  # add this prime for thermal_corr. Out'' 和 In'.
        val = (prime(prime(dag(psi[j]); plev=1), lind) * (oⱼ * Li21))[]  # Modify this line for thermal_corr
        C[nj, ni] = val / norm2_psi

        pL21 += 1
        if !using_auto_fermion() && fermionic1
          oᵢ = prime(adapt(datatype(psi[pL21]), op("F", s[pL21])))  # add this prime for thermal_corr. Out'' 和 In'.
          Li21 *= (oᵢ * replaceprime(dag(psi[pL21])', 1, 0, "Site"))
        else
          Li21 *= prime(dag(psi[pL21]), "Link")
        end
        @assert pL21 == j
      end #for j
      Op2 = _Op2 #"Restore Op2 with no Fs"
    end #if is_cm_hermitian

    pL += 1
    L = Li * prime(dag(psi[i]), "Link")
  end #for i

  # Get last diagonal element of C
  i = end_site
  while pL < i - 1
    pL += 1
    L = L * psi[pL] * prime(dag(psi[pL]), "Link")
  end
  lind = commonind(psi[i], psi[i - 1])
  oᵢ = prime(adapt(datatype(psi[i]), op(onsiteOp, s, i)))  # add this prime for thermal_corr. Out'' 和 In'.
  val = (L * (oᵢ * psi[i]) * prime(prime(dag(psi[i]); plev=1), lind))[]    # prime psi for thermal_corr.
  C[Nb, Nb] = val / norm2_psi

  return C
end

"""
    expect(psi::MPO, sites, op::AbstractString...; kwargs...)
    expect(psi::MPO, sites, op::Matrix{<:Number}...; kwargs...)
    expect(psi::MPO, sites, ops; kwargs...)

Given an MPO `psi = ρ(β/2)` and a single operator name, returns
a vector of the expected value of the operator on
each site of the MPO.

Tr(ρ(β/2) o ρ(β/2)) / Tr(ρ(β/2) ρ(β/2))

If multiple operator names are provided, returns a tuple
of expectation value vectors.

If a container of operator names is provided, returns the
same type of container with names replaced by vectors
of expectation values.

# Optional Keyword Arguments

  - `site_range = 1:length(psi)`: compute expected values only for sites in the given range

# Examples

```julia

Z = expect(psi, sites, "Sz") # compute for all sites
Z = expect(psi, sites, "Sz"; site_range=2:4) # compute for sites 2,3,4
Z3 = expect(psi, sites, "Sz"; site_range=3)  # compute for site 3 only (output will be a scalar)
XZ = expect(psi, sites, ["Sx", "Sz"]) # compute Sx and Sz for all sites
Z = expect(psi, sites, [1/2 0; 0 -1/2]) # same as expect(psi,"Sz")

updens, dndens = expect(psi, sites, "Nup", "Ndn") # pass more than one operator
```
"""
function expect(psi::MPO, s, ops; kwargs...)
  psi = copy(psi)
  N = length(psi)
  ElT = promote_itensor_eltype(psi)
  # s = siteinds(psi)

  if haskey(kwargs, :site_range)
    sites = kwargs[:site_range]
  else
    sites = 1:N
  end

  site_range = (sites isa AbstractRange) ? sites : collect(sites)
  Ns = length(site_range)
  start_site = first(site_range)

  el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

  ITensors.orthogonalize!(psi, start_site)
  norm2_psi = norm(psi)^2

  ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
  for (entry, j) in enumerate(site_range)
    ITensors.orthogonalize!(psi, j)
    for (n, opname) in enumerate(ops)
      oⱼ = prime(adapt(datatype(psi[j]), op(opname, s[j])))  # add this prime for MPO.
      val = scalar(prime(dag(psi[j]); plev=1) * oⱼ * psi[j]) / norm2_psi
      ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
    end
  end

  if sites isa Number
    return map(arr -> arr[1], ex)
  end
  return ex
end

function expect(psi::MPO, s, op::AbstractString; kwargs...)
  return first(expect(psi, s, (op,); kwargs...))
end

function expect(psi::MPO, s, op::Matrix{<:Number}; kwargs...)
  return first(expect(psi, s, (op,); kwargs...))
end

function expect(psi::MPO, s, op1::AbstractString, ops::AbstractString...; kwargs...)
  return expect(psi, s, (op1, ops...); kwargs...)
end

function expect(psi::MPO, s, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...)
  return expect(psi, s, (op1, ops...); kwargs...)
end