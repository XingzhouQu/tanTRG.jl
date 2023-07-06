using MKL
using .tanTRG
using JLD2
using UnPack
using ITensors
using LinearAlgebra

function main()
  ncpu = 10
  ITensors.Strided.disable_threads()
  BLAS.set_num_threads(ncpu)
  ITensors.enable_threaded_blocksparse(false)

  lsd = [500]
  nd = length(lsd)

  lx = 6
  ly = 2
  N = lx * ly
  sites = siteinds("tJ", N; conserve_qns=true)
  t = 3.0
  tp = 0.51
  J = 1.0
  Jp = 0.0289
  mu = 5.0

  #   H = MPO(ttpJJpMPO(false, lx, ly, t, tp, J, Jp, mu), sites)
  H = MPO(tJchain(lx * ly), sites)
  #   H = MPO(Heisenberg(lx * ly), sites)
  beta0 = 2^-9

  rho, lgnrm = rhoMPO(H, beta0, sites)
  @show maxlinkdim(rho)
  flush(stdout)

  nsweeps = 10

  lsfe = zeros(nd, nsweeps)

  lsie = zeros(nd, nsweeps)
  lsbeta = zeros(nsweeps)

  solver = "exponentiate"
  @show solver

  for (idx, dbond) in enumerate(lsd)
    totalTimeUsed, rslt = tdvp(
      H,
      -0.3,
      rho,
      lgnrm;
      nsweeps=nsweeps,
      reverse_step=true,
      normalize=false,
      maxdim=dbond,
      cutoff=1e-12,
      outputlevel=1,
      time_start=beta0,
      solver_krylovdim=20,
      solver_backend=solver,  # or "applyexp"
    )

    println("for D = $dbond, time used is $totalTimeUsed")
    flush(stdout)

    lsfe[idx, :] = rslt["lsfe"]
    lsie[idx, :] = rslt["lsie"]
    lsbeta = rslt["lsbeta"]
  end

  fname = "test.jld2"
  file = jldopen(fname, "w")
  @pack! file = lsfe, lsie, lsbeta, lsd
end

function ttpJJpMPO(
  pbcy::Bool,
  Nx::Int,
  Ny::Int,
  t::Float64,
  tp::Float64,
  J::Float64,
  Jp::Float64,
  mu::Float64,
)
  # Input operator terms which define a Hamiltonian
  os = OpSum()
  for j in 1:Nx
    for k in 1:Ny
      if k == Ny  # vertical terms
        if pbcy && (Ny > 2)
          os += -t, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * (j - 1) + 1
          os += -t, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * (j - 1) + 1
          os += -t, "Cdagup", Ny * (j - 1) + 1, "Cup", Ny * (j - 1) + k
          os += -t, "Cdagdn", Ny * (j - 1) + 1, "Cdn", Ny * (j - 1) + k
          os += J / 2, "S+", Ny * (j - 1) + k, "S-", Ny * (j - 1) + 1
          os += J / 2, "S-", Ny * (j - 1) + k, "S+", Ny * (j - 1) + 1
          os += J, "Sz", Ny * (j - 1) + k, "Sz", Ny * (j - 1) + 1
          os += -J / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * (j - 1) + 1
        end
      else
        os += -t, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * (j - 1) + 1 + k
        os += -t, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * (j - 1) + 1 + k
        os += -t, "Cdagup", Ny * (j - 1) + 1 + k, "Cup", Ny * (j - 1) + k
        os += -t, "Cdagdn", Ny * (j - 1) + 1 + k, "Cdn", Ny * (j - 1) + k
        os += J / 2, "S+", Ny * (j - 1) + k, "S-", Ny * (j - 1) + 1 + k
        os += J / 2, "S-", Ny * (j - 1) + k, "S+", Ny * (j - 1) + 1 + k
        os += J, "Sz", Ny * (j - 1) + k, "Sz", Ny * (j - 1) + 1 + k
        os += -J / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * (j - 1) + 1 + k
      end

      if j < Nx  # horiz terms
        os += -t, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * j + k
        os += -t, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * j + k
        os += -t, "Cdagup", Ny * j + k, "Cup", Ny * (j - 1) + k
        os += -t, "Cdagdn", Ny * j + k, "Cdn", Ny * (j - 1) + k
        os += J / 2, "S+", Ny * (j - 1) + k, "S-", Ny * j + k
        os += J / 2, "S-", Ny * (j - 1) + k, "S+", Ny * j + k
        os += J, "Sz", Ny * (j - 1) + k, "Sz", Ny * j + k
        os += -J / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * j + k
        if k == 1  # oblique terms
          if pbcy && (Ny > 2)  # row 1  \  to the rightbottom
            os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * (j + 1)
            os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * (j + 1)
            os += -tp, "Cdagup", Ny * (j + 1), "Cup", Ny * (j - 1) + k
            os += -tp, "Cdagdn", Ny * (j + 1), "Cdn", Ny * (j - 1) + k
            os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * (j + 1)
            os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * (j + 1)
            os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * (j + 1)
            os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * (j + 1)
          end
          # row 1  /  to the topright
          os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * j + k + 1
          os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * j + k + 1
          os += -tp, "Cdagup", Ny * j + k + 1, "Cup", Ny * (j - 1) + k
          os += -tp, "Cdagdn", Ny * j + k + 1, "Cdn", Ny * (j - 1) + k
          os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * j + k + 1
          os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * j + k + 1
          os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * j + k + 1
          os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * j + k + 1
        elseif k == Ny
          if pbcy && (Ny > 2)  # site at row Ny (top row), to the bottom line at topright
            os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * (j - 1) + k + 1
            os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * (j - 1) + k + 1
            os += -tp, "Cdagup", Ny * (j - 1) + k + 1, "Cup", Ny * (j - 1) + k
            os += -tp, "Cdagdn", Ny * (j - 1) + k + 1, "Cdn", Ny * (j - 1) + k
            os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * (j - 1) + k + 1
            os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * (j - 1) + k + 1
            os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * (j - 1) + k + 1
            os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * (j - 1) + k + 1
          end
          # site at row Ny (top row), to the Ny-1 line at bottomright
          os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * j + k - 1
          os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * j + k - 1
          os += -tp, "Cdagup", Ny * j + k - 1, "Cup", Ny * (j - 1) + k
          os += -tp, "Cdagdn", Ny * j + k - 1, "Cdn", Ny * (j - 1) + k
          os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * j + k - 1
          os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * j + k - 1
          os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * j + k - 1
          os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * j + k - 1
        else
          os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * j + k + 1
          os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * j + k + 1
          os += -tp, "Cdagup", Ny * j + k + 1, "Cup", Ny * (j - 1) + k
          os += -tp, "Cdagdn", Ny * j + k + 1, "Cdn", Ny * (j - 1) + k
          os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * j + k + 1
          os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * j + k + 1
          os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * j + k + 1
          os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * j + k + 1

          os += -tp, "Cdagup", Ny * (j - 1) + k, "Cup", Ny * j + k - 1
          os += -tp, "Cdagdn", Ny * (j - 1) + k, "Cdn", Ny * j + k - 1
          os += -tp, "Cdagup", Ny * j + k - 1, "Cup", Ny * (j - 1) + k
          os += -tp, "Cdagdn", Ny * j + k - 1, "Cdn", Ny * (j - 1) + k
          os += Jp / 2, "S+", Ny * (j - 1) + k, "S-", Ny * j + k - 1
          os += Jp / 2, "S-", Ny * (j - 1) + k, "S+", Ny * j + k - 1
          os += Jp, "Sz", Ny * (j - 1) + k, "Sz", Ny * j + k - 1
          os += -Jp / 4, "Ntot", Ny * (j - 1) + k, "Ntot", Ny * j + k - 1
        end
      end
    end
  end
  for si in 1:(Nx * Ny)
    os += -mu, "Ntot", si
  end
  return os
end

function Heisenberg(N)
  os = OpSum()
  for ii in 1:(N - 1)
    os += 0.5, "S+", ii, "S-", ii + 1
    os += 0.5, "S-", ii, "S+", ii + 1
    os += 1, "Sz", ii, "Sz", ii + 1
  end
  return os
end

function tJchain(N)
  os = OpSum()
  for ii in 1:(N - 1)
    os += 1, "Cdagup", ii, "Cup", ii + 1
    os += 1, "Cdagdn", ii, "Cdn", ii + 1
    os += 1, "Cdagup", ii + 1, "Cup", ii
    os += 1, "Cdagdn", ii + 1, "Cdn", ii
    os += 0.25, "S+", ii, "S-", ii + 1
    os += 0.25, "S-", ii, "S+", ii + 1
    os += 0.5, "Sz", ii, "Sz", ii + 1
    os += -0.5 / 4, "Ntot", ii, "Ntot", ii + 1
  end
  return os
end

# function HaldaneChain(N)
#     os = OpSum()
#     for ii in 1:(N -1)
#         os += 1, 
#     end
#     return os
# end

# main()
let
  sites = siteinds("tJ", 12; conserve_qns=true)
  H = MPO(tJchain(12), sites)
  #   H = MPO(Heisenberg(lx * ly), sites)
  beta0 = 2^-9

  rho, lgnrm = rhoMPO(H, beta0, sites)
  @show maxlinkdim(rho)
  @show inds(rho[12])
  @show inds(dag(rho[12]))
  si = siteind(rho, 12)
  sj = siteind(dag(rho), 12)
  @show si
  @show sj
end