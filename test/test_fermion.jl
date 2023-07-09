# using MKL
using .tanTRG
using JLD2
using UnPack
using ITensors
using LinearAlgebra
using ITensors.HDF5

function main()
  ncpu = 10
  ITensors.Strided.disable_threads()
  BLAS.set_num_threads(ncpu)
  ITensors.enable_threaded_blocksparse(false)

  lx = 6
  ly = 2
  N = lx * ly
  Ntot = 10

  #   f = h5open("psi0.h5", "r")
  #   rho = read(f, "rho", MPO)
  #   lgnrm = read(f, "lgnrm")
  #   close(f)
  #   sites = getsitesMPO(rho)
  sites = siteinds("tJ", N; conserve_qns=true)
  beta0 = 2^-18
  mu0 = FixtJNf_Inimu0(N::Int, Ntot::Int, beta0)
  @show mu0

  para = Dict{Symbol,Any}()
  para[:lx] = lx
  para[:ly] = ly
  para[:Ntot] = Ntot
  para[:fix_Nf] = Ntot
  para[:lstime] = vcat(-[beta0 * 2.0^ii for ii in 0:16] / 2, -[0.5, 0.75, 1, 1.5, 2] / 2)
  para[:nsweeps] = 21
  para[:maxdim] = vcat(fill(100, 4), fill(200, 4), fill(200, 4), fill(200, 4), fill(200, 5))
  para[:FixNf_begin_sw] = 15
  para[:t] = 3.0
  para[:tp] = 0.51
  para[:J] = 1.0
  para[:Jp] = 0.0289
  para[:mu0] = mu0
  para[:pbcy] = false
  para[:beta0] = beta0

  H = MPO(
    ttpJJpMPO(
      para[:pbcy],
      para[:lx],
      para[:ly],
      para[:t],
      para[:tp],
      para[:J],
      para[:Jp],
      para[:mu0],
    ),
    sites,
  )

  #   rho, lgnrm = rhoMPO(H, para[:beta0], sites)
  rho, lgnrm, mu_new = rhoMPO_FixNf(H, para[:beta0], sites, para)
  @show maxlinkdim(rho)
  flush(stdout)
  H = MPO(
    ttpJJpMPO(
      para[:pbcy], para[:lx], para[:ly], para[:t], para[:tp], para[:J], para[:Jp], mu_new
    ),
    sites,
  )
  #   f = h5open("psi0.h5", "w")
  #   write(f, "rho", rho)
  #   write(f, "sites", sites)
  #   write(f, "lgnrm", lgnrm)
  #   close(f)

  #   lsfe = zeros(para[:nsweeps], nsweeps)
  #   lsie = zeros(para[:nsweeps], nsweeps)
  #   lsbeta = zeros(nsweeps)

  solver = "exponentiate"
  @show solver
  @show para[:lstime]

  psi, rslt = tdvp(
    H,
    para[:lstime],
    rho,
    lgnrm,
    para,
    ttpJJpMPO;
    nsweeps=para[:nsweeps],
    reverse_step=true,
    normalize=false,
    maxdim=para[:maxdim],
    cutoff=1e-20,
    outputlevel=1,
    fix_Nf=para[:fix_Nf],
    solver_krylovdim=15,
    solver_backend=solver,  # or "applyexp"
  )

  @show flux(psi)

  #   lsfe[idx, :] = rslt["lsfe"]
  #   lsie[idx, :] = rslt["lsie"]
  #   lsbeta = rslt["lsbeta"]

  #   fname = "test.jld2"
  #   file = jldopen(fname, "w")
  #   @pack! file = lsfe, lsie, lsbeta, lsd
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

function FixtJNf_Inimu0(N::Int, Ntot::Int, beta0::Float64)
  # Site number N, Fermion number Ntot.
  α = log(Ntot / (2 * (N - Ntot)))
  mu0 = α / beta0
  return mu0
end

main()
# let
#   sites = siteinds("tJ", 12; conserve_sz=true)
#   H = MPO(tJchain(12), sites)
#   #   H = MPO(Heisenberg(lx * ly), sites)
#   beta0 = 2^-9

#   rho, lgnrm = rhoMPO(H, beta0, sites)
#   si = noprime.(siteinds(rho; plev=1))
#   @show maxlinkdim(rho)
#   @show si[1][1]
#   @show sites[1]
#   @assert si[1][1] == sites[1]

#   sitesnew = siteinds("tJ", 12; conserve_qns=true)
#   for ii in 1:length(H)
#     sitesnew[ii] = si[ii][1]
#     @show sitesnew[ii]
#   end
#   @show typeof(sitesnew)
#   @show typeof(sites)
#   @assert sites == sitesnew
#   str = split(string(tags(si[1][1])), ",")
#   str = chop(str[3])
#   @show str
# end