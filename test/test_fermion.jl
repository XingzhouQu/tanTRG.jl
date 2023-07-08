using MKL
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

  f = h5open("psi0.h5", "r")
  rho = read(f, "rho", MPO)
  lgnrm = read(f, "lgnrm")
  close(f)
  sites = getsitesMPO(rho)

  para = Dict{Symbol,Any}()
  para[:lx] = lx
  para[:ly] = ly
  para[:Ntot] = 10
  para[:fix_Nf] = 10
  para[:lstime] = -[0.0, 0.5, 1.0, 1.5, 1.9, 2.3, 2.6, 2.9, 3.1, 3.3, 3.4]
  para[:nsweeps] = 10
  para[:maxdim] = [100, 100, 100, 200, 300, 500, 500, 500, 500, 500]
  para[:t] = 3.0
  para[:tp] = 0.51
  para[:J] = 1.0
  para[:Jp] = 0.0289
  para[:mu0] = 5.0
  para[:pbcy] = false
  para[:beta0] = 2^-9

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
  #   H = MPO(tJchain(lx * ly), sites)

  #   rho, lgnrm = rhoMPO(H, para[:beta0], sites)
  #   @show maxlinkdim(rho)
  #   flush(stdout)
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
    cutoff=1e-12,
    outputlevel=1,
    fix_Nf=para[:fix_Nf],
    solver_krylovdim=15,
    solver_backend=solver,  # or "applyexp"
  )

  @show flux(psi)
  return flush(stdout)

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

# function HaldaneChain(N)
#     os = OpSum()
#     for ii in 1:(N -1)
#         os += 1, 
#     end
#     return os
# end

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