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
  Ntot = 9

  #   f = h5open("psi0.h5", "r")
  #   rho = read(f, "rho", MPO)
  #   lgnrm = read(f, "lgnrm")
  #   close(f)
  #   sites = getsitesMPO(rho)
  sites = siteinds("tJ", N; conserve_qns=true)
  beta0 = 2^-17

  #   H = MPO(
  #     ttpJJpMPO(
  #       para[:pbcy],
  #       para[:lx],
  #       para[:ly],
  #       para[:t],
  #       para[:tp],
  #       para[:J],
  #       para[:Jp],
  #       para[:mu0],
  #     ),
  #     sites,
  #   )
  H0 = MPO(tJchain(N, 0), sites)  # 这里先不带μ, 用H0计算μ₀
  alpha, mu0 = FixtJNf_Inimu0(N::Int, Ntot::Int, H0, sites)
  @show alpha, mu0

  # 注意这里不能用SETTN展开 H₀-α/β₀N, 否则大体系定不准
  # 应当先直接展开 H₀, 然后乘以 exp(αN)把粒子数拉回去
  rho, lgnrm = rhoMPO(H0, beta0 / 2, sites)  # 注意这里计算都是bilayer, 初始温度为β₀, SETTN展到 β₀/2 
  expαN_mpo = MPO(length(H0))  # bilayer, exp(αN) 的矩阵元也对应α/2
  for ii in 1:length(H0)
    expαN_mpo[ii] = op([1 0 0; 0 exp(alpha / 2) 0; 0 0 exp(alpha / 2)], sites[ii])
  end
  rho = apply(rho, expαN_mpo)

  @show maxlinkdim(rho)
  flush(stdout)

  para = Dict{Symbol,Any}()
  para[:lx] = lx
  para[:ly] = ly
  para[:Ntot] = Ntot
  para[:fix_Nf] = Ntot
  para[:lstime] = vcat(
    -[beta0 * 2.0^ii for ii in 0:14] / 2,
    -[0.07, 0.08, 0.09, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2] / 2,
  )
  para[:nsweeps] = 25
  para[:maxdim] = 200
  para[:FixNf_begin_sw] = 12
  para[:t] = 3.0
  para[:tp] = 0.51
  para[:J] = 1.0
  para[:Jp] = 0.0289
  para[:mu0] = mu0
  para[:pbcy] = false
  para[:beta0] = beta0

  solver = "exponentiate"
  @show solver
  #   OpS0 = tJchain(N, mu0)
  OpS0 = tJchain(N, 0.0)
  H = MPO(tJchain(N, mu0), sites)

  psi, rslt = tdvp(
    H,
    para[:lstime],
    rho,
    lgnrm,
    para,
    OpS0;
    nsweeps=para[:nsweeps],
    reverse_step=true,
    normalize=false,
    maxdim=para[:maxdim],
    cutoff=1e-15,
    outputlevel=1,
    fix_Nf=para[:fix_Nf],
    solver_krylovdim=15,
    solver_backend=solver,  # or "applyexp"
  )

  @show rslt
  psi, μ = pull_back_Nf(
    psi, rslt[:mu], rslt[:Nsq], rslt[:Ntot], para[:fix_Nf], para[:lstime][end], sites
  )
  @show sum(expect(psi, sites, "Ntot"))
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

function tJchain(N, mu)
  os = OpSum()
  for ii in 1:(N - 1)
    os += -1, "Cdagup", ii, "Cup", ii + 1
    os += -1, "Cdagdn", ii, "Cdn", ii + 1
    os += -1, "Cdagup", ii + 1, "Cup", ii
    os += -1, "Cdagdn", ii + 1, "Cdn", ii
    os += 0.25, "S+", ii, "S-", ii + 1
    os += 0.25, "S-", ii, "S+", ii + 1
    os += 0.5, "Sz", ii, "Sz", ii + 1
    os += -0.5 / 4, "Ntot", ii, "Ntot", ii + 1
  end
  for jj in 1:N
    os += -mu, "Ntot", jj
    os += -0.5, "Sz", jj
  end
  return os
end

function FixtJNf_Inimu0(N::Int, Ntot::Int, H::MPO, sites)
  # Site number N, Fermion number Ntot.
  α = log(Ntot / (2 * (N - Ntot)))
  fenzi =  # 最后一个输入决定是计算 NH(true) 还是 H(false)
    exp_alphaN(α, H, sites, true) * (1 + 2 * exp(α)) -
    2N * exp(α) * exp_alphaN(α, H, sites, false)
  fenmu = (1 + 2 * exp(α))^(N - 1) * 2 * N * exp(α)
  mu0 = fenzi / fenmu
  return α, mu0
end

main()