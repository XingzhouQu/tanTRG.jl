"""
TODO:

implement two-site variational product

"""
function getFe(trRho, beta, lgtr0)
  return -1 * (beta)^-1 * (log(trRho) + lgtr0)
end

function rhoMPO(H::MPO, beta::Number, s; tol=1e-12)
  nmaxHn = 12
  lstrHn = Vector{Float64}(undef, nmaxHn)

  Hid = MPO(s, "Id")
  tr0 = tr(Hid)
  Hid /= tr0
  trHid = tr(Hid)

  Hn = copy(H)
  Hn /= tr0
  lstrHn[1] = tr(Hn)

  rho = Hid
  trRho = trHid

  feold = getFe(trRho, beta, log(tr0))
  fe = 0.0
  for i in 1:nmaxHn
    if i > 1
      Hn = apply(H, Hn)
    end
    rho += convert(Float64, (-beta)^i / factorial(big(i))) * Hn
    trRho += convert(Float64, (-beta)^i / factorial(big(i))) * tr(Hn)
    fe = getFe(trRho, beta, log(tr0))
    if i < 2 # at least keep the 1st order
      continue
    end
    if abs((fe - feold) / feold) < tol
      println("SETTN converges at i = $i")
      break
    end
    feold = fe
  end
  flush(stdout)
  #   swapprime!(rho, 2, 1)
  return rho, log(trRho) + log(tr0)
end

# Some useful functions for Fixing number fermion.

function exp_alphaN(α::Float64, H::MPO, sites, cal_NH::Bool)
  # For tJ sitetype only!!!
  # calculate Tr(exp(αN) H) -- set `cal_NH=false`
  # or Tr(exp(αN) NH)  -- set `cal_NH=true`
  if cal_NH
    os = OpSum()
    for ii in 1:length(H)
      os += "Ntot", ii
    end
    Nmpo = MPO(os, sites)
    H = apply(H, Nmpo)
  else
    nothing
  end
  expαN_mpo = MPO(length(H))
  for ii in 1:length(H)
    expαN_mpo[ii] = op([1 0 0; 0 exp(α / 2) 0; 0 0 exp(α / 2)], sites[ii])
  end
  return inner(expαN_mpo, apply(H, expαN_mpo))
end

function getHN(psi::MPO, H::MPO, sites)
  # calculate ⟨HN⟩ for thermal density matrix psi.
  os = OpSum()
  for ii in 1:length(psi)
    os += "Ntot", ii
  end
  Nmpo = MPO(os, sites)
  HN = inner(psi, apply(apply(H, Nmpo), psi)) / norm(psi)^2
  return HN
end

function pull_back_Nf(
  rho::MPO,
  μ₀::Float64,
  N²::Float64,
  Ntot::Float64,
  Nf_fix::Int,
  βhalf::Float64,
  sites;
  iter=5,
  Nf_tol=1e-3,
)
  # 用exp(αN)把粒子数严格拉回到 Nf_fix. 使用 bilayer, 输入的温度 βhalf 是 β/2.
  # ∂⟨N⟩/∂μ = β(⟨N²⟩-⟨N⟩²), 牛顿法迭代
  for jj in 1:iter
    # 计算新的μ
    μ = μ₀ - (Ntot - Nf_fix) / (βhalf * (N² - Ntot^2))
    # 计算新的ρ
    α = βhalf * (μ - μ₀)
    expαN_mpo = MPO(length(rho))
    for ii in 1:length(rho)
      expαN_mpo[ii] = op([1 0 0; 0 exp(α) 0; 0 0 exp(α)], sites[ii])
    end
    rho = apply(expαN_mpo, rho)
    rho = rho / norm(rho)^2
    # 计算观测量，用于检查收敛和下一次迭代
    ntot = expect(rho, sites, "Ntot")
    nnCorr = thermal_corr(rho, sites, "Ntot", "Ntot"; ishermitian=true)
    N² = sum(nnCorr)  # ⟨N²⟩
    Ntot = sum(ntot)
    # 这一步的μ作为下一步的初始值
    μ₀ = μ
    # 检查收敛
    if abs(Ntot - Nf_fix) < Nf_tol
      println("pull_back converges at iter=$iter with Ntot $Ntot")
      break
    end
    if jj == iter
      println(
        "pull_back finishs after $iter iterations with Ntot $Ntot. (May not converge)"
      )
    end
  end
  return rho, μ₀
end

function updateOps(OpS0, μ::Float64, Nsites::Int)
  # 从给定的不含化学势的OpS0出发，
  Ops = copy(OpS0)
  for si in 1:Nsites
    Ops += -mu, "Ntot", si
  end
  return Ops
end