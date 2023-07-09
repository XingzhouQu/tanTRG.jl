"""
TODO:

implement two-site variational product

"""
function getFe(trRho, beta, lgtr0)
  return -1 * (beta)^-1 * (log(trRho) + lgtr0)
end

function rhoMPO_FixNf(H::MPO, beta::Number, s, para; tol=1e-12)
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

  # calculate μ_new for the next tanTRG process.
  ntot = expect(rho, s, "Ntot")
  nnCorr = thermal_corr(rho, s, "Ntot", "Ntot"; ishermitian=true)
  N² = sum(nnCorr)  # ⟨N²⟩
  Ntot = sum(ntot)
  HN = getHN(rho::MPO, H::MPO, s)
  # 这里默认SETTN之后的第一步还是走beta的步长
  μ_new =
    (0.5 * (fix_Nf - Ntot) / (para[:lstime][1] - para[:lstime][2]) + HN - Ntot * ie) /
    (N² - Ntot^2)
  println("Adjust μ to $mu_new after SETTN. (t-J sitetype here)")
  flush(stdout)

  return rho, log(trRho) + log(tr0), μ_new
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

  # for i in 2:nmaxHn
  #     Hn = apply(H, Hn)
  #     lstrHn[i] = tr(Hn)
  # end

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