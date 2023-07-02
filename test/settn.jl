using ITensors
using CairoMakie

(lsfe, lsbeta) = let
  function getFe(trRho, beta, lgtr0)
    return -1 * (beta)^-1 * (log(trRho) + lgtr0)
  end

  nsite = 12
  tol = 1e-12
  site = siteinds("S=½", nsite; conserve_qns=true)

  function getH(n)
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += 1.5, "Sz", j, "Sz", j + 1
    end
    return os
  end

  H = MPO(getH(nsite), site)

  # maximal n of H^n
  nmaxHn = 128
  lstrHn = Vector{Float64}(undef, nmaxHn)

  Hid = MPO(site, "Id")
  tr0 = tr(Hid)
  Hid /= tr0
  trHid = tr(Hid)

  Hn = copy(H)
  Hn /= tr0
  lstrHn[1] = tr(Hn)

  for i in 2:nmaxHn
    Hn = apply(H, Hn)
    lstrHn[i] = tr(Hn)
  end

  nls = 10
  lsbeta = [1 / 256 * 2^(i / 2) for i in 0:nls]
  lsbeta = vcat(lsbeta, [lsbeta[end] * i for i in 2:10])
  nls = length(lsbeta)

  lsfe = Vector{Float64}(undef, nls)

  for (idx, beta) in enumerate(lsbeta)
    trRho = trHid
    for i in 1:nmaxHn
      feold = getFe(trRho, beta, log(tr0))
      κi = convert(Float64, (-beta)^i / factorial(big(i))) * lstrHn[i]
      trRho += κi
      fe = getFe(trRho, beta, log(tr0))
      diff = abs((fe - feold) / feold)
      if i < 2 # at least keep the 1st order
        continue
      end
      if diff < tol
        lsfe[idx] = fe
        # @show diff
        println("for (idx, beta) = ($idx, $beta), converges at n = $i")
        @show fe
        break
      end
    end
  end
  (lsfe, lsbeta)
end

# fe --> free energy; ie --> internal energy.
include("simpleED.jl")
lsfeED, lsieED = mainED(H, s, lsbeta .^ -1)

f = Figure()
ax = Axis(
  f[1, 1];
  yscale=log10,
  xlabel="beta",
  yminorticksvisible=true,
  yminorgridvisible=true,
  yminorticks=IntervalsBetween(8),
  title="relative internal energy",
)
scatter!(ax, lsbeta, abs.((lsfe - lsfeED) ./ lsfeED))
current_figure()
