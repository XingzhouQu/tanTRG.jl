using ITensors
using tanTRG
using CairoMakie

f = Figure()
ax1 = Axis(f[1, 1], yscale = log10,
xlabel = "beta", yminorticksvisible = true, yminorgridvisible = true,
        yminorticks = IntervalsBetween(8),
        title = "relative internal energy"
        )
ax2 = Axis(f[1, 2], yscale = log10,
xlabel = "beta", yminorticksvisible = true, yminorgridvisible = true,
        yminorticks = IntervalsBetween(8),
        title = "relative free energy"
        )

lsd = [2^i for i in 5:9]

(lsbeta, lsfe, lsie, fels_ED, iels_ED) = let
  n = 12
  s = siteinds("S=1/2", n, conserve_qns=true)
  function heisenberg(n)
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += 1.5,"Sz", j, "Sz", j + 1
    end
    return os
  end

  H = MPO(heisenberg(n), s)
  beta0 = 1/256

  rho, lgnrm = rhoMPO(H,beta0,s)

  lsfe = Dict()
  lsie = Dict()
  lsbeta = Dict()
  for (idx, dbond) in enumerate(lsd)
    totalTimeUsed, rslt = tdvp(
      H,
      -.3,
      rho,
      lgnrm;
      nsweeps=100,
      reverse_step=true,
      normalize=false,
      maxdim=dbond,
      cutoff=1e-12,
      outputlevel=1,
      time_start=beta0,
      solver_krylovdim=50,
    )

    println("for D = $dbond, time used is $totalTimeUsed")

    lsfe[idx] = rslt["lsfe"]
    lsie[idx] = rslt["lsie"]
    lsbeta[idx] = rslt["lsbeta"]

    if idx == 1
      include("simpleED.jl")
      fels_ED, iels_ED = mainED(H, s, lsbeta[1].^-1)
    end

    scatter!(ax1, lsbeta[idx], abs.((lsie[idx]-iels_ED)./iels_ED), label = "D = $dbond")
    scatter!(ax2, lsbeta[idx], (lsfe[idx]-fels_ED)./fels_ED, label = "D = $dbond")
    axislegend(ax1)
    axislegend(ax2)
    current_figure()
  end
  (lsbeta, lsfe, lsie, fels_ED, iels_ED)
end
