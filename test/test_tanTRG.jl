using ITensors
using tanTRG
using JLD2
using UnPack

function main()
  include("simpleED.jl")

  lsd = [2^i for i in 5:9]
  nd = length(lsd)

  n = 12
  s = siteinds("S=1/2", n; conserve_qns=false)
  function heisenberg(n)
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += 1.5, "Sz", j, "Sz", j + 1
    end
    return os
  end

  H = MPO(heisenberg(n), s)
  beta0 = 2^-9

  rho, lgnrm = rhoMPO(H, beta0, s)

  nsweeps = 100

  lsfe = zeros(nd, nsweeps)
  lsie = zeros(nd, nsweeps)
  lsbeta = zeros(nsweeps)
  fels_ED = zeros(nsweeps)
  iels_ED = zeros(nsweeps)
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
      solver_krylovdim=50,
    )

    println("for D = $dbond, time used is $totalTimeUsed")

    lsfe[idx, :] = rslt["lsfe"]
    lsie[idx, :] = rslt["lsie"]
    lsbeta = rslt["lsbeta"]

    if idx == 1
      println("Now for ED calculation!")
      fels_ED, iels_ED = mainED(H, s, lsbeta .^ -1)
    end
  end

  fname = "test.jld2"
  file = jldopen(fname, "w")
  @pack! file = lsfe, lsie, lsbeta, fels_ED, iels_ED, lsd
end

main()