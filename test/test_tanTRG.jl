using ITensors
using tanTRG
using ITensors.HDF5
using MAT
using CairoMakie

let
n = 14
@show n
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
# ψ = randomMPS(s, "↑"; linkdims=10)
# ψ = randomMPS(s, "↑")
# H(ψ)
beta0 = 1/256

rho, lgnrm = rhoMPO(H,beta0,s)

@show lgnrm

totalTimeUsed, rslt = tdvp(
  H,
  -.1,
  rho,
  lgnrm;
  nsweeps=100,
  reverse_step=true,
  normalize=false,
  maxdim=256,
  cutoff=1e-12,
  outputlevel=1,
  time_start=beta0,
  solver_krylovdim=50,
)

@show totalTimeUsed

# fid = ITensors.HDF5.h5open("test.h5","r")
# lsbeta = read(fid,"lsbeta")
# lsfe = read(fid,"lsfe")
# lsie = read(fid,"lsie")
# close(fid)

lsbeta = rslt["lsbeta"]
lsfe = rslt["lsfe"]
lsie = rslt["lsie"]

include("simpleED.jl")
fels_ED, iels_ED = mainED(H, s, lsbeta.^-1)

f = Figure()
ax = Axis(f[1, 1], yscale = log10,
xlabel = "beta"
        # yminorticksvisible = true, yminorgridvisible = true,
        # yminorticks = IntervalsBetween(8)
        )
# scatter(f[1,1], lsbeta, lsie, color=:red,title="Internal Energy")
scatter!(ax, lsbeta, abs.((lsie-iels_ED)./iels_ED), color=:blue, label = "relative internal energy")
scatter!(ax, lsbeta, (lsfe-fels_ED)./fels_ED, color=:red, label = "relative free energy")
# lines!(f[1,2], lsbeta, fels_ED, color=:blue)
# @show (lsie-iels_ED)./iels_ED
axislegend()
current_figure()
end