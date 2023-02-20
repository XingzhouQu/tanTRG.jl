using ITensors
using tanTRG
# using ITensors.HDF5
using MAT

n = 6
@show n
s = siteinds("S=1/2", n, conserve_qns=true)

function heisenberg(n)
  os = OpSum()
  for j in 1:(n - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

H = MPO(heisenberg(n), s)
# ψ = randomMPS(s, "↑"; linkdims=10)
ψ = randomMPS(s, "↑")
# H(ψ)
beta0 = 1/8

rho, lgnrm = rhoMPO(H,beta0,s)

@show lgnrm

totalTimeUsed = tdvp(
  H,
  -1.,
  rho,
  lgnrm;
  nsweeps=20,
  reverse_step=true,
  normalize=false,
  maxdim=64,
  cutoff=1e-10,
  outputlevel=1,
  time_start=beta0
  # solver_krylovdim=50,
)

@show totalTimeUsed