using ITensors
using ITensors.HDF5
include("./src/thermal_corr.jl")

function main()
  for sw in 20:25
    mpsname = "Sweep$sw-MPO.h5"
    f = h5open(mpsname, "r")
    psi = read(f, "psi", MPO)
    close(f)
    sites = getsitesMPO(psi)
    ntot = expect(psi, sites, "Ntot")
    Ntot = sum(ntot)
    println("Sweep$sw, Nnow = $Ntot")
  end
end

function getsitesMPO(rho::MPO)
  # Given an MPO, obtain the sites object when it is constructed.
  N = length(rho)
  si = noprime.(siteinds(rho; plev=1))
  str = split(string(tags(si[1][1])), ",")
  str = chop(str[3])
  sites = siteinds(string(str), N; conserve_qns=hasqns(rho))
  for ii in 1:length(rho)
    sites[ii] = si[ii][1]
  end
  return sites
end

main()