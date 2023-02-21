using ITensors
using KrylovKit
using LinearAlgebra
using MKL

include("fuse_inds.jl")

ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()

function mainED(H, s, tls; blas_num_threads=Sys.CPU_THREADS, fuse=true, binary=true)
  nsite = size(H)[1]
  if nsite > 16
    @warn "System size of $nsite is likely too large for exact diagonalization."
  end

  BLAS.set_num_threads(blas_num_threads)

  if fuse
    if binary
      println("Fuse the indices using a binary tree")
      T = fusion_tree_binary(s)
      H_full = @time fuse_inds_binary(H, T)
    else
      println("Fuse the indices using an unbalances tree")
      T = fusion_tree(s)
      H_full = @time fuse_inds(H, T)
    end
  else
    println("Don't fuse the indices")
    @disable_warn_order begin
      H_full = @time contract(H)
    end
  end

  vals = eigvals(array(H_full))
  
  betals = tls.^-1
  fels = Vector{Float64}(undef, length(betals))
  iels = Vector{Float64}(undef, length(betals))
  for (cnt, β) in enumerate(betals)
    bigz = sum(exp.(-β*vals))
    fels[cnt] = -tls[cnt] * log(bigz)
    iels[cnt] = sum(vals .* exp.(-β*vals))/bigz
  end
  return fels, iels
end