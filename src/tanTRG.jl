module tanTRG

using KrylovKit
using ITensors
using Printf
using TimerOutputs
using Observers
using MAT

using ITensors:
  AbstractMPS,
  @debug_check,
  @timeit_debug,
  check_hascommoninds,
  orthocenter,
  ProjMPS,
  set_nsite!,
  # added by phyjswang
  leftlim,
  setleftlim!,
  rightlim,
  setrightlim!

# using ITensorTDVP:
#     _tdvp_compute_nsweeps
#     process_sweeps
#     TDVPOrder
#     tdvp_step

# Compatibility of ITensor observer and Observers
include("update_observer.jl")

# Utilities for making it easier
# to define solvers (like ODE solvers)
# for TDVP
include("solver_utils.jl")

include("applyexp.jl")
include("tdvporder.jl")
include("tdvpinfo.jl")
include("tdvp_step.jl")
include("tdvp_generic.jl")
include("tdvp.jl")
include("dmrg.jl")
include("dmrg_x.jl")
include("projmpo_apply.jl")
include("contract_mpo_mps.jl")
include("projmps2.jl")
include("projmpo_mps2.jl")
include("linsolve.jl")
include("rhoMPO.jl")
# added by phyjswang
include("ITensors_additional.jl")
# include("ITensorTDVP_additional.jl")

export tdvp, dmrg_x, to_vec, TimeDependentSum, linsolve, rhoMPO

end # module tanTRG
