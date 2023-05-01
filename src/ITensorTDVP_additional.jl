# modifications for `src/tdvp.jl`
function ITensorTDVP.tdvp(H, t::Number, psi0::MPO; kwargs...)
    return tdvp(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end

# modifications for `src/tdvp_generic.jl` [ITensorTDVP v0.1.3]
function ITensorTDVP.tdvp(solver, PH, t::Number, psi0::MPO, lgnrm; kwargs...)
    reverse_step = get(kwargs, :reverse_step, true)

    nsweeps = _tdvp_compute_nsweeps(t; kwargs...)
    maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, kwargs...)

    time_start::Number = get(kwargs, :time_start, 0.0)
    time_step::Number = get(kwargs, :time_step, t)
    order = get(kwargs, :order, 2)
    tdvp_order = TDVPOrder(order, Base.Forward)

    checkdone = get(kwargs, :checkdone, nothing)
    write_when_maxdim_exceeds::Union{Int,Nothing} = get(
      kwargs, :write_when_maxdim_exceeds, nothing
    )
    observer = get(kwargs, :observer!, NoObserver())
    step_observer = get(kwargs, :step_observer!, NoObserver())
    outputlevel::Int = get(kwargs, :outputlevel, 0)

    psi = copy(psi0)

    # Keep track of the start of the current time step.
    # Helpful for tracking the total time, for example
    # when using time-dependent solvers.
    # This will be passed as a keyword argument to the
    # `solver`.
    current_time = time_start
    info = nothing
    totalTimeUsed = 0.0

    # initialize data to be saved

    rslt = Dict("lsfe" => Vector{Float64}(undef, nsweeps), "lsie" => Vector{Float64}(undef, nsweeps), "lsbeta" => Vector{Float64}(undef, nsweeps))

    for sw in 1:nsweeps
      if !isnothing(write_when_maxdim_exceeds) && maxdim[sw] > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
          )
        end
        PH = disk(PH)
      end

      sw_time = @elapsed begin
        psi, PH, lgnrm, info = tdvp_step(
          tdvp_order,
          solver,
          PH,
          time_step,
          psi,
          lgnrm;
          kwargs...,
          current_time,
          reverse_step,
          sweep=sw,
          maxdim=maxdim[sw],
          mindim=mindim[sw],
          cutoff=cutoff[sw],
          noise=noise[sw],
        )
      end

      current_time += time_step
      totalTimeUsed += sw_time
      rslt["lsbeta"][sw] = -current_time*2 # bilayer, negative evolution step
      rslt["lsfe"][sw] = -1 * (rslt["lsbeta"][sw])^-1 * 2*lgnrm # √[tr(ρ†ρ)]
      rslt["lsie"][sw] = inner(psi, swapprime(contract(PH.H, psi), 1, 0))/norm(psi)^2
      # ITensors.HDF5.h5open("test.h5","w") do fid
      #   fid["lsbeta"] = lsbeta
      #   fid["lsfe"] = lsfe
      #   fid["lsie"] = lsie
      # end


      update!(step_observer; psi, sweep=sw, outputlevel, current_time)

      if outputlevel >= 1
        print("After sweep ", sw, ":")
        print(" maxlinkdim=", maxlinkdim(psi))
        @printf(" maxerr=%.2E", info.maxtruncerr)
        print(" current_time=", round(current_time; digits=3))
        print(" time=", round(sw_time; digits=3))
        println()
        flush(stdout)
      end

      isdone = false
      if !isnothing(checkdone)
        isdone = checkdone(; psi, sweep=sw, outputlevel, kwargs...)
      elseif observer isa ITensors.AbstractObserver
        isdone = checkdone!(observer; psi, sweep=sw, outputlevel)
      end
      isdone && break
    end
    # return psi
    return totalTimeUsed, rslt
end
