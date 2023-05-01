"""
TODO:

implement two-site variational product

"""
function getFe(trRho, beta, lgtr0)
    return -1*(beta)^-1*(log(trRho)+lgtr0)
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
        rho += convert(Float64,(-beta)^i/factorial(big(i))) * Hn
        trRho += convert(Float64, (-beta)^i/factorial(big(i))) * tr(Hn)
        fe = getFe(trRho, beta, log(tr0))
        if i < 2 # at least keep the 1st order
            continue
        end
        if abs((fe-feold)/feold) < tol
            println("SETTN converges at i = $i")
            break
        end
        feold = fe
    end

    swapprime!(rho, 1, 2)
    return rho, log(trRho)+log(tr0)
end
