"""
TODO:

implement two-site variational product

"""
function getFe(trRho, beta, lgtr0)
    return -1*(beta)^-1*(log(trRho)+lgtr0)
end

function rhoMPO(H::MPO, beta0::Number, s; tol=1e-6)
    nmaxHn = 8
    lstrHn = Vector{Float64}(undef, nmaxHn)

    Hid = MPO(s, "Id")
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



    lgnrm = lognorm(rho)
    nrm0 = exp(lgnrm)
    rho /= nrm0

    Hn = copy(H)
    Hn /= nrm0
    rho += (-beta0) * Hn

    fe = -1 * (2*beta0)^-1 * 2*lognorm(rho)

    i = 2
    while i<1000
        Hn = apply(H, Hn)
        rho += (-beta0)^i/factorial(i) * Hn
        feold = fe
        lgnrm = lognorm(rho)
        fe = -1 * (2*beta0)^-1 * 2*lgnrm
        diff = abs((fe-feold)/feold)
        println("H^i for i = $i")
        println("relative Δ Fe =  $diff")
        if diff < tol
            break
        end
        i += 1
    end

    return rho, lgnrm
end