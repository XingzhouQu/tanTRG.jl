"""
TODO:

implement two-site variational product

"""
function rhoMPO(H, beta0, s;tol=1e-6)
    rho = MPO(s, "Id")
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
        fe = -1 * (2*beta0)^-1 * 2*lognorm(rho)
        diff = abs((fe-feold)/feold)
        @show i
        @show diff
        if diff < tol
            break
        end
        i += 1
    end

    return rho
end