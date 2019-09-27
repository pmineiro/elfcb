# See ciDifference.ipynb for derivation, implementation notes, and test
def cidifference(datagen, umin, umax, wmin, wmax, alpha=0.05,
                 rmin=0, rmax=1, raiseonerr=False):
    import numpy as np
    from cvxopt import solvers, matrix
    from math import log, exp
    from scipy.stats import f
    from .estimatediff import estimatediff

    assert umin >= 0
    assert umin < 1
    assert umax > 1
    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin

    _, mle = estimatediff(datagen, umin, umax, wmin, wmax, rmin, rmax, raiseonerr=raiseonerr)
    num = mle['num']

    Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)
    phi = Delta - mle['primal']

    rscale = max(1.0, rmax - rmin)

    def dualobjective(p, sign):
        beta, gamma, tau = p

        logcost = -phi
        n = 0
        
        for c, u, w, r in datagen():
            if c > 0:
                n += c
                denom = beta + gamma * u + tau * w + sign * (u - w) * r
                logcost += c * log(denom)

        logcost /= n
        cost = exp(logcost)

        return (- beta - gamma - tau + n * cost) / rscale

    def jacdualobjective(p, sign):
        beta, gamma, tau = p

        logcost = -phi
        jaclogcost = np.zeros(3)
        n = 0

        for c, u, w, r in datagen():
            if c > 0:
                n += c
                denom = beta + gamma * u + tau * w + sign * (u - w) * r
                logcost += c * log(denom)
                gradlogcost = c / denom
                jaclogcost[0] += gradlogcost
                jaclogcost[1] += u * gradlogcost
                jaclogcost[2] += w * gradlogcost

        logcost /= n
        cost = exp(logcost)

        return (-np.ones(3) + exp(logcost) * jaclogcost) / rscale

    def hessdualobjective(p, sign):
        beta, gamma, tau = p

        logcost = -phi
        jaclogcost = np.zeros(3)
        hesslogcost = np.zeros((3,3))
        n = 0

        for c, u, w, r in datagen():
            if c > 0:
                n += c
                denom = beta + gamma * u + tau * w + sign * (u - w) * r
                logcost += c * log(denom)
                gradlogcost = c / denom
                jaclogcost[0] += gradlogcost
                jaclogcost[1] += u * gradlogcost
                jaclogcost[2] += w * gradlogcost
                gradgradlogcost = -c / denom**2
                hesslogcost[0, 0] += gradgradlogcost
                hesslogcost[0, 1] += gradgradlogcost * u
                hesslogcost[0, 2] += gradgradlogcost * w
                hesslogcost[1, 1] += gradgradlogcost * u**2
                hesslogcost[1, 2] += gradgradlogcost * u * w
                hesslogcost[2, 2] += gradgradlogcost * w**2


        logcost /= n
        cost = exp(logcost)

        hesslogcost[1, 0] = hesslogcost[0, 1]
        hesslogcost[2, 0] = hesslogcost[0, 2]
        hesslogcost[2, 1] = hesslogcost[1, 2]

        return (cost * (hesslogcost + np.outer(jaclogcost, jaclogcost) / n)) / rscale

    # solve

    consE = np.array([
        [ 1, u, w ]
        for u in (umin, umax)
        for w in (wmin, wmax)
        for r in (rmin, rmax)
    ], dtype='float64')

    retvals = []

    easybounds = [ (mle['deltavmin'] <= (rmin - rmax) + 1e-4, rmin - rmax),
                   (mle['deltavmax'] >= (rmax - rmin) - 1e-4, rmax - rmin) ]
    for what in range(2):
        if easybounds[what][0]:
            retvals.append((easybounds[what][1], None))
            continue

        sign = 1 - 2 * what
        x0 = np.array([num, -rmin, rmax] if sign > 0 else [num, rmax, -rmin],
                      dtype='float64')

        d = np.array([ -sign * (u - w) * r + 1e-4
                       for u in (umin, umax)
                       for w in (wmin, wmax)
                       for r in (rmin, rmax)
                     ], dtype='float64')

        # from .gradcheck import gradcheck, hesscheck
        # gradcheck(f=lambda p: dualobjective(p, sign),
        #           jac=lambda p: jacdualobjective(p, sign),
        #           x=x0,
        #           what='dualobjective')
    
        # hesscheck(jac=lambda p: jacdualobjective(p, sign),
        #           hess=lambda p: hessdualobjective(p, sign),
        #           x=x0,
        #           what='jacdualobjective')
    
        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)
            f = -dualobjective(x, sign)
            jf = -jacdualobjective(x, sign)
            Df = matrix(jf).T
            if z is None: return f, Df
            hf = -z[0] * hessdualobjective(x, sign)
            H = matrix(hf, hf.shape)
            return f, Df, H

        soln = solvers.cp(F,
                          G=-matrix(consE, consE.shape),
                          h=-matrix(d),
                          options={'show_progress': False})

        if raiseonerr:
            from pprint import pformat
            assert soln['status'] == 'optimal', pformat({
                'soln': soln,
                'phi': phi,
                'mle': mle,
            })

        betastar, gammastar, taustar = soln['x']
        fstar = -rscale * soln['primal objective']
        kappastar = (fstar + betastar + gammastar + taustar) / num

        qfunc = lambda c, u, w, r, kappa=kappastar, beta=betastar, gamma=gammastar, tau=taustar: c*kappa / (beta + gamma * u + tau * w + (u - w) * r)

        vbound = sign * fstar

        retvals.append( ( vbound,
                          {
                              'kappastar': kappastar,
                              'betastar': betastar,
                              'gammastar': gammastar,
                              'taustar': taustar,
                              'qfunc': qfunc,
                              'phi': phi,
                              'mle': mle,
                          } 
                      ) )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])
