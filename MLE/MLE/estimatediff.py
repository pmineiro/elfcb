# See estimateDifference.ipynb for derivation, implementation notes, and test
def estimatediff(datagen, umin, umax, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, censored=False):
    import numpy as np
    from scipy.optimize import minimize

    assert umin >= 0
    assert umin < 1
    assert umax > 1
    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin
    assert not censored

    num, sumusq, sumwsq = 0, 0, 0
    for c, u, w, _ in datagen():
        num += c
        sumusq += c * (u - 1)**2
        sumwsq += c * (w - 1)**2

    assert num >= 1
    uscale = max(1, np.sqrt(sumusq / num))
    wscale = max(1, np.sqrt(sumwsq / num))

    # solve dual

    def logstar(x):
        from math import log

        return log(x) if x > 1 else -1.5 + 2.0*x - 0.5*x*x

    def jaclogstar(x):
        return 1/x if x > 1 else 2 - x

    def hesslogstar(x):
        return -1/(x*x) if x > 1 else -1


    def dualobjective(p):
        gamma, tau = p
        cost = 0

        for c, u, w, r in datagen():
            if c > 0:
                denom = num*(1 + gamma * (u - 1) / uscale + tau * (w - 1) / wscale)
                cost -= c * logstar(denom)

        cost /= num

        return cost

    def jacdualobjective(p):
        gamma, tau = p
        jac = np.zeros_like(p)

        for c, u, w, r in datagen():
            if c > 0:
                denom = num*(1 + gamma * (u - 1) / uscale + tau * (w - 1) / wscale)
                jaccost = c * jaclogstar(denom)
                jac[0] -= (u - 1) * jaccost / uscale
                jac[1] -= (w - 1) * jaccost / wscale

        return jac

    x0 = [ 0.0, 0.0 ]

    from .gradcheck import gradcheck, hesscheck
    gradcheck(f=dualobjective,
              jac=jacdualobjective,
              x=x0,
              what='dualobjective')

    consE = np.array([
                [ (u - 1) / uscale, (w - 1) / wscale ]
                for u in (umin, umax)
                for w in (wmin, wmax)
            ],
            dtype='float64')
    d = np.array([ -num
                   for u in (umin, umax)
                   for w in (wmin, wmax)
                 ],
                 dtype='float64')

    from scipy.optimize import minimize
    optresult = minimize(fun=dualobjective,
                         x0=x0,
                         jac=jacdualobjective,
                         constraints=[{
                             'type': 'ineq',
                             'fun': lambda x: consE.dot(x) - d,
                             'jac': lambda x: consE
                         }],
                         method='slsqp',
                         options={
                            'ftol': 1e-12,
                            'maxiter': 1000,
                         })

    fstar, xstar = optresult.fun, optresult.x
    if raiseonerr:
        from pprint import pformat
        assert optresult.success, pformat(optresult)

    gammastar, taustar = xstar

    qfunc = lambda c, u, w, r, n=num, g=gammastar, t=taustar, us=uscale, ws=wscale: (c/n)/(1 + g * (u - 1) / us + t * (w - 1) / ws)

    deltavhat = 0
    sumofuminusw = 0
    for c, u, w, r in datagen():
        q = qfunc(c, u, w, r)
        deltavhat += q * (u - w) * r
        sumofuminusw += q * (u - w)

    deltavmin = deltavhat + min(rmin * -sumofuminusw, rmax * -sumofuminusw)
    deltavmax = deltavhat + max(rmin * -sumofuminusw, rmax * -sumofuminusw)
    deltavhat = (deltavmin + deltavmax) / 2

    deltavmin, deltavmax, deltavhat = (min(rmax - rmin, max(rmin - rmax, x))
                                       for x in (deltavmin,
                                                 deltavmax,
                                                 deltavhat))

    return deltavhat, {
            'deltavmin': deltavmin,
            'deltavmax': deltavmax,
            'num': num,
            'gammastar': gammastar,
            'taustar': taustar,
            'qfunc': qfunc,
            'primal': num * fstar
    }
