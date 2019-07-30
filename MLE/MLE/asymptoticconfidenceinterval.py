# See JohnStyleProfile.ipynb for derivation, implementation notes, and test
def asymptoticconfidenceinterval(datagen, wmin, wmax, alpha=0.05,
                                 rmin=0, rmax=1, raiseonerr=False):
    from scipy.special import xlogy
    from scipy.stats import f
    from .estimate import estimate
    from math import exp, log
    import numpy as np

    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin

    vhat, qmle = estimate(datagen=datagen, wmin=wmin, wmax=wmax,
                          rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)
    num = qmle['num']
    if num < 2:
        return ((rmin, rmax), (None, None))
    betamle = qmle['betastar']

    Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

    rscale = max(1.0, np.abs(rmin), np.abs(rmax))

    # solve dual

    tiny = 1e-5
    logtiny = log(tiny)

    def safedenom(x):
        return x if x > tiny else exp(logstar(x))

    def logstar(x):
        return log(x) if x > tiny else -1.5 + logtiny + 2.0*(x/tiny) - 0.5*(x/tiny)*(x/tiny)

    def jaclogstar(x):
        return 1/x if x > tiny else (2.0 - (x/tiny))/tiny

    def hesslogstar(x):
        return -1/(x*x) if x > tiny else -1/(tiny*tiny)

    def dualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wmax * r) * (w / wmax)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

        assert n == num

        if n > 0:
            logcost /= n

        return (-n * exp(logcost) + gamma + beta / wmax) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wmax * r) * (w / wmax)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -(n / rscale) * exp(logcost)
        jac[0] += 1 / rscale
        jac[1] += 1 / (wmax * rscale)

        return jac

    def hessdualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta
        jac = np.zeros_like(p)
        hess = np.zeros((2,2))

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wmax * r) * (w / wmax)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)

                hesslogcost = c * hesslogstar(denom)
                hess[0][0] += hesslogcost
                hess[0][1] += hesslogcost * (w / wmax)
                hess[1][1] += hesslogcost * (w / wmax) * (w / wmax)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n
            hess /= n

        hess[1][0] = hess[0][1]
        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * exp(logcost)

        return hess

    consE = np.array([
        [ 1, w / wmax ]
        for w in (wmin, wmax)
        for r in (rmin, rmax)
    ], dtype='float64')

    retvals = []

    easybounds = [ (qmle['vmin'] <= rmin + tiny, rmin),
                   (qmle['vmax'] >= rmax - tiny, rmax) ]
    for what in range(2):
        if easybounds[what][0]:
            retvals.append((easybounds[what][1], None))
            continue

        sign = 1 - 2 * what
        d = np.array([ -sign*w*r  + tiny
                       for w in (wmin, wmax)
                       for r in (rmin, rmax)
                     ],
                     dtype='float64')

        minsr = min(sign*rmin, sign*rmax)
        gamma0, beta0 = ( (num - qmle['betastar']) + 2 * tiny,
                          wmax * (qmle['betastar'] - (1 + 1 / wmax) * minsr)
                        )

#        from .gradcheck import gradcheck, hesscheck
#        gradcheck(f=lambda p: dualobjective(p, sign),
#                  jac=lambda p: jacdualobjective(p, sign),
#                  x=[gamma0, beta0],
#                  what='dualobjective')
#
#        hesscheck(jac=lambda p: jacdualobjective(p, sign),
#                  hess=lambda p: hessdualobjective(p, sign),
#                  x=[gamma0, beta0],
#                  what='jacdualobjective')

#        # ipopt is tyically slower than sqp
#        # for low-dimensional problems
#        # but ipopt is _very_ reliable
#
#        from ipopt import minimize_ipopt
#        optresult = minimize_ipopt(
#                             fun=dualobjective,
#                             x0=[gamma0, beta0],
#                             args=(sign,),
#                             jac=jacdualobjective,
#                             #hess=hessdualobjective,
#                             constraints=[{
#                                 'type': 'ineq',
#                                 'fun': lambda x: consE.dot(x) - d,
#                                 'jac': lambda x: consE
#                             }],
#                             options={
#                                #'ftol': 1e-12,
#                                'tol': 1e-12,
#                                'maxiter': 1000,
#                             },
#                    )
#        if raiseonerr:
#            from pprint import pformat
#            assert optresult.success, pformat(optresult)
#
#        fstar, xstar = optresult.fun, optresult.x

        from .sqp import sqp
        fstar, xstar = sqp(
                f=lambda p: dualobjective(p, sign),
                gradf=lambda p: jacdualobjective(p, sign),
                hessf=lambda p: hessdualobjective(p, sign),
                E=consE,
                d=d,
                x0=[ gamma0, beta0 ],
                strict=True,
        )

        kappastar = (-rscale * fstar + xstar[0] + xstar[1] / wmax) / num
        gammastar = xstar[0]
        betastar = xstar[1] / wmax

        qfunc = lambda c, w, r, kappa=kappastar, gamma=gammastar, beta=betastar, s=sign: kappa * c / (gamma + (beta + s * r) * w)

        vbound = -sign * rscale * fstar

        retvals.append(
           (vbound,
            {
                'gammastar': gammastar,
                'betastar': betastar,
                'kappastar': kappastar,
                'qfunc': qfunc,
            })
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])
