# See JohnStyleProfile.ipynb for derivation, implementation notes, and test
def asymptoticconfidenceinterval(datagen, wmin, wmax, alpha=0.05,
                                 rmin=0, rmax=1, raiseonerr=False):
    from scipy.optimize import nnls
    from scipy.special import xlogy
    from scipy.stats import f
    from .estimate import estimate
    from .sqp import sqp
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

    delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

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
        logcost = -delta

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + beta * w + sign * w * r
                mledenom = betamle * (w - 1) + num
                logcost += c * (logstar(denom) - logstar(mledenom))

        if n > 0:
            logcost /= n

        return -exp(logcost) + gamma / n + beta / n

    def jacdualobjective(p, sign):
        gamma, beta = p
        logcost = -delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + beta * w + sign * w * r
                mledenom = betamle * (w - 1) + num
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += w * jaclogcost

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -exp(logcost)
        jac[0] += 1/n
        jac[1] += 1/n

        return jac

    def hessdualobjective(p, sign):
        gamma, beta = p
        logcost = -delta
        jac = np.zeros_like(p)
        hess = np.zeros((2,2))

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + beta * w + sign * w * r
                mledenom = betamle * (w - 1) + num
                logcost += c * (logstar(denom) - logstar(mledenom))
                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += w * jaclogcost

                hesslogcost = c * hesslogstar(denom)
                hess[0][0] += hesslogcost
                hess[0][1] += w * hesslogcost
                hess[1][1] += w * w * hesslogcost

        if n > 0:
            logcost /= n
            jac /= n
            hess /= n

        hess[1][0] = hess[0][1]
        hess += np.outer(jac, jac)
        hess *= -exp(logcost)

        return hess

    def sumstats(p, sign):
        gamma, beta = p
        logcost = -delta

        sumofone = 0
        sumofw = 0
        sumofwr = 0
        n = 0
        total = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + beta * w + sign * w * r
                mledenom = betamle * (w - 1) + num
                logcost += c * (logstar(denom) - logstar(mledenom))

                safe = safedenom(denom)

                try:
                    sumofone += c / safe
                    sumofw += c*w / safe
                    sumofwr += c*w*r / safe
                except Exception as e:
                    from pprint import pformat
                    print(pformat({
                        'e': e,
                        'denom': denom,
                        'safedenom': safe,
                        'consE(p)-d': consE.dot(p)-d,
                    }))
                    raise

        if n > 0:
            logcost /= n

        kappa = exp(logcost)
        sumofone *= kappa
        sumofw *= kappa
        sumofwr *= kappa

        return sumofone, sumofw, sumofwr, kappa

    consE = np.array([
        [ 1/max(w, 1), min(w, 1) ]
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
        d = np.array([ -sign*min(w, 1)*r  + tiny/max(w, 1)
                       for w in (wmin, wmax)
                       for r in (rmin, rmax)
                     ],
                     dtype='float64')

        gamma0, beta0 = 1.0, 0.5

        fstar, [ gammastar, betastar ] = sqp(
                f=lambda p: dualobjective(p, sign),
                gradf=lambda p: jacdualobjective(p, sign),
                hessf=lambda p: hessdualobjective(p, sign),
                E=consE,
                d=d,
                x0=[ gamma0, beta0 ],
                strict=True,
                abscondfac=1e-12,
        )

        x = [ gammastar, betastar ]

        # recover primal

        sumone, sumw, sumwr, kappastar = sumstats(x, sign)

        remone = 1 - sumone
        remw = 1 - sumw

        if remone >= 0 and remw >= 0:
            A = np.array([ [ 1, w ] for w in (wmin, wmax) ])
            b = np.array([ remone, remw ])

            qexlst, _ = nnls(A.T, b)
            rmissing = rmin if sign > 0 else rmax

            qex = { (w, rmissing): v for w, v in zip((wmin, wmax), qexlst) }
        else:
            qex = {}

        sumone += sum(q for _, q in qex.items())
        sumw += sum(w*q for (w, _), q in qex.items())
        sumwr += sum(w*r*q for (w, r), q in qex.items())

        retvals.append(
           (sumwr,
            {
                'remone': remone,
                'remw': remw,
                'sumone': sumone,
                'sumw': sumw,
                'gammastar': gammastar,
                'betastar': betastar,
                'kappastar': kappastar,
                'qex': qex,
                'qfunc': lambda c, w, r: kappastar * c / (gammastar + betastar * w + sign * w * r)
            })
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])
