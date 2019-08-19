# See JohnStyleProfile.ipynb for derivation, implementation notes, and test
def asymptoticconfidenceintervalwithcv(datagen, rangefn,
                                       alpha=0.05, rmin=0, rmax=1, 
                                       raiseonerr=False):
    from scipy.stats import f
    from .estimatewithcv import estimatewithcv
    from .asymptoticconfidenceinterval import asymptoticconfidenceinterval
    from math import exp, log
    import numpy as np

    assert rmax >= rmin

    vhat, qmle = estimatewithcv(datagen=datagen, rangefn=rangefn,
                                rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)
    num = qmle['num']
    if num < 2:
        return ((rmin, rmax), (None, None))
    betamle = qmle['betastar']
    deltamle = qmle['deltastar']

    vboundsnocv, qstarnocv = asymptoticconfidenceinterval(
        datagen=lambda: ((c, w, r) for c, w, r, _ in datagen()),
        wmin=rangefn('wmin'), wmax=rangefn('wmax'),
        rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)

    Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

    sumwsq = 0
    for c, w, r, cvs in datagen():
        sumcvsq = np.zeros_like(cvs, dtype='float64')
        break

    n = 0
    for c, w, r, cvs in datagen():
        if c > 0:
            sumwsq += c * (w - 1) * (w - 1)
            sumcvsq += c * np.square(cvs)
            n += c

    assert n == num

    wscale = max(1, np.sqrt(sumwsq / n))
    cvscale = np.maximum(1, np.atleast_1d(np.sqrt(sumcvsq / n)))
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
        gamma, beta = p[0:2]
        logcost = -Delta

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = gamma + (beta + sign * wscale * r) * (w / wscale) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

        assert n == num

        if n > 0:
            logcost /= n

        return (-n * exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = p[0:2]
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = gamma + (beta + sign * wscale * r) * (w / wscale) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
                jac[2:] += jaclogcost * nicecvs

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -(n / rscale) * exp(logcost)
        jac[0] += 1 / rscale
        jac[1] += 1 / (wscale * rscale)

        return jac


    def hessdualobjective(p, sign):
        gamma, beta = p[0:2]
        logcost = -Delta
        jac = np.zeros_like(p)
        hess = np.zeros((len(p),len(p)))

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = gamma + (beta + sign * wscale * r) * (w / wscale) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)
                jac[2:] += jaclogcost * nicecvs

                hesslogcost = c * hesslogstar(denom)
                coeffs = np.hstack(( 1, w / wscale, nicecvs ))
                hess += hesslogcost * np.outer(coeffs, coeffs)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n
            hess /= n

        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * exp(logcost)

        return hess

    consE = np.array([
        np.hstack(( 1, w  / wscale, cv / cvscale ))
        for w, cv in rangefn()
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
        d = np.array([ -sign*w*r + tiny
                       for w, cv in rangefn()
                       for r in (rmin, rmax)
                     ],
                     dtype='float64')

        if qstarnocv[what] is None:
            minsr = min(sign*rmin, sign*rmax)
            x0 = [ (num - qmle['betastar']) + 2 * tiny,
                    wscale * (qmle['betastar'] - (1 + 1 / wscale) * minsr) ] + [
                   0.0 for _ in cvscale
                 ]
        else:
            x0 = [ qstarnocv[what]['gammastar'] + 2 * tiny,
                   wscale * qstarnocv[what]['betastar'] ] + [
                   0.0 for _ in cvscale
                 ]

        if raiseonerr:
           active = np.nonzero(consE.dot(x0) - d < 0)[0]
           from pprint import pformat
           assert active.size == 0, pformat({
                   'cons': consE.dot(x0) - d,
                   'd': d,
                   'consE.dot(x0)': consE.dot(x0),
                   'active': active,
                   'x0': x0,
                   'qstarnocv[{}]'.format(what): qstarnocv[what],
               })

#        from .gradcheck import gradcheck, hesscheck
#        gradcheck(f=lambda p: dualobjective(p, sign),
#                  jac=lambda p: jacdualobjective(p, sign),
#                  x=x0,
#                  what='dualobjective')
#
#        hesscheck(jac=lambda p: jacdualobjective(p, sign),
#                  hess=lambda p: hessdualobjective(p, sign),
#                  x=x0,
#                  what='jacdualobjective')

        # NB: things i've tried
        # slsqp: 10.56s/it, appears reliable
        # cvxopt: 19.72s/it, appears reliable

        from scipy.optimize import minimize
        optresult = minimize(method='slsqp',
                             fun=dualobjective,
                             x0=x0,
                             args=(sign,),
                             jac=jacdualobjective,
                             #hess=hessdualobjective,
                             constraints=[{
                                 'type': 'ineq',
                                 'fun': lambda x: consE.dot(x) - d,
                                 'jac': lambda x: consE
                             }],
                             options={
                                'ftol': 1e-12,
                                'maxiter': 1000,
                             },
                    )
        if raiseonerr:
            from pprint import pformat
            assert optresult.success, pformat(optresult)

        fstar, xstar = optresult.fun, optresult.x

#        from cvxopt import solvers, matrix
#        def F(x=None, z=None):
#            if x is None: return 0, matrix(x0)
#            p = np.reshape(np.array(x), -1)
#            f = dualobjective(p, sign)
#            jf = jacdualobjective(p, sign)
#            Df = matrix(jf).T
#            if z is None: return f, Df
#            hf = z[0] * hessdualobjective(p, sign)
#            H = matrix(hf, hf.shape)
#            return f, Df, H
#
#        soln = solvers.cp(F,
#                          G=-matrix(consE, consE.shape),
#                          h=-matrix(d),
#                          kktsolver='ldl',
#                          options={'show_progress':False,
#                                   'kktreg': 1e-9})
#        if raiseonerr:
#            from pprint import pformat
#            assert soln['status'] == 'optimal', pformat(soln)
#
#        xstar = soln['x']
#        fstar = soln['primal objective']

        gammastar = xstar[0]
        betastar = xstar[1] / wscale
        deltastar = xstar[2:] / cvscale
        kappastar = (-rscale * fstar + gammastar + betastar) / num
        vbound = -sign * fstar

        qfunc = lambda c, w, r, cvs: kappastar * c / (gammastar + betastar * w + np.asscalar(np.dot(deltastar, cvs)) + sign * w * r)

        from scipy.special import xlogy

        retvals.append(
           (vbound,
            {
                'gammastar': gammastar,
                'betastar': betastar,
                'deltastar': deltastar,
                'kappastar': kappastar,
                'qfunc': qfunc,
            })
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])
