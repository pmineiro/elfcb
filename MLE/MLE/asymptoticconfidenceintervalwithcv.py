# See JohnStyleProfile.ipynb for derivation, implementation notes, and test
def asymptoticconfidenceintervalwithcv(datagen, wmin, wmax, cvmin, cvmax,
                                       alpha=0.05, rmin=0, rmax=1, 
                                       raiseonerr=False):
    from scipy.stats import f
    from .estimatewithcv import estimatewithcv
    from .asymptoticconfidenceinterval import asymptoticconfidenceinterval
    from math import exp, log
    import numpy as np

    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin
    assert np.all(cvmax >= cvmin)

    def bitgen(minv, maxv):
        def bitgenhelp(vals, minv, maxv, pos, length):
            if pos >= length:
                yield tuple(vals)
            else:
                vals[pos] = minv[pos]
                yield from bitgenhelp(vals, minv, maxv, pos+1, length)
                vals[pos] = maxv[pos]
                yield from bitgenhelp(vals, minv, maxv, pos+1, length)
            
        assert len(minv) == len(maxv)
        length = len(minv)
        yield from bitgenhelp([None]*length, minv, maxv, 0, length) 

    vhat, qmle = estimatewithcv(datagen=datagen, wmin=wmin, wmax=wmax,
                                cvmin=cvmin, cvmax=cvmax,
                                rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)
    num = qmle['num']
    if num < 2:
        return ((rmin, rmax), (None, None))
    betamle = qmle['betastar']
    deltamle = qmle['deltastar']

    vboundsnocv, qstarnocv = asymptoticconfidenceinterval(
        datagen=lambda: ((c, w, r) for c, w, r, _ in datagen()),
        wmin=wmin, wmax=wmax, rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)

    Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

    cvscale = np.maximum(1, np.maximum(np.abs(cvmin), np.abs(cvmax)))
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
                denom = gamma + (beta + sign * wmax * r) * (w / wmax) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

        assert n == num

        if n > 0:
            logcost /= n

        return (-n * exp(logcost) + gamma + beta / wmax) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = p[0:2]
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = gamma + (beta + sign * wmax * r) * (w / wmax) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)
                jac[2:] += jaclogcost * nicecvs

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -(n / rscale) * exp(logcost)
        jac[0] += 1 / rscale
        jac[1] += 1 / (wmax * rscale)

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
                denom = gamma + (beta + sign * wmax * r) * (w / wmax) + np.dot(p[2:], nicecvs)
                mledenom = num + betamle * (w - 1) + np.dot(deltamle, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)
                jac[2:] += jaclogcost * nicecvs

                hesslogcost = c * hesslogstar(denom)
                coeffs = np.hstack(( 1, w / wmax, nicecvs ))
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
        np.hstack(( 1, w  / wmax, bitvec / cvscale ))
        for w in (wmin, wmax)
        for r in (rmin, rmax)
        for bitvec in bitgen(cvmin, cvmax)
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
                       for w in (wmin, wmax)
                       for r in (rmin, rmax)
                       for bitvec in bitgen(cvmin, cvmax)
                     ],
                     dtype='float64')

        if qstarnocv[what] is None:
            minsr = min(sign*rmin, sign*rmax)
            x0 = [ (num - qmle['betastar']) + 2 * tiny,
                   wmax * (qmle['betastar'] - (1 + 1 / wmax) * minsr) ] + [
                   0.0 for _ in cvscale
                 ]
        else:
            x0 = [ qstarnocv[what]['gammastar'] + 2 * tiny,
                   wmax * qstarnocv[what]['betastar'] ] + [
                   0.0 for _ in cvscale
                 ]

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


        # TODO: until my cheesy sqp routine is fixed use slsqp
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

#        # TODO: sometimes just hangs (inside fortan), figure out why
#        from .sqp import sqp
#        fstar, xstar = sqp(
#                f=lambda p: dualobjective(p, sign),
#                gradf=lambda p: jacdualobjective(p, sign),
#                hessf=lambda p: hessdualobjective(p, sign),
#                E=consE,
#                d=d,
#                x0=x0,
#                strict=True,
#        )

        kappastar = (-rscale * fstar + xstar[0] + xstar[1] / wmax) / num
        gammastar = xstar[0]
        betastar = xstar[1] / wmax
        deltastar = xstar[2:] / cvscale
        vbound = -sign * fstar

        qfunc = lambda c, w, r, cvs: kappastar * c / (gammastar + betastar * w + np.dot(deltastar, cvs) + sign * w * r)

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
