# See JohnStyleProfile.ipynb for derivation, implementation notes, and test
def asymptoticconfidenceintervalwithcv(datagen, wmin, wmax, cvmin, cvmax,
                                       alpha=0.05, rmin=0, rmax=1, 
                                       raiseonerr=False):
    from scipy.optimize import nnls
    from scipy.special import xlogy
    from scipy.stats import f
    from .estimate import estimate
    from .sqp import sqp
    from math import exp, log
    import numpy as np
#    from .gradcheck import gradcheck, hesscheck

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
    gammamle = qmle['gammastar']
    deltamle = qmle['deltastar']

    Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

    cvscale = np.maximum(1, np.maximum(np.abs(cvmin), np.abs(cvmax)))

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
        beta, gamma = p[0:2]
        logcost = -Delta

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = beta + gamma * w / wmax + np.dot(p[3:], nicecvs) + sign * w * r
                mledenom = num + gammastar * (w - 1) + np.dot(deltastar, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

        if n > 0:
            logcost /= n

        return -exp(logcost) + gamma / (wmax * n) + beta / n

    def jacdualobjective(p, sign):
        beta, gamma = p[0:2]
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = beta + gamma * w / wmax + np.dot(p[3:], nicecvs) + sign * w * r
                mledenom = num + gammastar * (w - 1) + np.dot(deltastar, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)
                jac[2:] += jaclogcost * nicecvs

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -exp(logcost)
        jac[0] += 1/n
        jac[1] += 1/(wmax * n)

        return jac

    def hessdualobjective(p, sign):
        beta, gamma = p[0:2]
        logcost = -Delta
        jac = np.zeros_like(p)
        hess = np.zeros((len(p),len(p)))

        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = beta + gamma * w / wmax + np.dot(p[3:], nicecvs) + sign * w * r
                mledenom = num + gammastar * (w - 1) + np.dot(deltastar, cvs)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wmax)
                jac[2:] += jaclogcost * nicecvs

                hesslogcost = c * hesslogstar(denom)
                coeffs = np.hstack(( 1, (w - 1) / wmax, nicecvs ))
                hess += hessdenom * np.outer(coeffs, coeffs)

        if n > 0:
            logcost /= n
            jac /= n
            hess /= n

        hess += np.outer(jac, jac)
        hess *= -exp(logcost)

        return hess

    consE = np.array([
        np.hstack(( 1, (w - 1) / wmax, bitvec / cvscale ))
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

        maxwr = max(sign*w*r for w in (wmin, wmax) for r in (rmin, rmax))
        x0 = [ max(maxwr + 2 * tiny, num), 0.0 ] + [ 0.0 for i, _ in enumerate(cvscale) ]

        gradcheck(f=lambda p: dualobjective(p, sign),
                  jac=lambda p: jacdualobjective(p, sign),
                  x=x0,
                  what='dualobjective')
    
        hesscheck(jac=lambda p: jacdualobjective(p, sign),
                  hess=lambda p: hessdualobjective(p, sign),
                  x=x0,
                  what='jacdualobjective')

        fstar, xstar = sqp(
                f=lambda p: dualobjective(p, sign),
                gradf=lambda p: jacdualobjective(p, sign),
                hessf=lambda p: hessdualobjective(p, sign),
                E=consE,
                d=d,
                x0=x0,
                strict=True,
                abscondfac=1e-12,
        )

        betastar = xstar[0]
        gammastar = xstar[1] / wmax
        deltastar = xstar[2:] / cvsscale
        vlb = -num * fstar
        kappastar = vlb + betastar + gammastar

        retvals.append(
           (vlb,
            {
                'betastar': betastar,
                'gammastar': gammastar,
                'deltastar': deltastar,
                'kappastar': kappastar,
                'qfunc': lambda c, w, r, cvs: kappastar * c / (betastar + gammastar * w + np.dot(deltastar, cvs) + sign * w * r),
            })
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])
