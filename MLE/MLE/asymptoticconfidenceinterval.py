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

    sumwsq = sum(c * w * w for c, w, _ in datagen())
    wscale = max(1, np.sqrt(sumwsq / num))
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
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

        assert n == num

        if n > 0:
            logcost /= n

        return (-n * exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -(n / rscale) * exp(logcost)
        jac[0] += 1 / rscale
        jac[1] += 1 / (wscale * rscale)

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
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)

                hesslogcost = c * hesslogstar(denom)
                hess[0][0] += hesslogcost
                hess[0][1] += hesslogcost * (w / wscale)
                hess[1][1] += hesslogcost * (w / wscale) * (w / wscale)

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
        [ 1, w / wscale ]
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
                          wscale * (qmle['betastar'] - (1 + 1 / wscale) * minsr)
                        )

        x0 = np.array([ gamma0, beta0 ])

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
        #
        # scipy.minimize method='slsqp': 3.78 it/s, sometimes fails
        # sqp with quadprog: 1.75 it/s, sometimes fails
        # sqp with cvxopt.qp: 1.05 s/it, reliable
        # cvxopt.cp: 1.37 s/it, reliable
        # minimize_ipopt: 4.85 s/it, reliable

##       from ipopt import minimize_ipopt
##       optresult = minimize_ipopt(
##                           options={
##                              'tol': 1e-12,
#        from scipy.optimize import minimize
#        optresult = minimize(method='slsqp',
#                             options={
#                               'ftol': 1e-12,
#                               'maxiter': 1000,
#                            },
#                            fun=dualobjective,
#                            x0=x0,
#                            args=(sign,),
#                            jac=jacdualobjective,
#                            #hess=hessdualobjective,
#                            constraints=[{
#                                'type': 'ineq',
#                                'fun': lambda x: consE.dot(x) - d,
#                                'jac': lambda x: consE
#                            }],
#                   )
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
                x0=x0,
                strict=True,
        )

#        from cvxopt import solvers, matrix
#        def F(x=None, z=None):
#            if x is None: return 0, matrix(x0)
#            p = np.array([ x[0], x[1] ])
#            f = dualobjective(p, sign)
#            jf = jacdualobjective(p, sign)
#            Df = matrix(jf).T
#            if z is None: return f, Df
#            hf = z[0] * hessdualobjective(p, sign)
#            H = matrix(hf, hf.shape)
#            return f, Df, H
#
#        solvers.options['show_progress'] = False
#        soln = solvers.cp(F,
#                          G=-matrix(consE, consE.shape),
#                          h=-matrix(d))
#        if raiseonerr:
#            from pprint import pformat
#            assert soln['status'] == 'optimal', pformat(soln)
#
#        xstar = soln['x']
#        fstar = soln['primal objective']

        kappastar = (-rscale * fstar + xstar[0] + xstar[1] / wscale) / num
        gammastar = xstar[0]
        betastar = xstar[1] / wscale

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
