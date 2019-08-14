# See estimate.ipynb for derivation, implementation notes, and test
def estimatewithcv(datagen, rangefn, rmin=0, rmax=1, raiseonerr=False):
    import numpy as np
    from .estimate import estimate

    assert rmax >= rmin

    vhat, qmle = estimate(datagen=lambda: ((c, w, r) for c, w, r, _ in datagen()), wmin=rangefn('wmin'), wmax=rangefn('wmax'), rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)

    num = qmle['num']
    assert num >= 1

    sumwsq = 0
    for c, w, r, cvs in datagen():
        sumcvsq = np.zeros_like(cvs)
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

    # solve dual

    def logstar(x):
        from math import log

        return log(x) if x > 1 else -1.5 + 2.0*x - 0.5*x*x

    def jaclogstar(x):
        return 1/x if x > 1 else 2 - x

    def hesslogstar(x):
        return -1/(x*x) if x > 1 else -1


    def dualobjective(p):
        cost = 0
        n = 0
        for c, w, r, cvs in datagen():
            if c > 0:
                n += c
                nicecvs = cvs / cvscale
                denom = num*(1 + p[0] * (w - 1) / wscale + np.dot(p[1:], nicecvs))
                cost -= c * logstar(denom) 

        assert n == num

        if n > 0:
            cost /= n

        return cost 

    def jacdualobjective(p):
        jac = np.zeros_like(p)

        for c, w, r, cvs in datagen():
            if c > 0:
                nicecvs = cvs / cvscale
                denom = num*(1 + p[0] * (w - 1) / wscale + np.dot(p[1:], nicecvs))
                jacdenom = c * jaclogstar(denom)
                jac[0] -= (w - 1) * jacdenom / wscale
                jac[1:] -= jacdenom * nicecvs

        return jac

    def hessdualobjective(p):
        hess = np.zeros((len(p),len(p)))

        for c, w, r, cvs in datagen():
            if c > 0:
                nicecvs = cvs / cvscale
                denom = num*(1 + p[0] * (w - 1) / wscale + np.dot(p[1:], nicecvs))
                hessdenom = c * hesslogstar(denom)
                coeffs = np.hstack(( (w - 1) / wscale, nicecvs ))
                hess -= hessdenom * np.outer(coeffs, coeffs)

        hess *= num

        return hess

    x0 = [ qmle['betastar'] * wscale / num ] + [ 0.0 for i, _ in enumerate(cvscale) ]

#    from .gradcheck import gradcheck, hesscheck
#    gradcheck(f=dualobjective,
#              jac=jacdualobjective,
#              x=x0,
#              what='dualobjective')
#
#    hesscheck(jac=jacdualobjective,
#              hess=hessdualobjective,
#              x=x0,
#              what='jacdualobjective')

    consE = np.array([
                np.hstack(((w - 1) / wscale, cv / cvscale))
                for w, cv in rangefn()
            ],
            dtype='float64')
    d = np.array([ -1
                   for _ in rangefn()
                 ],
                 dtype='float64')

    # NB: slsqp is faster than cvxopt and appears reliable
    #
    # scipy.minimize method='slsqp': 7.25 s/it, reliable
    # cvxopt.cp: 9.43s/it, reliable

    from scipy.optimize import minimize
    optresult = minimize(fun=dualobjective,
                         x0=x0,
                         jac=jacdualobjective,
                         #hess=hessdualobjective,
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

#    from cvxopt import solvers, matrix
#    def F(x=None, z=None):
#        if x is None: return 0, matrix(x0)
#        p = np.reshape(np.array(x), -1)
#        f = dualobjective(p)
#        jf = jacdualobjective(p)
#        Df = matrix(jf).T
#        if z is None: return f, Df
#        hf = z[0] * hessdualobjective(p)
#        H = matrix(hf, hf.shape)
#        return f, Df, H
#    soln = solvers.cp(F,
#                      G=-matrix(consE, consE.shape),
#                      h=-matrix(d),
#                      options={'show_progress': False})
#    fstar, xstar = soln['primal objective'], soln['x']
#    if raiseonerr:
#        from pprint import pformat
#        assert soln['status'] == 'optimal', pformat(soln)

    vhat = 0
    rawsumofw = 0
    for c, w, r, cvs in datagen():
        if c > 0:
            nicecvs = cvs / cvscale
            denom = num * (1 + xstar[0] * (w - 1) / wscale + np.dot(xstar[1:], nicecvs))
            q = c / denom
            vhat += q * w * r
            rawsumofw += q * w

    if raiseonerr:
        from pprint import pformat
        assert (rawsumofw <= 1.0 + 1e-4 and
                np.all(consE.dot(xstar) >= d - 1e-4)
               ), pformat({
                   'rawsumofw': rawsumofw,
                   'consE.dot(xstar) - d': consE.dot(xstar) - d,
               })

    vmin = max(rmin, vhat + max(0.0, 1.0 - rawsumofw) * rmin)
    vmax = min(rmax, vhat + max(0.0, 1.0 - rawsumofw) * rmax)
    vhat += max(0.0, 1.0 - rawsumofw) * (rmax - rmin) / 2.0
    vhat = min(rmax, max(rmin, vhat))

    betastar = xstar[0] * (num / wscale)
    deltastar = num * (xstar[1:] / cvscale)
    qfunc = lambda c, w, r, cvs: np.asscalar(c / (num + betastar * (w - 1) + np.dot(deltastar, cvs)))

    return vhat, {
        'vmin': vmin,
        'vmax': vmax,
        'num': num,
        'betastar': betastar,
        'deltastar': deltastar,
        'qfunc': qfunc,
        'rawsumofw': rawsumofw
    }
