# See estimate.ipynb for derivation, implementation notes, and test
def estimatewithcv(datagen, wmin, wmax, cvmin, cvmax, rmin=0, rmax=1, raiseonerr=False):
    import numpy as np
    from scipy.special import xlogy
    from .sqp import sqp
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

    num = sum(c for c, _, _, _ in datagen())
    assert num >= 1

    cvscale = np.maximum(1, np.maximum(np.abs(cvmin), np.abs(cvmax)))

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
                denom = num*(1 + p[0] * (w - 1) / wmax + np.dot(p[1:], nicecvs))
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
                denom = num*(1 + p[0] * (w - 1) / wmax + np.dot(p[1:], nicecvs))
                jacdenom = c * jaclogstar(denom)
                jac[0] -= (w - 1) * jacdenom / wmax
                jac[1:] -= jacdenom * nicecvs

        return jac


    def hessdualobjective(p):
        hess = np.zeros((len(p),len(p)))

        for c, w, r, cvs in datagen():
            if c > 0:
                nicecvs = cvs / cvscale
                denom = num*(1 + p[0] * (w - 1) / wmax + np.dot(p[1:], nicecvs))
                hessdenom = c * hesslogstar(denom)
                coeffs = np.hstack(( (w - 1) / wmax, nicecvs ))
                hess -= hessdenom * np.outer(coeffs, coeffs)

        hess *= num

        return hess

    x0 = [ 1.0 ] + [ 1.0 / (i + 1) for i, _ in enumerate(cvscale) ]

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
        np.hstack(([ (w - 1) / wmax ], bitvec / cvscale))
        for w in (wmin, wmax)
        for bitvec in bitgen(cvmin, cvmax)
    ])
    d = np.array([ -1
                   for w in (wmin, wmax)
                   for bitvec in bitgen(cvmin, cvmax)
                 ])

    fstar, xstar = sqp(dualobjective,
                       jacdualobjective,
                       hessdualobjective,
                       consE,
                       d,
                       x0,
                       strict=True)

    vhat = 0
    rawsumofw = 0
    for c, w, r, cvs in datagen():
        if c > 0:
            nicecvs = cvs / cvscale
            denom = num * (1 + xstar[0] * (w - 1) / wmax + np.dot(xstar[1:], nicecvs))
            q = c / denom
            vhat += q * w * r
            rawsumofw += q * w

    if raiseonerr:
        from pprint import pformat
        assert (rawsumofw <= 1.0 + 1e-6 and 
                np.all(consE.dot(xstar) >= d - 1e-6)
               ), pformat({
                   'rawsumofw': rawsumofw,
                   'consE.dot(xstar) - d': consE.dot(xstar) - d,
               })

    vmin = vhat + max(0.0, 1.0 - rawsumofw) * rmin
    vmax = vhat + max(0.0, 1.0 - rawsumofw) * rmax
    vhat += max(0.0, 1.0 - rawsumofw) * (rmax - rmin) / 2.0

    qstar = lambda c, w, r, cvs: c / (num * (1 + xstar[0] * (w - 1) / wmax + np.dot(xstar[1:], cvs / cvscale)))

    from scipy.special import xlogy

    return vhat, {
        'vmin': vmin,
        'vmax': vmax,
        'gammastar': xstar[0] * (num / wmax),
        'deltastar': xstar[1:] * (num / cvscale),
        # TODO: don't materialize
        'qstar': { (w, r, tuple(cvs)): qstar(c, w, r, cvs) for c, w, r, cvs in datagen() },
        'likelihood': sum(xlogy(c/num, qstar(c, w, r, cvs)) for c, w, r, cvs in datagen()),
        'rawsumofw': rawsumofw
    }
