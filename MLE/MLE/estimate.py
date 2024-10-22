# See estimate.ipynb for derivation, implementation notes, and test
def estimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, censored=False):
    import numpy as np
    from scipy.optimize import brentq

    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin

    num = sum(c for c, w, r in datagen())
    assert num >= 1

    # solve dual

    def sumofw(beta):
        return sum((c * w)/((w - 1) * beta + num)
                   for c, w, _ in datagen()
                   if c > 0)

    # fun fact about the MLE:
    #
    # if \frac{1}{n} \sum_n w_n < 1 then \beta^* wants to be negative
    # but as wmax \to \infty, lower bound on \beta^* is 0
    # therefore the estimate becomes
    #
    # \hat{V}(\pi) = \left( \frac{1}{n} \sum_n w_n r_n \right) +
    #                \left( 1 - \frac{1}{n} \sum_n w_n \right) \rho
    #
    # where \rho is anything between rmin and rmax

    def graddualobjective(beta):
        return sum(c * (w - 1)/((w - 1) * beta + num)
                   for c, w, _ in datagen()
                   if c > 0)

    betamax = min( ((num - c) / (1 - w)
                    for c, w, _ in datagen()
                    if w < 1 and c > 0 ),
                   default=num / (1 - wmin))
    betamax = min(betamax, num / (1 - wmin))

    betamin = max( ((num - c) / (1 - w)
                    for c, w, _ in datagen()
                    if w > 1 and c > 0 ),
                   default=num / (1 - wmax))
    betamin = max(betamin, num / (1 - wmax))

    gradmin = graddualobjective(betamin)
    gradmax = graddualobjective(betamax)
    if gradmin * gradmax < 0:
        betastar = brentq(f=graddualobjective, a=betamin, b=betamax)
    elif gradmin < 0:
        betastar = betamin
    else:
        betastar = betamax

    remw = max(0.0, 1.0 - sumofw(betastar))

    if censored:
        vnumhat = 0
        vdenomhat = 0

        for c, w, r in datagen():
            if c > 0:
                if r is not None:
                    vnumhat += w*r* c/((w - 1) * betastar + num)
                    vdenomhat += w*1* c/((w - 1) * betastar + num)

        if np.allclose(vdenomhat, 0):
            vhat = vmin = vmax = None
        else:
            vnummin = vnumhat + remw * rmin
            vdenommin = vdenomhat + remw
            vmin = min([ vnummin / vdenommin, vnumhat / vdenomhat ])

            vnummax = vnumhat + remw * rmax
            vdenommax = vdenomhat + remw
            vmax = max([ vnummax / vdenommax, vnumhat / vdenomhat ])

            vhat = 0.5*(vmin + vmax)
    else:
        vhat = 0
        for c, w, r in datagen():
            if c > 0:
                vhat += w*r* c/((w - 1) * betastar + num)

        vmin = vhat + remw * rmin
        vmax = vhat + remw * rmax
        vhat += remw * (rmin + rmax) / 2.0

    return vhat, {
            'betastar': betastar,
            'vmin': vmin,
            'vmax': vmax,
            'num': num,
            'qfunc': lambda c, w, r: c / (num + betastar * (w - 1)),
           }
