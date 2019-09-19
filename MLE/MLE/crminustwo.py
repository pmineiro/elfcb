class CrMinusTwo:
    @staticmethod
    def estimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, **kwargs):
        n, sumw, sumwsq, sumwr, sumwsqr = 0, 0, 0, 0, 0
        for c, w, r in datagen():
            n += c
            sumw += c*w
            sumwsq += c*w*w
            sumwr += c*w*r
            sumwsqr += c*w*w*r
        assert n > 0

        wfake = wmax if sumw < n else wmin

        a = (wfake + sumw) / (1 + n)
        b = (wfake**2 + sumwsq) / (1 + n)
        assert a*a < b
        gammastar = (b - a) / (a*a - b)
        betastar = (1 - a) / (a*a - b)
        gstar = (n + 1) * (a - 1)**2 / (b - a*a)
        vhat = (-gammastar * sumwr - betastar * sumwsqr) / (1 + n)
        missing = (-gammastar * wfake - betastar * wfake**2) / (1 + n)

        vmin = max(rmin, min(rmax, vhat + missing * rmin))
        vmax = max(rmin, min(rmax, vhat + missing * rmax))
        vhat += missing * (rmin + rmax) / 2.0
        vhat = max(rmin, min(rmax, vhat))

        return vhat, {
            'primal': gstar,
            'barw': sumw / n,
            'gammastar': gammastar,
            'betastar': betastar,
            'vmin': vmin,
            'vmax': vmax,
            'num': n,
            'qfunc': lambda c, w, r: (c/(1 + n)) * w * (-gammastar - betastar * w),
        }

    @staticmethod
    def interval(datagen, wmin, wmax, alpha=0.05,
                 rmin=0, rmax=1, raiseonerr=False):
        from scipy.stats import f

        assert wmin < 1
        assert wmax > 1
        assert rmin <= rmax

        n, sumw, sumwsq, sumwr, sumwsqr, sumwsqrsq = 0, 0, 0, 0, 0, 0

        for c, w, r in datagen():
            n += c
            sumw += c * w
            sumwsq += c * w**2
            sumwr += c * w * r
            sumwsqr += c * w**2 * r
            sumwsqrsq += c * w**2 * r**2
        assert n > 0

        uncwfake = wmax if sumw < n else wmin
        unca = (uncwfake + sumw) / (1 + n)
        uncb = (uncwfake**2 + sumwsq) / (1 + n)
        uncgstar = (n + 1) * (unca - 1)**2 / (uncb - unca*unca)
        Delta = f.isf(q=alpha, dfn=1, dfd=n-1)
        phi = (-uncgstar - Delta) / (2 * (n + 1))

        bounds = []
        for r, sign in ((rmin, 1), (rmax, -1)):
            candidates = []
            for wfake in (wmin, wmax):
                barw = (wfake + sumw) / (1 + n)
                barwsq = (wfake*wfake + sumwsq) / (1 + n)
                barwr = sign * (wfake * r + sumwr) / (1 + n)
                barwsqr = sign * (wfake * wfake * r + sumwsqr) / (1 + n)
                barwsqrsq = (wfake * wfake * r * r + sumwsqrsq) / (1 + n)

                if barwsq > barw**2:
                    from math import isclose

                    x = barwr + ((1 - barw) * (barwsqr - barw * barwr) / (barwsq - barw**2))
                    y = (barwsqr - barw * barwr)**2 / (barwsq - barw**2) - (barwsqrsq - barwr**2)
                    z = phi + (1/2) * (1 - barw)**2 / (barwsq - barw**2)
                    if isclose(y*z, 0, abs_tol=1e-9):
                        y = 0

                    if z <= 0 and y * z >= 0:
                        from math import sqrt
                        gstar = x - sqrt(2 * y * z)
                        kappa = sqrt(y / (2 * z)) if z < 0 else 0
                        beta = (-kappa * (1 - barw) - (barwsqr - barw * barwr)) / (barwsq - barw*barw)
                        gamma = -kappa - beta * barw - barwr

                        candidates.append((gstar, None if isclose(kappa, 0) else {
                            'kappastar': kappa,
                            'betastar': beta,
                            'gammastar': gamma,
                            'wfake': wfake,
                        # Q_{w,r} &= -\frac{\gamma + \beta w + w r}{(N+1) \kappa} \\
                            'qfunc': lambda c, w, r, k=kappa, g=gamma, b=beta, s=sign, num=n: -(g + (b + s * r) * w) / ((num + 1) * kappa),
                        }))

            best = min(candidates, key=lambda x: x[0])
            vbound = min(rmax, max(rmin, sign*best[0]))
            bounds.append((vbound, best[1]))

        return (bounds[0][0], bounds[1][0]), (bounds[0][1], bounds[1][1])
