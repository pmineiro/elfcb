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
