# somewhat cheesy, but remarkably robust:
# 1. approximate histogram of historical counts
# 2. do one Newton step per datum

class Online:
    class HistApprox:
        def __init__(self, wmin, wmax, numbuckets):
            from collections import defaultdict
            self.wmin = wmin
            self.wmax = wmax
            self.gwmin = self.gw(wmin)
            self.gwmax = self.gw(wmax)
            self.numbuckets = numbuckets
            self.hist = defaultdict(float)

        @staticmethod
        def gw(w):
            from math import log
            return w if w <= 1 else 1 + log(w)

        @staticmethod
        def gwinv(gw):
            from math import exp
            return gw if gw <= 1 else exp(gw - 1)

        def update(self, c, w, r):
            from math import floor, ceil

            gw = self.gw(w)
            b = self.numbuckets * (gw - self.gwmin) / (self.gwmax - self.gwmin)
            blo = max(int(floor(b)), 0)
            bhi = min(int(ceil(b)), self.numbuckets)
            wlo = round(self.gwinv(self.gwmin + blo * (self.gwmax - self.gwmin) / self.numbuckets), 3)
            whi = round(self.gwinv(self.gwmin + bhi * (self.gwmax - self.gwmin) / self.numbuckets), 3)
#            from pprint import pformat
#            assert wlo >= self.wmin and whi <= self.wmax, pformat({
#                'blo': blo,
#                'bhi': bhi,
#                'numbuckets': self.numbuckets,
#                'wlo': wlo,
#                'whi': whi,
#                'wmin': self.wmin,
#                'wmax': self.wmax
#                })

            rapprox = round(100*r)

            if w <= wlo:
                self.hist[(wlo, rapprox)] += c
            elif w >= whi:
                self.hist[(whi, rapprox)] += c
            else:
                fraclo = (whi - w) / (whi - wlo)
                frachi = (w - wlo) / (whi - wlo)
                self.hist[(wlo, rapprox)] += c*fraclo
                self.hist[(whi, rapprox)] += c*frachi

        def iterator(self):
            return ((c, w, r / 100.0) for (w, r), c in self.hist.items())

    class MLE:
        def __init__(self, wmin, wmax):
            self.betastar = 0
            self.wmin = wmin
            self.wmax = wmax
            self.n = 0

        def update(self, datagen):
            wminobs = min(w for c, w, _ in datagen() if c > 0)
            wmaxobs = max(w for c, w, _ in datagen() if c > 0)
            self.n = sum(c for c, w, _ in datagen())

            if wmaxobs > 1:
                self.betastar = max(self.betastar, (-1 + 1/(self.n + 1)) / (wmaxobs - 1))
            if wminobs < 1:
                self.betastar = min(self.betastar, (1 - 1/(self.n + 1)) / (1 - wminobs))

            g = sum(-c * (w - 1)/((w - 1) * self.betastar + 1)
                    for c, w, _ in datagen()
                    if c > 0)
            H = sum(c * ((w - 1) / ((w - 1) * self.betastar + 1))**2
                    for c, w, _ in datagen()
                    if c > 0)
            self.betastar += -g / H

            self.betastar = max(self.betastar, -1 / (self.wmax - 1))
            self.betastar = min(self.betastar, 1 / (1 - self.wmin))

            return self.betastar * self.n

    class CI:
        from math import log

        tiny = 1e-5
        logtiny = log(tiny)

        @staticmethod
        def logstar(x):
            from math import log

            return log(x) if x > Online.CI.tiny else -1.5 + Online.CI.logtiny + 2.0*(x/Online.CI.tiny) - 0.5*(x/Online.CI.tiny)*(x/Online.CI.tiny)

        @staticmethod
        def jaclogstar(x):
            return 1/x if x > Online.CI.tiny else (2.0 - (x/Online.CI.tiny))/Online.CI.tiny

        @staticmethod
        def hesslogstar(x):
            return -1/(x*x) if x > Online.CI.tiny else -1/(Online.CI.tiny*Online.CI.tiny)

        @staticmethod
        def dual(p, sign, betamle, Delta, num, wscale, rscale, datagen):
            from math import exp

            gamma, beta = p
            logcost = -Delta

            n = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                    mledenom = num + betamle * (w - 1)
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))

            assert n == num

            if n > 0:
                logcost /= n

            return (-n * exp(logcost) + gamma + beta / wscale) / rscale

        @staticmethod
        def jacdual(p, sign, betamle, Delta, num, wscale, rscale, datagen):
            from math import exp
            import numpy as np

            gamma, beta = p
            logcost = -Delta
            jac = np.zeros_like(p)

            n = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                    mledenom = num + betamle * (w - 1)
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))

                    jaclogcost = c * Online.CI.jaclogstar(denom)
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

        @staticmethod
        def hessdual(p, sign, betamle, Delta, num, wscale, rscale, datagen):
            from math import exp
            import numpy as np

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
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))

                    jaclogcost = c * Online.CI.jaclogstar(denom)
                    jac[0] += jaclogcost
                    jac[1] += jaclogcost * (w / wscale)

                    hesslogcost = c * Online.CI.hesslogstar(denom)
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

        def __init__(self, wmin, wmax, rmin, rmax, alpha=0.05):
            import numpy as np

            self.wmin = wmin
            self.wmax = wmax
            self.rmin = rmin
            self.rmax = rmax
            self.duals = None
            self.alpha = alpha
            self.mle = Online.MLE(wmin, wmax)
            self.n = 0
            self.CI = None

        def update(self, datagen):
            from .sqp import sqp
            from scipy.stats import f
            import numpy as np

            betastar = self.mle.update(datagen)
            self.n = sum(c for c, _, _ in datagen())

            if self.n >= 3:
                if self.duals is None:
                    self.duals = np.array([self.n, 0.0], dtype='float64')

                sumwsq = sum(c * w * w for c, w, _ in datagen())
                wscale = max(1.0, np.sqrt(sumwsq / self.n))
                rscale = max(1.0, np.abs(self.rmin), np.abs(self.rmax))

                consE = np.array([
                    [ 1, w / wscale ]
                    for w in (self.wmin, self.wmax)
                    for r in (self.rmin, self.rmax)
                ], dtype='float64')

                sign = 1
                d = np.array([ -sign*w*r + Online.CI.tiny
                               for w in (self.wmin, self.wmax)
                               for r in (self.rmin, self.rmax)
                             ],
                             dtype='float64')

                Delta = f.isf(q=self.alpha, dfn=1, dfd=self.n-1)

                fstar, self.duals = sqp(
                        f=lambda p: Online.CI.dual(p, 1, betastar, Delta, self.n, wscale, rscale, datagen),
                        gradf=lambda p: Online.CI.jacdual(p, 1, betastar, Delta, self.n, wscale, rscale, datagen),
                        hessf=lambda p: Online.CI.hessdual(p, 1, betastar, Delta, self.n, wscale, rscale, datagen),
                        E=consE,
                        d=d,
                        x0=[self.duals[0], self.duals[1] * wscale],
                        strict=True,
                        abscondfac=1e-3,
                        maxiter=1
                )
                self.duals[1] /= wscale

                gammastar = self.duals[0]
                betastar = self.duals[1] 
                kappastar = (-rscale * fstar + gammastar + betastar) / self.n
                vbound = -sign * rscale * fstar

                qfunc = lambda c, w, r, kappa=kappastar, gamma=gammastar, beta=betastar, s=sign: kappa * c / (gamma + (beta + s * r) * w)

                self.CI = {
                            'gammastar': gammastar,
                            'betastar': betastar,
                            'kappastar': kappastar,
                            'vbound': vbound,
                            'qfunc': qfunc,
                            'ci': True
                }

        def getqfunc(self, datagen):
            assert self.n > 0

            if self.CI is not None:
                return self.CI
            else:
                beta = self.mle.betastar
                n = self.mle.n
                qfunc = lambda c, w, r: ((c / n) / (beta * (w - 1) + 1))

                return { 
                    'qfunc': qfunc,
                    'betastar': beta,
                    'ci': False
                }
