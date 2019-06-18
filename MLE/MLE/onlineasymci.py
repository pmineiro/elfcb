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
            blo = int(floor(b))
            bhi = int(ceil(b))
            wlo = round(self.gwinv(self.gwmin + blo * (self.gwmax - self.gwmin) / self.numbuckets), 3)
            whi = round(self.gwinv(self.gwmin + bhi * (self.gwmax - self.gwmin) / self.numbuckets), 3) 

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
        def safedenom(x):
            from math import exp

            return x if x > Online.CI.tiny else exp(Online.CI.logstar(x))

        @staticmethod
        def jaclogstar(x):
            return 1/x if x > Online.CI.tiny else (2.0 - (x/Online.CI.tiny))/Online.CI.tiny

        @staticmethod
        def hesslogstar(x):
            return -1/(x*x) if x > Online.CI.tiny else -1/(Online.CI.tiny*Online.CI.tiny)

        @staticmethod
        def dual(p, sign, betamle, delta, num, datagen):
            from math import exp

            gamma, beta = p
            logcost = -delta
    
            n = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + beta * w + sign * w * r
                    mledenom = betamle * (w - 1) + num
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))
    
            if n > 0:
                logcost /= n
    
            return -exp(logcost) + gamma / n + beta / n

        @staticmethod
        def jacdual(p, sign, betamle, delta, num, datagen):
            from math import exp
            import numpy as np

            gamma, beta = p
            logcost = -delta
            jac = np.zeros_like(p)

            n = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + beta * w + sign * w * r
                    mledenom = betamle * (w - 1) + num
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))
                    jaclogcost = c * Online.CI.jaclogstar(denom)
                    jac[0] += jaclogcost
                    jac[1] += w * jaclogcost

            if n > 0:
                logcost /= n
                jac /= n

            jac *= -exp(logcost)
            jac[0] += 1/n
            jac[1] += 1/n

            return jac

        @staticmethod
        def hessdual(p, sign, betamle, delta, num, datagen):
            from math import exp
            import numpy as np

            gamma, beta = p
            logcost = -delta
            jac = np.zeros_like(p)
            hess = np.zeros((2,2))
    
            n = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + beta * w + sign * w * r
                    mledenom = betamle * (w - 1) + num
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))
                    jaclogcost = c * Online.CI.jaclogstar(denom)
                    jac[0] += jaclogcost
                    jac[1] += w * jaclogcost
    
                    hesslogcost = c * Online.CI.hesslogstar(denom)
                    hess[0][0] += hesslogcost
                    hess[0][1] += w * hesslogcost
                    hess[1][1] += w * w * hesslogcost
    
            if n > 0:
                logcost /= n
                jac /= n
                hess /= n
    
            hess[1][0] = hess[0][1]
            hess += np.outer(jac, jac)
            hess *= -exp(logcost)
    
            return hess

        @staticmethod
        def sumstats(p, sign, betamle, delta, num, datagen):
            from math import exp

            gamma, beta = p
            logcost = -delta
    
            sumofone = 0
            sumofw = 0
            sumofwr = 0
            n = 0
            total = 0
            for c, w, r in datagen():
                if c > 0:
                    n += c
                    denom = gamma + beta * w + sign * w * r
                    mledenom = betamle * (w - 1) + num
                    logcost += c * (Online.CI.logstar(denom) - Online.CI.logstar(mledenom))
    
                    safe = Online.CI.safedenom(denom)
                    sumofone += c / safe
                    sumofw += c*w / safe
                    sumofwr += c*w*r / safe
    
            if n > 0:
                logcost /= n
    
            kappa = exp(logcost)
            sumofone *= kappa
            sumofw *= kappa
            sumofwr *= kappa
    
            return sumofone, sumofw, sumofwr, kappa

        def __init__(self, wmin, wmax, rmin, rmax, alpha=0.05):
            import numpy as np

            self.consE = np.array([
                [ 1/max(w, 1), min(w, 1) ]
                for w in (wmin, wmax)
                for r in (rmin, rmax)
            ], dtype='float64')

            self.d = np.array([ -1*min(w, 1)*r + Online.CI.tiny/max(w, 1)
                            for w in (wmin, wmax)
                            for r in (rmin, rmax)
                         ],
                         dtype='float64')

            self.duals = np.array([1.0, 0.0], dtype='float64')
            self.alpha = alpha
            self.mle = Online.MLE(wmin, wmax)
            self.n = 0
            self.stats = None

        def update(self, datagen):
            from .sqp import sqp
            from scipy.stats import f

            betastar = self.mle.update(datagen)
            self.n = sum(c for c, _, _ in datagen())

            if self.n >= 3:
                delta = f.isf(q=self.alpha, dfn=1, dfd=self.n-1)
                _, self.duals = sqp(
                        f=lambda p: Online.CI.dual(p, 1, betastar, delta, self.n, datagen),
                        gradf=lambda p: Online.CI.jacdual(p, 1, betastar, delta, self.n, datagen),
                        hessf=lambda p: Online.CI.hessdual(p, 1, betastar, delta, self.n, datagen),
                        E=self.consE,
                        d=self.d,
                        x0=self.duals,
                        strict=True,
                        maxiter=1
                )
                self.stats = Online.CI.sumstats(
                        self.duals, 
                        1, 
                        self.mle.betastar * self.mle.n,
                        delta,
                        self.n,
                        datagen
                )


        def getduals(self, datagen):
            from scipy.stats import f

            if self.stats is not None:
                gamma = self.duals[0]
                beta = self.duals[1]
                kappa = self.stats[3]
                return { 
                    'qfunc': lambda c, w, r: (
                        c * kappa / (gamma + beta * w + w * r)
                    ),
                    'extra': {
                        'gamma': self.duals[0],
                        'beta': self.duals[1],
                        'kappa': self.stats[3],
                        'sumofone': self.stats[0],
                        'sumofw': self.stats[1],
                        'sumofwr': self.stats[2],
                        'betamlestar': self.mle.betastar,
                        'ci': True,
                    },
                }
            else:
                beta = self.mle.betastar
                n = self.mle.n
                return { 
                    'qfunc': lambda c, w, r: (
                     (c / n) / (beta * (w - 1) + 1)
                    ),
                    'extra': {
                       'beta': beta,
                       'n': n,
                       'ci': False
                    },
                }

