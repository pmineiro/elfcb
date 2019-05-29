import numpy as np
import cvxpy as cp
from pprint import pformat

class ControlledRangeVariance:
    def __init__(self, seed, wsupport, expwsq, rvala=1, rvalb=1):
        wmax = max(wsupport)
        assert wmax > 1
        assert wmax > expwsq
        assert min(wsupport) < 1

        # dual of entropy maximization
        #
        # f (pw) = pw log(pw)
        # f^*(y) = exp(y - 1)
        #
        # min \sum_w pw log(pw) = \sum_w f(pw)
        # s.t.
        # 1^T pw = 1
        # w^T pw = 1
        # (w*w)^T pw = v
        #
        # => [ 1^T; w^T; (w*w)^T ] pw = [ 1; 1; v ]
        #
        # g(a, b, c) = -a - b - v*c - sum exp(-1 - a - wi*b - wi*wi*c)
        # 
        # eliminate a 
        #
        # g(b, c) = -1 - b - v*c - log sum exp(-1 - wi*b - wi*wi*c)

        self.wsupport = np.sort(np.array(wsupport))

        wnice = self.wsupport / wmax

        A = np.array([wnice, wnice * wnice]).reshape(2, -1)
        b = np.array([1 / wmax, expwsq / (wmax * wmax)])
        mu = cp.Variable(len(b))
        prob = cp.Problem(cp.Maximize(mu.T * b - cp.log_sum_exp(mu.T * A)), [])
        tol=5e-12
        prob.solve(solver='ECOS', verbose=False, max_iters=1000, feastol=tol, reltol=tol, abstol=tol)
        assert prob.status == 'optimal'
        logits = np.asarray((mu.T * A).value).ravel()

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / np.sum(e)

        self.pw = softmax(logits)

        assert np.allclose(
                self.pw.dot(np.multiply(self.wsupport, self.wsupport)),
                expwsq), pformat(
                {
                    'self.pw.dot(np.multiply(self.wsupport, self.wsupport))': self.pw.dot(np.multiply(self.wsupport, self.wsupport)),
                    'expwsq': expwsq
                }
        )

        assert np.allclose(
                self.pw.dot(self.wsupport),
                1), pformat(
                {
                    'self.pw.dot(self.wsupport)': self.pw.dot(self.wsupport),
                }
        )

        assert np.allclose(np.sum(self.pw), 1), pformat(
                {
                    'np.sum(self.pw)': np.sum(self.pw)
                }
        )

        self.rvala = rvala
        self.rvalb = rvalb
        self.state = np.random.RandomState(seed)
        self.seed = seed

    def getpw(self):
        return (self.pw, self.wsupport)

    def range(self):
        return min(self.wsupport), max(self.wsupport)

    def expectedwsq(self):
        return self.pw.dot(self.wsupport*self.wsupport)

    def sample(self, ndata, censor=None):
        truevalue = self.state.beta(a=self.rvala, b=self.rvalb)

        pw = self.pw
        wsupport = self.wsupport

        rempw = 1
        remtv = truevalue
        pr = []

        for wi, pwi in zip(wsupport[:-1], pw[:-1]):
            if wi > 0 and pwi > 0:
                rempw -= wi*pwi
                rmin = 0 if rempw > remtv else (remtv - rempw)/(wi*pwi)
                rmax = 1 if (wi*pwi) <= remtv else remtv/(wi*pwi)
                pri = self.state.uniform(rmin, rmax)
                pr.append(pri)
                remtv -= wi*pwi*pri
            else:
                pr.append(0)

        prlast = 0 if pw[-1] == 0 else remtv / (wsupport[-1]*pw[-1])
        pr.append(prlast)
        pr = np.clip(pr, a_min=0, a_max=1)

        assert np.allclose(truevalue,
                           pw.dot(wsupport*pr),
                           atol=1e-6), pformat(
            { 'truevalue': truevalue,
              'pw': pw,
              'pr': pr,
              'np.all(pr <= 1)': np.all(pr <= 1),
              'np.all(pr >= 0)': np.all(pr >= 0),
              'actualvalue': pw.dot(wsupport*pr),
            })


        counts = self.state.multinomial(n=ndata, pvals=pw)
        sufstat = []
        for prw, cw, w in zip(pr, counts, wsupport):
            if cw > 0:
                ones = self.state.binomial(cw, prw)
                zeros = cw - ones
                if zeros > 0:
                    sufstat.append( (zeros, w, 0 ) )
                if ones > 0:
                    sufstat.append( (ones, w, 1 ) )

        rv = (truevalue, sufstat)

        if censor is not None:
            from collections import defaultdict

            censorcounts = defaultdict(int)
            for c, w, r in rv[1]:
                censored = self.state.binomial(n=c, p=censor)
                notcensored = c - censored

                censorcounts[(w, r)] += notcensored
                if censored > 0:
                    censorcounts[(w, None)] += censored

            rv = (rv[0], [ (c, w, r) for (w, r), c in censorcounts.items() ])

        return rv
