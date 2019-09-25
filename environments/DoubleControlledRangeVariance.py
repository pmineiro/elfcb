import numpy as np
import cvxpy as cp
from pprint import pformat

class DoubleControlledRangeVariance:
    def __init__(self, seed, usupport, wsupport, expusq, expwsq, rvala=1, rvalb=1):
        umax = max(usupport)
        assert umax > 1
        assert umax > expusq
        assert min(usupport) < 1

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
        # g(b, c) = -b - v*c - log sum exp(-wi*b - wi*wi*c)

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / np.sum(e)

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

        self.pw = softmax(logits)

        self.usupport = np.sort(np.array(usupport))

        unice = self.usupport / umax

        A = np.array([unice, unice * unice]).reshape(2, -1)
        b = np.array([1 / umax, expusq / (umax * umax)])
        mu = cp.Variable(len(b))
        prob = cp.Problem(cp.Maximize(mu.T * b - cp.log_sum_exp(mu.T * A)), [])
        tol=5e-12
        prob.solve(solver='ECOS', verbose=False, max_iters=1000, feastol=tol, reltol=tol, abstol=tol)
        assert prob.status == 'optimal'
        logits = np.asarray((mu.T * A).value).ravel()

        self.pu = softmax(logits)

        self.puw = np.outer(self.pu, self.pw)

        uval = cp.Variable()
        wval = cp.Variable()
        finalr = cp.Variable((len(self.usupport), len(self.wsupport)))

        maxprob = cp.Problem(cp.Maximize(uval - wval), [
            0 <= finalr,
            finalr <= 1,
            cp.sum(unice.T * cp.multiply(self.puw, finalr)) == uval / umax,
            cp.sum(cp.multiply(self.puw, finalr) * wnice) == wval / wmax,
        ])
        maxprob.solve(solver='ECOS')
        assert maxprob.status == 'optimal'
        maxdeltav = maxprob.value
        assert maxdeltav >= 0.52, 'maxdeltav = {}'.format(maxdeltav)

        minprob = cp.Problem(cp.Minimize(uval - wval), [
            0 <= finalr,
            finalr <= 1,
            cp.sum(unice.T * cp.multiply(self.puw, finalr)) == uval / umax,
            cp.sum(cp.multiply(self.puw, finalr) * wnice) == wval / wmax,
        ])
        minprob.solve(solver='ECOS')
        assert minprob.status == 'optimal'
        mindeltav = minprob.value
        assert mindeltav <= -0.52, 'mindeltav = {}'.format(mindeltav)

        # random draw step looks like
        #
        # minimize |r-x|^2 s.t. 0 <= x <= 1 and c^\top x == DeltaV

        self.c = (
            self.puw * np.subtract.outer(self.usupport, self.wsupport)
        ).ravel()

        self.rvala = rvala
        self.rvalb = rvalb
        self.state = np.random.RandomState(seed)
        self.seed = seed

    def range(self):
        return ( (min(self.usupport), max(self.usupport)),
                 (min(self.wsupport), max(self.wsupport)) )

    def sample(self, ndata):
        from scipy.optimize import minimize

        puw = self.puw
        usupport = self.usupport
        wsupport = self.wsupport

        deltav = -0.5 + self.state.beta(a=self.rvala, b=self.rvalb)
        rawr = self.state.uniform(0, 1, size=puw.shape).ravel()

        optresult = minimize(fun=lambda x: 0.5*(x - rawr).dot(x-rawr),
                             jac=lambda x: (x - rawr),
                             x0=rawr,
                             method='slsqp',
                             bounds=[(0,1)]*len(rawr),
                             constraints=[
                                 {
                                     'type':'eq',
                                     'fun': lambda x: self.c.dot(x) - deltav,
                                     'jac': lambda x: self.c,
                                 }
                             ])
        assert optresult.success, pformat(optresult)
        pr = optresult.x.reshape((len(usupport),len(wsupport)))

        pr = np.clip(pr, a_min=0, a_max=1)

        utruevalue = np.sum((puw*pr).T.dot(usupport))
        wtruevalue = np.sum((puw*pr).dot(wsupport))

        assert np.allclose(deltav,
                           utruevalue - wtruevalue,
                           atol=1e-4), pformat(
            { 'deltav': deltav,
              'puw': puw,
              'pr': pr,
              'utruevalue': utruevalue,
              'wtruevalue': wtruevalue,
              'actualvalue': utruevalue - wtruevalue,
            })

        counts = self.state.multinomial(n=ndata,
                                        pvals=puw.ravel()
                                       ).reshape(len(usupport),
                                                 len(wsupport))
        sufstat = []
        for i, ui in enumerate(usupport):
            for j, wj in enumerate(wsupport):
                if counts[i,j] > 0:
                    ones = self.state.binomial(counts[i,j], pr[i,j])
                    zeros = counts[i,j] - ones
                    if zeros > 0:
                        sufstat.append( (zeros, ui, wj, 0) )
                    if ones > 0:
                        sufstat.append( (ones, ui, wj, 1) )

        rv = (utruevalue, wtruevalue, sufstat)

        return rv
