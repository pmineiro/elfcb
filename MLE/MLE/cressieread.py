class CressieRead:
    class Estimate:
        @staticmethod
        def dualobjective(gamma, beta, datagen, n, lam):
            from math import log
            
            lampow = lam / (1 + lam)
    
            def approxpow(loga):
                z = loga * lampow
                return loga * (1 + 1/2 * z * (1 + 1/3 * z * (1 + 1/4 * z * (1 + 1/5 * z * (1 + 1/6 * z * (1 + 1/7 * z))))))
            
            dual = - gamma - beta
            if -1e-2 <= lampow and lampow <= 1e-2:
                dual += sum((c/n) * approxpow(log((gamma + beta * w))) for c, w, _ in datagen())
            else:
                dual += sum((c/n) * ((gamma + beta * w)**lampow - 1) / lampow for c, w, _ in datagen())
                
            return -dual
        
        @staticmethod
        def jacdualobjective(gamma, beta, datagen, n, lam):       
            lampow = lam / (1 + lam)
            j = [ -1, -1 ]
            for c, w, _ in datagen():
                dx = (c/n) * (gamma + beta * w)**(lampow - 1)
                j[0] += dx
                j[1] += dx * w
                
            return -j[0], -j[1]
            
        @staticmethod
        def hessdualobjective(gamma, beta, datagen, n, lam):
            lampow = lam / (1 + lam)
            h = [ 0, 0, 0 ]
            for c, w, _ in datagen():
                d2x = (c/n) * (lampow - 1) * (gamma + beta * w)**(lampow - 2)
                h[0] += d2x 
                h[1] += d2x * w
                h[2] += d2x * w * w
                
            return [ [ -h[0], -h[1] ], [ -h[1], -h[2] ] ]
    
    @staticmethod
    def estimate(datagen, lam, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, **kwargs):
        from cvxopt import matrix, solvers
        
        n = sum(c for c, _, _ in datagen())
        assert n > 0
        
        x0 = 1.0, 0.0
        
        G = matrix([ [ -1.0, -float(w)  ] for w in (wmin, wmax) ])
        h = matrix([ 0.0 for w in (wmin, wmax) ])

#        from .gradcheck import gradcheck, hesscheck
#        from numpy import array as arr
#        gradcheck(f = lambda x: CressieRead.Estimate.dualobjective(x[0], x[1], datagen, n, lam),
#                  jac = lambda x: arr(CressieRead.Estimate.jacdualobjective(x[0], x[1], datagen, n, lam)),
#                  x = x0,
#                  what='dualobjective',
#                  eps = 1e-6)
#        hesscheck(jac = lambda x: arr(CressieRead.Estimate.jacdualobjective(x[0], x[1], datagen, n, lam)),
#                  hess = lambda x: arr(CressieRead.Estimate.hessdualobjective(x[0], x[1], datagen, n, lam)),
#                  x = x0,
#                  what='jacdualobjective')

        actualwmin = wmax
        actualwmax = wmin
        for _, w, _ in datagen():
            actualwmax = max(actualwmax, w)
            actualwmin = min(actualwmin, w)
        
        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)
            if x[0] + x[1] * actualwmin <= 0 or x[0] + x[1] * actualwmax <= 0:
                return None
            f = CressieRead.Estimate.dualobjective(x[0], x[1], datagen, n, lam)
            jf = CressieRead.Estimate.jacdualobjective(x[0], x[1], datagen, n, lam)
            Df = matrix(jf).T
            if z is None: return f, Df
            hf = CressieRead.Estimate.hessdualobjective(x[0], x[1], datagen, n, lam)
            H = z[0] * matrix(hf)
            return f, Df, H
        
        soln = solvers.cp(F, G, h, options={'show_progress': False })
        fstar = -(2 * n / (1 + lam)) * (soln['primal objective'] - 1)
        (gammastar, betastar) = soln['x']
                
        estimate = sum((c/n) * w * r * (gammastar + betastar * w)**(-1 / (1 + lam)) for c, w, r in datagen())
        missing = max(0, 1 - sum((c/n) * w * 1 * (gammastar + betastar * w)**(-1 / (1 + lam)) for c, w, _ in datagen()))
        
        return max(0, min(1, estimate + 0.5 * missing)), { 
            'fstar': fstar, 
            'gammastar': gammastar, 
            'betastar': betastar,
            'n': n,
            'status': soln['status'],
        }
