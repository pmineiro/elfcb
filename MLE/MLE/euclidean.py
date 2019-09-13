class Euclidean:
    @staticmethod
    def cvxmegestimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, **kwargs):
        from cvxopt import matrix, solvers

        x0 = 0.0, 0.0
        n = sum(c for c, _, _ in datagen())

        G = matrix([ [ -1.0, -float(w)  ] for w in (wmin, wmax) ])
        h = matrix([ 0.0 for w in (wmin, wmax) ])

        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)
            gamma, beta = x
            f = - gamma - beta
            jf = [ -1.0, -1.0 ]
            hf = [ 0.0, 0.0, 0.0 ]
            for c, w, _ in datagen():
                rawq = (1/n) * (1 - (gamma + beta * w) / n)
                q = max(0, rawq)
                f += c * ((gamma + beta * w) * q + 0.5 * (n*q - 1)**2)
                dq = 0 if rawq <= 0 else -1/(n*n)
                dfdq = (gamma + beta * w) + n*(n*q-1)
                jf[0] += c * (q + dq * dfdq)
                jf[1] += c * w * (q + dq * dfdq)
                hf[0] += c * dq * (2 + n*n*dq)
                hf[1] += c * w * dq * (2 + n*n*dq)
                hf[2] += c * w * w * dq * (2 + n*n*dq)

            Df = matrix(jf).T
            if z is None: return -f, -Df
            H = z[0] * matrix([ [ hf[0], hf[1] ], [ hf[1], hf[2] ] ])
            return -f, -Df, -H
            
        soln = solvers.cp(F=F, G=G.T, h=h, options={'show_progress':False})
        fstar = -soln['primal objective']
        (gammastar, betastar) = soln['x']

        vhat = 0
        sumofw = 0
        sumofone = 0
        for c, w, r in datagen():
            q = (c/n) * max(0, (1 - (gammastar + betastar * w) / n))
            vhat += q * w * r
            sumofw += q * w
            sumofone += q

        remw = max(0, 1 - sumofw)

        vmin = max(rmin, min(rmax, vhat + remw * rmin))
        vmax = max(rmin, min(rmax, vhat + remw * rmax))
        vhat += remw * (rmin + rmax) / 2.0
        vhat = max(rmin, min(rmax, vhat))

        return vhat, {
            'primal': fstar,
            'gammastar': gammastar,
            'betastar': betastar,
            'vmin': vmin,
            'vmax': vmax,
            'sumofone': sumofone,
            'sumofw': sumofw,
            'num': n,
            'qfunc': lambda c, w, r: max(0, (c/n) * (1 - (gamma + beta * w) / n)),
        }


    @staticmethod
    def cvxestimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, **kwargs):
        from cvxopt import matrix, solvers

        x0 = 0.0, 0.0
        n = sum(c for c, _, _ in datagen())

        G = matrix([ [ -1.0, -float(w)  ] for w in (wmin, wmax) ])
        h = matrix([ 0.0 for w in (wmin, wmax) ])

        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)
            gamma, beta = x
            f = -beta
            jf = [ 0.0, -1.0 ]
            hf = [ -1.0, 0.0, 0.0 ]
            for c, w, _ in datagen():
                f += (c/n) * (beta * w - (gamma + beta * w)**2 / (2*n))
                jf[0] -= (c/n) * (gamma + beta * w) / n
                jf[1] += (c/n) * w * (1 - (gamma + beta * w) / n)
                hf[1] -= (c/n) * w
                hf[2] -= (c/n) * w * w

            Df = matrix(jf).T
            if z is None: return -f, -Df
            H = (z[0] / n) * matrix([ [ hf[0], hf[1] ], [ hf[1], hf[2] ] ])
            return -f, -Df, -H
            
        soln = solvers.cp(F=F, G=G.T, h=h, options={'show_progress':False})
        fstar = -soln['primal objective']
        (gammastar, betastar) = soln['x']

        vhat = 0
        sumofw = 0
        sumofone = 0
        for c, w, r in datagen():
            q = (c/n) * (1 - (gammastar + betastar * w) / n)
            vhat += q * w * r
            sumofw += q * w
            sumofone += q

        remw = max(0, 1 - sumofw)

        vmin = max(rmin, min(rmax, vhat + remw * rmin))
        vmax = max(rmin, min(rmax, vhat + remw * rmax))
        vhat += remw * (rmin + rmax) / 2.0
        vhat = max(rmin, min(rmax, vhat))

        return vhat, {
            'primal': fstar,
            'gammastar': gammastar,
            'betastar': betastar,
            'vmin': vmin,
            'vmax': vmax,
            'sumofone': sumofone,
            'sumofw': sumofw,
            'num': n,
            'qfunc': lambda c, w, r: max(0, (c/n) * (1 - (gamma + beta * w) / n)),
        }

    @staticmethod
    def estimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, **kwargs):
        n, barw, barwsq, barwr, barwsqr = 0, 0, 0, 0, 0
        for c, w, r in datagen():
            n += c
            barw += c*w
            barwsq += c*w*w
            barwr += c*w*r
            barwsqr += c*w*w*r

        assert n > 0
        barw /= n
        barwsq /= n
        barwr /= n
        barwsqr /= n

        wextreme = wmin if barw > 1 else wmax
        denom = barwsq - 2 * wextreme * barw + wextreme * wextreme

        betastarovern = (barw - 1) / denom
        gammastarovern = -betastarovern * wextreme
        vhat = barwr - gammastarovern * barwr - betastarovern * barwsqr
        sumofone = 1 - betastarovern * barw - gammastarovern
        sumofw = barw - gammastarovern * barw - betastarovern * barwsq
        remw = max(0, 1 - sumofw)

        vmin = max(rmin, min(rmax, vhat + remw * rmin))
        vmax = max(rmin, min(rmax, vhat + remw * rmax))
        vhat += remw * (rmin + rmax) / 2.0
        vhat = max(rmin, min(rmax, vhat))

        return vhat, {
            'primal': n/2 * betastarovern * (barw - 1),
            'barw': barw,
            'gammastar': n*gammastarovern,
            'betastar': n*betastarovern,
            'vmin': vmin,
            'vmax': vmax,
            'sumofone': sumofone,
            'sumofw': sumofw,
            'num': n,
            'qfunc': lambda c, w, r: (c/n) * w * (1 - betastarovern*w - gammastarovern),
        }
