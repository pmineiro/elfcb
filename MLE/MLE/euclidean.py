class Euclidean:
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
