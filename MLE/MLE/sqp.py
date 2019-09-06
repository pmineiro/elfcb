def sqp(f, gradf, hessf, E, d, x0,
        gtol=1e-14, ftol=1e-14, xtol=1e-14, feastol=1e-6,
        condfac=1e-6, abscondfac=0, strict=False, maxiter=10000):
    import numpy
#    from cvxopt import matrix, solvers
    from quadprog import solve_qp
    from scipy.linalg import null_space
    
    x = numpy.array(x0, dtype='float64')
    violation = d - E.dot(x)
    active = numpy.nonzero(violation >= feastol)[0]
    
    # find feasible point
    if active.size > 0:
        EA = E[active, :]
        A = numpy.block([
            [numpy.eye(len(x)), EA.T],
            [EA, numpy.zeros((EA.shape[0], EA.shape[0]))]
        ])
        b = numpy.hstack((numpy.zeros_like(x), d[active] - EA.dot(x)))
        dxlmult = numpy.linalg.lstsq(A, b, rcond=None)[0]
        dx = dxlmult[0:len(x)]
        #lmult = dxlmult[len(x):]
        x += dx
    
    fx = f(x)
    for iter in range(maxiter):
        c = gradf(x)
        violation = d - E.dot(x)
        active = numpy.nonzero(violation >= feastol)[0]

        if active.size > 0:
            ns = null_space(E[active, :])
            nullc = ns.T.dot(c)
        else:
            nullc = c

        if numpy.linalg.norm(nullc) < gtol:
            break

        try:
            Q = hessf(x)
            eigs = numpy.linalg.eigvalsh(Q)
            mineig = numpy.min(eigs)
            maxeig = numpy.max(eigs)
            targetmineig = max(abscondfac, condfac*maxeig)
            perturb = max(0, targetmineig - mineig)
            dx = solve_qp(Q + perturb*numpy.eye(Q.shape[0]),
                          -c,
                          E.T,
                          violation)[0]
#            soln = solvers.qp(matrix(Q), matrix(c), -matrix(E), -matrix(violation), kktsolver='ldl', options={'kktreg': condfac, 'show_progress': False})
#            dx = numpy.reshape(numpy.array(soln['x']), -1)
        except:
            from pprint import pformat
            print(pformat({ 'eigh(Q)': numpy.linalg.eigh(Q),
                            'Q': Q,
                            'c': c,
                            'x': x}))
            raise

        if strict:
            # E.dot(x+dx) - d >= 0
            # d - E.dot(x) - E.dot(dx) <= 0
            # violation - E.dot(dx) <= 0

            # violation - alpha * E.dot(dx) == 0
            # alpha = violation / E.dot(dx)

            Edotdx = E.dot(dx)
            badind = numpy.nonzero(numpy.logical_and(violation < 0,
                                                     violation > Edotdx))
            if numpy.any(badind):
                alpha = numpy.min(violation[badind] / Edotdx[badind])
                dx *= alpha

        if numpy.linalg.norm(dx) < xtol:
            break

        fxnew = f(x + dx)
        while fxnew > fx and numpy.linalg.norm(dx) >= xtol:
            dx /= 2
            fxnew = f(x + dx)  

        if fxnew > fx - ftol or numpy.linalg.norm(dx) < xtol:
            break
            
        fx = fxnew
        x += dx
        
    return fx, x
