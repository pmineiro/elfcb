def gradcheck(f, jac, x, what, eps=1e-8):
    import numpy as np
    import sys
    from pprint import pformat

    fx = f(x)
    jacx = jac(x)
    dx = np.zeros_like(x, dtype='double')
    dfx = np.zeros_like(x, dtype='double')

    for i in range(len(x)):
        dx[i] = eps*max([1, np.abs(x[i])])
        fdx = f(x + dx) - f(x - dx)
        dfx[i] = fdx/(2*dx[i])
        dx[i] = 0

    if not np.allclose(jacx, dfx, atol=1e-4):
        print(pformat(
            {
                'x': x,
                'fx': fx,
                'jacx': jacx,
                'dfx': dfx,
                'dx': eps*np.maximum(1, np.abs(x)),
                'what': what,
            }), file=sys.stderr)

def hesscheck(jac, hess, x, what):
    for i in range(len(x)):
        gradcheck(f=lambda z: jac(z)[i],
                  jac=lambda z: hess(z)[i,:],
                  x=x,
                  what='hesscheck {} coordinate {}'.format(what, i))

