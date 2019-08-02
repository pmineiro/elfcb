import numpy
from . import ControlledRangeVariance

class DoubleDouble:
    def __init__(self, numactions, seed, wsupport, expwsq):
        self.numactions = numactions
        self.state = numpy.random.RandomState(seed+1)
        self.env = ControlledRangeVariance.ControlledRangeVariance(seed=seed, wsupport=wsupport, expwsq=expwsq)
        # NB: range() and rawsample() assume pi is deterministic
        assert wsupport[0] == 0
        
    def range(self, what=None):
        (wmin, wmax) = self.env.range()
        if what == 'wmin':
            return wmin
        elif what == 'wmax':
            return wmax
        else:
            def iter_func():
                # from samplewithcvs():
                # 1 cv is (w-1) and the rest are 0
                for index in range(self.numactions):
                    for w in (wmin, wmax):
                        cvvals = numpy.zeros(self.numactions, dtype='float64')
                        cvvals[index] = w - 1.0
                        yield (w, cvvals)

            return iter_func()

        (wmin, wmax) = self.env.range()
        return wmin, wmax, (wmin-1)*numpy.ones(self.numactions-1), (wmax-1)*numpy.ones(self.numactions-1)
        
    def rawsample(self, ndata):
        from collections import Counter
        (truevalue, data) = self.env.sample(ndata)
        
        nicedata = Counter()
        for c, w, r in data:
            actioncounts = Counter(self.state.choice(a=self.numactions, p=None, size=c))
            for pia, ca in actioncounts.items():
                nicedata.update({ (w, r, pia): ca })
            
        return (truevalue, [ (c, w, r, pia) for (w, r, pia), c in nicedata.items() ])
            
    def sample(self, ndata):
        from collections import Counter

        (truevalue, data) = self.rawsample(ndata)
        nicedata = Counter()
        for c, w, r, pia in data:
            nicedata.update({ (w, r): c })
        return (truevalue, [ (c, w, r) for (w, r), c in nicedata.items() ])
    
    def samplewithcvs(self, ndata):
        from collections import Counter
        import numpy

        (truevalue, data) = self.rawsample(ndata)
   
        nicedata = Counter()
    
        for c, w, r, pia in data:
            cvs = tuple(
                       w - 1.0 if a == pia else 0.0 for a in range(self.numactions)
            )
            nicedata.update({ (w, r, cvs): c})

        return (truevalue, [ (c, w, r, numpy.array(cv)) for (w, r, cv), c in nicedata.items() ])
