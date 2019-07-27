import numpy
from . import ControlledRangeVariance

class DoubleDouble:
    def __init__(self, numactions, seed, wsupport, expwsq):
        self.numactions = numactions
        self.state = numpy.random.RandomState(seed+1)
        self.env = ControlledRangeVariance.ControlledRangeVariance(seed=seed, wsupport=wsupport, expwsq=expwsq)
        
    def range(self):
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
                       w - 1 if a == pia else 0 for a in range(self.numactions)
                       if a > 0
            )
            nicedata.update({ (w, r, cvs): c})

        return (truevalue, [ (c, w, r, numpy.array(cv)) for (w, r, cv), c in nicedata.items() ])
