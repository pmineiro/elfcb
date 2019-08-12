#! /usr/bin/env python3

class ClippedIPS:
    @staticmethod
    def estimate(data, **kwargs):
        import numpy as np
        n = sum(c for c, _, _ in data)
        return 0.5 if n == 0 else np.clip(sum(c*w*r for c, w, r in data) / n, a_min=0, a_max=1)

class SNIPS:
    @staticmethod
    def estimate(data, **kwargs):
        effn = sum(c*w for c, w, _ in data)
        return 0.5 if effn == 0 else sum(c*w*r for c, w, r in data) / effn

class Dataset(object):
    def __init__(self, lineseed, path):
        import gzip
        import numpy

        state = numpy.random.RandomState(lineseed)

        self.path = path
        with gzip.open(self.path, 'rb') as f:
            data = f.read().decode("ascii")
            self.lines = data.split('\n')
            self.lines = [ self.lines[n] for n in state.permutation(len(self.lines))[:10000] ]

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return len(self.lines)

    def __next__(self):
        try:
            result = self.lines[self.i]
        except IndexError:
            raise StopIteration
        self.i += 1
        return result

    def metadata(self):
        import re

        m = re.search('ds_(\d+)_(\d+).vw.gz', self.path)
        if m is None:
            return { 'id': None, 'numclasses': 0 }
        return { 'id': int(m[1]), 'numclasses': int(m[2]) }

    def numclasses(self):
        return self.metadata()['numclasses']

    def splits(self):
        n = len(self.lines)
        return n // 5, 4 * n // 5, n

def all_data_files(dirname):
    import glob
    import os

    return list(sorted(glob.glob(os.path.join(dirname, '*.vw.gz'))))

def make_historical_policy(data, actionseed, vwextraargs):
    from vowpalwabbit import pyvw
    import numpy
    import os
    import pylibvw

    numclasses = data.numclasses()
    assert numclasses > 0
    vwargs = '--quiet -f damodel{} -b 20 --cb_type dr --cb_explore {} {}'.format(
                 os.getpid(), numclasses, vwextraargs
    )
    vw = pyvw.vw(vwargs)
    learnlog, _, _ = data.splits()

    state = numpy.random.RandomState(actionseed)

    for _ in range(1):
        for i, line in enumerate(data, 1):
            if "|" not in line: continue

            mclabel, rest = line.split(' ', 1)
            label = int(mclabel)

            ex = vw.example(rest)
            probs = numpy.array(vw.predict(ex, prediction_type=pylibvw.vw.pACTION_PROBS))
            probs = probs / numpy.sum(probs)
            del ex

            action = state.choice(numclasses, p=probs)
            cost = 0 if 1+action == label else 1

            extext = '{}:{}:{} {}'.format(1+action, cost, probs[action], rest)

            ex = vw.example(extext)
            vw.learn(ex)
            del ex

            if i >= learnlog:
                break

    del vw

    return vwargs

def importance_weighted_learn(vw, action, cost, probability, importance, features):
    if importance > 0:
        cbex = '{}:{}:{} {}'.format(1 + action, cost, min(1, probability / importance), features)
        ex = vw.example(cbex)
        vw.learn(ex)
        del ex

def learn_pi(data, actionseed, passes):
    from vowpalwabbit import pyvw
    import numpy
    import pylibvw
    import os

    learnlog, learnpi, _ = data.splits()
    numclasses = data.numclasses()

    logvw = pyvw.vw('--quiet -i damodel{}'.format(os.getpid()))
    vw = pyvw.vw('--quiet -i damodel{} -f damodelx{}'.format(os.getpid(), os.getpid()))

    for p in range(passes):
        alldatums = []
        state = numpy.random.RandomState(actionseed)

        for i, line in enumerate(data, 1):
            if "|" not in line: continue

            if i <= learnlog:
                continue

            if i > learnpi:
                break

            mclabel, rest = line.split(' ', 1)
            label = int(mclabel)

            logex = logvw.example(rest)
            logprobs = numpy.array(logvw.predict(logex, prediction_type=pylibvw.vw.pACTION_PROBS))
            logprobs = logprobs / numpy.sum(logprobs)
            del logex

            action = state.choice(numclasses, p=logprobs)
            cost = 0 if 1+action == label else 1

            ex = vw.example(rest)
            probs = numpy.array(vw.predict(ex, prediction_type=pylibvw.vw.pACTION_PROBS))
            pred = numpy.argmax(probs)
            iw = 1 / logprobs[action] if action == pred else 0
            del ex

            importance_weighted_learn(vw, action, cost, logprobs[action], iw, rest)

    del vw
    del logvw

    return None

def eval_pi(data, actionseed):
    from collections import defaultdict
    from vowpalwabbit import pyvw
    import numpy
    import pylibvw
    import os

    counts = defaultdict(int)
    countswithcvs = defaultdict(int)
    numclasses = data.numclasses()

    state = numpy.random.RandomState(actionseed)

    logvw = pyvw.vw('--quiet -i damodel{}'.format(os.getpid()))
    vw = pyvw.vw('--quiet -i damodelx{}'.format(os.getpid()))

    learnlog, learnpi, _ = data.splits()

    truepv = 0.0
    lineno = 0

    for i, line in enumerate(data, 1):
        if "|" not in line: continue

        if i <= learnpi:
            continue

        mclabel, rest = line.split(' ', 1)
        label = int(mclabel)

        logex = logvw.example(rest)
        logprobs = numpy.array(logvw.predict(logex, prediction_type=pylibvw.vw.pACTION_PROBS))
        logprobs = logprobs / numpy.sum(logprobs)
        action = state.choice(numclasses, p=logprobs)
        del logex

        ex = vw.example(rest)
        probs = numpy.array(vw.predict(ex, prediction_type=pylibvw.vw.pACTION_PROBS))
        pred = numpy.argmax(probs)
        del ex

        truepv += 1.0 if 1+pred == label else 0.0
        lineno += 1

        iw = 1.0 / logprobs[action] if action == pred else 0.0
        obspv = 1.0 if 1+action == label else 0.0
        counts[(iw, obspv)] += 1
        countswithcvs[(iw, obspv, pred)] += 1

    del vw

    cvdata = defaultdict(int)
    for (w, r, pia), c in countswithcvs.items():
        cvs = tuple(
                   w - 1.0 if a == pia else 0.0 for a in range(numclasses)
        )
        cvdata[(w, r, cvs)] += c
    
    return ([(c, w, r) for (w, r), c in counts.items()],
            truepv/lineno,
            [(c, w, r, numpy.array(cvs)) for (w, r, cvs), c in cvdata.items()]
           )

# main

def dofile(filename, lineseed, actionseed, passes, exploration, challenger):
    try:
        import MLE.MLE
        import numpy

        data = Dataset(lineseed, filename)
        vwextraargs = exploration.getvwextraargs()
        make_historical_policy(data, actionseed, vwextraargs)
        wmax = exploration.getwmax(data.numclasses())

        def rangefn(what=None):
            wmin = 0
            numactions = data.numclasses()
            if what == 'wmin':
                return wmin
            elif what == 'wmax':
                return wmax
            else:
                def iter_func():
                    # from samplewithcvs():
                    # 1 cv is (w-1) and the rest are 0
                    for index in range(numactions):
                        for w in (wmin, wmax):
                            cvvals = numpy.zeros(numactions, dtype='float64')
                            cvvals[index] = w - 1.0
                            yield (w, cvvals)

            return iter_func()

        ips = []
        snips = []
        mle = []
        mlecv = []
        truevals = []

        # 95% for t-test with dof=60 => 2x std-dev
        # 90% for t-test with dof=5 => 2x std-dev
        for x in range(60):
            learn_pi(data, actionseed+x, passes)
            counts, truepv, countswithcvs = eval_pi(data, actionseed+x)
            ips.append(ClippedIPS.estimate(counts))
            snipsres = SNIPS.estimate(counts)
            snips.append(snipsres)
            if challenger == Challenger.MLE:
                mleres = MLE.MLE.estimate(datagen=lambda: counts, wmin=0, wmax=wmax)
                mle.append(snipsres*mleres[1]['vmax'] + (1 - snipsres)*mleres[1]['vmin'])
            elif challenger == Challenger.MLECV:
                mlecvres = MLE.MLE.estimatewithcv(datagen=lambda: countswithcvs,
                                                  rangefn=rangefn)
                mle.append(snipsres*mlecvres[1]['vmax'] + (1 - snipsres)*mlecvres[1]['vmin'])
            truevals.append(truepv)

        ips = numpy.array(ips)
        snips = numpy.array(snips)
        mle = numpy.array(mle)
        truevals = numpy.array(truevals)

        ipsvsmle = numpy.mean(numpy.square(ips - truevals)
                              - numpy.square(mle - truevals))
        ipsvsmlevar = numpy.std(numpy.square(ips - truevals)
                                - numpy.square(mle - truevals),
                                ddof=1) / numpy.sqrt(len(ips))

        ipsvsmlewinloss = (challenger.value if ipsvsmle > max(1e-8, 2*ipsvsmlevar) else
                           'ips' if ipsvsmle < min(-1e-8, -2*ipsvsmlevar) else
                           'tie')


        snipsvsmle = numpy.mean(numpy.square(snips - truevals)
                                - numpy.square(mle - truevals))
        snipsvsmlevar = numpy.std(numpy.square(snips - truevals)
                                  - numpy.square(mle - truevals),
                                  ddof=1) / numpy.sqrt(len(snips))
        snipsvsmlewinloss = (
                challenger.value if snipsvsmle > max(1e-8, 2*snipsvsmlevar) else
                'snips' if snipsvsmle < min(-1e-8, -2*snipsvsmlevar) else
                'tie')

    finally:
        try:
            import os
            os.remove("damodel{}".format(os.getpid()))
            os.remove("damodelx{}".format(os.getpid()))
        except:
            pass

    return ipsvsmlewinloss, snipsvsmlewinloss

def doit(lineseed, actionseed, passes, dirname, exploration, poolsize, challenger):
    from collections import Counter
    from multiprocessing import Pool

    # NB: maxtasksperchild avoids memory leaks
    pool = Pool(processes=poolsize, maxtasksperchild=1)
    results = []

    jobs = [ pool.apply_async(dofile, (filename,
                                       lineseed,
                                       actionseed,
                                       passes,
                                       exploration,
                                       challenger))
             for fileno, filename in enumerate(all_data_files(dirname))
# 	     if '1413_3' in filename
           ]

    for job in jobs:
        results.append(job.get())

    pool.close()
    pool.join()

    return { 'ipsvs{}'.format(challenger.value): Counter([ x[0] for x in results ]),
             'snipsvs{}'.format(challenger.value): Counter([ x[1] for x in results ]),
           }

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def getwmax(self, numclasses):
        return 1 / (self.epsilon / numclasses)

    def getvwextraargs(self):
        return "--epsilon {}".format(self.epsilon)

    def __str__(self):
        return "EpsilonGreedy {}".format(self.epsilon)

class Bag:
    def __init__(self, bags):
        self.bags = bags

    def getwmax(self, numclasses):
        return self.bags

    def getvwextraargs(self):
        return "--bag {}".format(self.bags)

    def __str__(self):
        return "Bag {}".format(self.bags)

class Cover:
    def __init__(self, policies):
        self.policies = policies

    def getwmax(self, numclasses):
        return self.policies

    def getvwextraargs(self):
        return "--cover {}".format(self.policies)

    def __str__(self):
        return "Cover {}".format(self.policies)

# main

from enum import Enum
import argparse

class Challenger(Enum):
    MLE = 'mle'
    MLECV = 'mlecv'

parser = argparse.ArgumentParser(description='run estimation shootout')
parser.add_argument('--lineseed', type=int, default=45)
parser.add_argument('--actionseed', type=int, default=2112)
parser.add_argument('--passes', type=int, default=4)
parser.add_argument('--poolsize', type=int, default=None)
parser.add_argument('--dirname', type=str, required=True)
parser.add_argument('--challenger',
                    type=Challenger,
                    choices=list(Challenger),
                    default=Challenger.MLE)
args = parser.parse_args()

for exploration in (
    EpsilonGreedy(0.05),
    EpsilonGreedy(0.1),
    EpsilonGreedy(0.25),
    Bag(10),
    Bag(32),
    Cover(10),
    Cover(32)
):
    result = doit(args.lineseed, args.actionseed, args.passes,
                  args.dirname, exploration, args.poolsize, args.challenger)

    from pprint import pformat
    print(pformat((str(exploration), result)), flush=True)
