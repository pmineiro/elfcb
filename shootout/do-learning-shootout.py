#! /usr/bin/env python3

class BatchDualSolver():
    def __init__(self, *args, **kwargs):
        pass

    def learn(self, data, *args, **kwargs):
        raise NotImplementedError

    def infer(self, datum, *args, **kwargs):
        raise NotImplementedError

class BaselineBatchDualSolver(BatchDualSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self, *args, **kwargs):
        pass

    def infer(self, datum, *args, **kwargs):
        return 1

class MLEBatchDualSolver(BatchDualSolver):
    def __init__(self, *args, **kwargs):
        self.wmax = kwargs.pop('wmax')
        super().__init__(*args, **kwargs)

    def learn(self, data):
        import MLE.MLE
        from collections import Counter

        counts = Counter((round(w, 5), r) for (w, r) in data)

        data = [ (c, w, r) for (w, r), c in counts.items() ]
        self.n = sum(c for c, _, _ in data)
        self.mle = MLE.MLE.estimate(datagen=lambda: data, wmin=0, wmax=self.wmax)

    def infer(self, datum):
        (w, r) = datum
        w = max(0, min(self.wmax, w))
        try:
            return self.mle[1]['qfunc'](self.n, w, r)
        except:
            return 0

class CIBatchDualSolver(BatchDualSolver):
    def __init__(self, *args, **kwargs):
        self.wmax = kwargs.pop('wmax')
        super().__init__(*args, **kwargs)

    def learn(self, data):
        import MLE.MLE
        from collections import Counter

        counts = Counter((round(w, 5), r) for (w, r) in data)
        data = [ (c, w, r) for (w, r), c in counts.items() ]
        self.n = sum(c for c, _, _ in data)

        self.ci = MLE.MLE.asymptoticconfidenceinterval(datagen=lambda: data, wmin=0, wmax=self.wmax, alpha=0.05)
        self.mle = MLE.MLE.estimate(datagen=lambda: data, wmin=0, wmax=self.wmax) if self.ci[1][0] is None else None

    def infer(self, datum):
        (w, r) = datum
        w = max(0, min(self.wmax, w))

        try:
            if self.ci[1][0] is None:
                return self.mle[1]['qfunc'](self.n, w, r)
            else:
                return self.ci[1][0]['qfunc'](self.n, w, r)
        except:
            return 0

class OnlineSolver():
    def __init__(self, *args, **kwargs):
        pass

    def update(self, datum, *args, **kwargs):
        raise NotImplementedError

    def infer(self, datum, *args, **kwargs):
        raise NotImplementedError

class BaselineOnlineSolver():
    def __init__(self, *args, **kwargs):
        pass

    def update(self, datum, *args, **kwargs):
        pass

    def infer(self, datum, *args, **kwargs):
        return 1

class CIOnlineSolver():
    def __init__(self, *args, **kwargs):
        import MLE.MLE

        wmax = kwargs.pop('wmax')
        wmin = kwargs.pop('wmin', 0)
        rmax = kwargs.pop('rmax', 1)
        rmin = kwargs.pop('rmin', 0)
        numbuckets = kwargs.pop('numbuckets', 10)
        alpha = kwargs.pop('alpha', 0.05)
        self.happrox = MLE.MLE.Online.HistApprox(wmin=wmin, wmax=wmax, numbuckets=numbuckets)
        self.onlineci = MLE.MLE.Online.CI(wmin=wmin, wmax=wmax, rmin=rmin, rmax=rmax, alpha=alpha)
        self.wmin = wmin
        self.wmax = wmax

    def update(self, datum, *args, **kwargs):
        (w, r) = datum
        w = max(self.wmin, min(self.wmax, w))
        self.happrox.update(1, w, r)
        self.onlineci.update(self.happrox.iterator)

    def infer(self, datum, *args, **kwargs):
        (w, r) = datum
        w = max(self.wmin, min(self.wmax, w))
        return self.onlineci.getqfunc(self.happrox.iterator)['qfunc'](self.onlineci.n, w, r)

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
        return n // 5, n // 5 + n // 5, n

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
    # NB: l=0.1 due to factor of 10 in importance_weighted_learn
    vwargs = '--quiet -f damodel{} -b 20 -l 0.1 --cb_type ips --cb_explore {} {}'.format(
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

def importance_weighted_learn(vw, action, cost, importance, features):
    if importance > 0:
        # NB: can't pass probabilities > 1 (vw rejects)
        #     so scale everything by 10 and lower the learning rate by 10

        cbex = '{}:{}:{} {}'.format(1 + action,
                                    cost,
                                    min(1.0, 1.0 / (10.0 * importance)),
                                    features)
        ex = vw.example(cbex)
        vw.learn(ex)
        del ex

def learn_pi(data, actionseed, solver, passes, online):
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
        stages = ('online',) if online else ('dual', 'policy')

        for stage in stages:
            state = numpy.random.RandomState(actionseed)

            if stage == 'policy':
                alldatums.reverse()
                solver.learn([ (w, r) for (w, cost, r) in alldatums])

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
                reward = 1 if 1+action == label else 0

                ex = vw.example(rest)
                probs = numpy.array(vw.predict(ex, prediction_type=pylibvw.vw.pACTION_PROBS))
                pred = numpy.argmax(probs)
                iw = 1 / logprobs[action] if action == pred else 0
                del ex

                if stage == 'dual':
                    alldatums.append((iw, cost, reward))
                elif stage == 'policy':
                    modifiediw = iw * solver.infer((iw, reward))
                    importance_weighted_learn(vw, action, cost, modifiediw, rest)
                elif stage == 'online':
                    datum = (iw, reward)
                    solver.update(datum)
                    modifiediw = iw * solver.infer(datum)
                    importance_weighted_learn(vw, action, cost, modifiediw, rest)

    del vw
    del logvw

    return None

def eval_pi(data, actionseed):
    from vowpalwabbit import pyvw
    import numpy
    import pylibvw
    import os

    state = numpy.random.RandomState(actionseed)

    vw = pyvw.vw('--quiet -i damodelx{}'.format(os.getpid()))

    learnlog, learnpi, _ = data.splits()

    truepv = 0
    lineno = 0

    for i, line in enumerate(data, 1):
        if "|" not in line: continue

        if i <= learnpi:
            continue

        mclabel, rest = line.split(' ', 1)
        label = int(mclabel)

        ex = vw.example(rest)
        probs = numpy.array(vw.predict(ex, prediction_type=pylibvw.vw.pACTION_PROBS))
        pred = numpy.argmax(probs)
        del ex

        truepv += 1 if 1+pred == label else 0
        lineno += 1

    del vw

    return truepv/lineno

# main

def dofile(filename, lineseed, actionseed, passes, challenger, exploration, online):
    try:
        import numpy

        data = Dataset(lineseed, filename)
        vwextraargs = exploration.getvwextraargs()
        make_historical_policy(data, actionseed, vwextraargs)
        wmax = exploration.getwmax(data.numclasses())

        bpvs = []
        mlepvs = []

        # 95% for t-test with dof=60 => 2x std-dev
        # 90% for t-test with dof=5 => 2x std-dev
        for x in range(60):
            baseline = BaselineOnlineSolver(wmax=wmax) if online else BaselineBatchDualSolver(wmax=wmax)
            learn_pi(data, actionseed+x, baseline, passes, online)
            bpvs.append(eval_pi(data, actionseed+x))

            mle = CIOnlineSolver(wmax=wmax) if online else challenger.solver(wmax=wmax)

            learn_pi(data, actionseed+x, mle, passes, online)
            mlepvs.append(eval_pi(data, actionseed+x))

        bpvs = numpy.array(bpvs)
        mlepvs = numpy.array(mlepvs)
        deltasmean = numpy.mean(mlepvs - bpvs)
        deltastd = numpy.std(mlepvs - bpvs, ddof=1) / numpy.sqrt(len(bpvs))

#        from pprint import pformat
#        print(pformat({
#            'bpvs': bpvs,
#            'mlepvs': mlepvs,
#            'deltasmean': deltasmean,
#            'deltastd': deltastd,
#            'filename': filename,
#            }
#        ), flush=True)

        mlewinloss = ('mle' if deltasmean > max(1e-8, 2*deltastd) else
                      'base' if deltasmean < min(-1e-8, -2*deltastd) else
                      'tie')
    finally:
        import os

        def removeit(name):
            try:
                os.remove(name)
            except:
                pass

        removeit("damodel{}".format(os.getpid()))
        removeit("damodel{}.writing".format(os.getpid()))
        removeit("damodelx{}".format(os.getpid()))
        removeit("damodelx{}.writing".format(os.getpid()))

    return mlewinloss

def doit(lineseed, actionseed, passes, dirname, challenger, exploration, online, poolsize):
    from collections import Counter
    from multiprocessing import Pool

    # NB: maxtasksperchild avoids memory leaks
    pool = Pool(processes=poolsize, maxtasksperchild=1)
    results = []

    jobs = [ pool.apply_async(dofile, (filename,
                                       lineseed,
                                       actionseed,
                                       passes,
                                       challenger,
                                       exploration,
                                       online))
             for fileno, filename in enumerate(all_data_files(dirname))
#             if fileno < 1
           ]

    for job in jobs:
        results.append(job.get())

    pool.close()
    pool.join()

    return Counter(results)

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
    CI = 'ci'
    MLE = 'mle'

    def __str__(self):
        return self.name.lower()

    def solver(self, *args, **kwargs):
        return CIBatchDualSolver(*args, **kwargs) if self == Challenger.CI else MLEBatchDualSolver(*args, **kwargs)

parser = argparse.ArgumentParser(description='run CI shootout')
parser.add_argument('--lineseed', type=int, default=45)
parser.add_argument('--actionseed', type=int, default=2112)
parser.add_argument('--passes', type=int, default=4)
parser.add_argument('--poolsize', type=int, default=None)
parser.add_argument('--dirname', type=str, required=True)
parser.add_argument('--challenger',
                    type=Challenger,
                    choices=list(Challenger),
                    default=Challenger.CI)
parser.add_argument('--online',
                    dest='online',
                    default=False,
                    action='store_true')

args = parser.parse_args()

assert args.challenger == Challenger.CI or not args.online

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
                  args.dirname, args.challenger, exploration,
                  args.online,
                  args.poolsize)

    from pprint import pformat
    print(pformat((str(exploration), result)), flush=True)
