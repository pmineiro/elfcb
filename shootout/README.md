# Shootout

## CI shootout

 * ```make ciorig40``` will produce the first column of results from figure 3.  Should eventually produce:
 ```console
(elfcb) pmineiro@PMINEIRO-52% make ciorig40
./do-ci-shootout.py --dirname orig40
('EpsilonGreedy 0.05', Counter({'tie': 19, 'mle': 16, 'base': 5}))
('EpsilonGreedy 0.1', Counter({'tie': 23, 'mle': 14, 'base': 3}))
('EpsilonGreedy 0.25', Counter({'tie': 26, 'mle': 10, 'base': 4}))
('Bag 10', Counter({'tie': 30, 'mle': 7, 'base': 3}))
('Bag 32', Counter({'tie': 28, 'mle': 10, 'base': 2}))
('Cover 10', Counter({'tie': 28, 'mle': 8, 'base': 4}))
('Cover 32', Counter({'tie': 25, 'mle': 10, 'base': 5}))
```
