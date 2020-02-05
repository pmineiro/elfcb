# Shootout

Results from the paper comparing empirical likelihood to baselines for estimation and learning.

## Estimation

* ```make estimationshootout``` will (eventually) produce the results in Tables 1 and 4.
```console
(elfcb) % make estimationshootout
./do-estimation-shootout.py --dirname orig40
eval ./do-estimation-shootout.py --dirname orig40
('EpsilonGreedy 0.05',
 {'ipsvsmle': Counter({'mle': 26, 'tie': 11, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 34, 'mle': 5, 'snips': 1})})
('EpsilonGreedy 0.1',
 {'ipsvsmle': Counter({'mle': 27, 'tie': 10, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 7})})
('EpsilonGreedy 0.25',
 {'ipsvsmle': Counter({'mle': 28, 'tie': 9, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 37, 'snips': 2, 'mle': 1})})
('Bag 10',
 {'ipsvsmle': Counter({'tie': 19, 'mle': 13, 'ips': 8}),
  'snipsvsmle': Counter({'tie': 30, 'mle': 7, 'snips': 3})})
('Bag 32',
 {'ipsvsmle': Counter({'mle': 22, 'tie': 10, 'ips': 8}),
  'snipsvsmle': Counter({'tie': 34, 'mle': 4, 'snips': 2})})
('Cover 10',
 {'ipsvsmle': Counter({'tie': 16, 'mle': 15, 'ips': 9}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 7})})
('Cover 32',
 {'ipsvsmle': Counter({'ips': 16, 'tie': 13, 'mle': 11}),
  'snipsvsmle': Counter({'tie': 29, 'mle': 6, 'snips': 5})})
```

## Learning

 * ```make learningshootoutorig40``` will (eventually) produce the first column of results from Figure 3.
 ```console
(elfcb) % make learningshootoutorig40
eval ./do-learning-shootout.py --dirname orig40
('EpsilonGreedy 0.05', Counter({'tie': 18, 'ci': 16, 'base': 6}))
('EpsilonGreedy 0.1', Counter({'tie': 19, 'ci': 16, 'base': 5}))
('EpsilonGreedy 0.25', Counter({'tie': 22, 'ci': 15, 'base': 3}))
('Bag 10', Counter({'ci': 21, 'tie': 18, 'base': 1}))
('Bag 32', Counter({'tie': 26, 'ci': 10, 'base': 4}))
('Cover 10', Counter({'tie': 21, 'ci': 18, 'base': 1}))
('Cover 32', Counter({'tie': 29, 'ci': 9, 'base': 2}))
```

* ```make learningshootoutmleorig40``` will (eventually) produce the second column of results from Figure 3.
```console
(elfcb) % make learningshootoutmleorig40
eval ./do-learning-shootout.py --dirname orig40 --challenger mle
('EpsilonGreedy 0.05', Counter({'tie': 26, 'mle': 11, 'base': 3}))
('EpsilonGreedy 0.1', Counter({'tie': 24, 'mle': 13, 'base': 3}))
('EpsilonGreedy 0.25', Counter({'tie': 34, 'mle': 3, 'base': 3}))
('Bag 10', Counter({'tie': 28, 'mle': 11, 'base': 1}))
('Bag 32', Counter({'tie': 31, 'mle': 7, 'base': 2}))
('Cover 10', Counter({'tie': 30, 'mle': 6, 'base': 4}))
('Cover 32', Counter({'tie': 34, 'mle': 6}))
```
