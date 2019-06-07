# Shootout

## Estimation

* ```make estimationshootout``` will (eventually) produce the results in Tables 1 and 4.
```console
(elfcb) pmineiro@PMINEIRO-253% make estimationshootout 
./do-estimation-shootout.py --dirname orig40
('EpsilonGreedy 0.05',
 {'ipsvsmle': Counter({'mle': 26, 'tie': 11, 'base': 3}),
  'snipsvsmle': Counter({'tie': 29, 'mle': 10, 'base': 1})})
('EpsilonGreedy 0.1',
 {'ipsvsmle': Counter({'mle': 24, 'tie': 13, 'base': 3}),
  'snipsvsmle': Counter({'tie': 35, 'mle': 5})})
('EpsilonGreedy 0.25',
 {'ipsvsmle': Counter({'mle': 27, 'tie': 10, 'base': 3}),
  'snipsvsmle': Counter({'tie': 38, 'mle': 2})})
('Bag 10',
 {'ipsvsmle': Counter({'tie': 20, 'mle': 11, 'base': 9}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 5, 'base': 2})})
('Bag 32',
 {'ipsvsmle': Counter({'mle': 18, 'tie': 15, 'base': 7}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 5, 'base': 2})})
('Cover 10',
 {'ipsvsmle': Counter({'tie': 15, 'mle': 14, 'base': 11}),
  'snipsvsmle': Counter({'tie': 32, 'mle': 8})})
('Cover 32',
 {'ipsvsmle': Counter({'base': 16, 'tie': 13, 'mle': 11}),
  'snipsvsmle': Counter({'tie': 29, 'base': 7, 'mle': 4})})
```

## Learning

 * ```make learningshootoutorig40``` will (eventually) produce the first column of results from Figure 3.
 ```console
(elfcb) pmineiro@PMINEIRO-52% make learningshootoutorig40
./do-learning-shootout.py --dirname orig40
('EpsilonGreedy 0.05', Counter({'tie': 19, 'mle': 16, 'base': 5}))
('EpsilonGreedy 0.1', Counter({'tie': 23, 'mle': 14, 'base': 3}))
('EpsilonGreedy 0.25', Counter({'tie': 26, 'mle': 10, 'base': 4}))
('Bag 10', Counter({'tie': 30, 'mle': 7, 'base': 3}))
('Bag 32', Counter({'tie': 28, 'mle': 10, 'base': 2}))
('Cover 10', Counter({'tie': 28, 'mle': 8, 'base': 4}))
('Cover 32', Counter({'tie': 25, 'mle': 10, 'base': 5}))
```
 * ```make learningshootoutgt10class``` will (eventually) produce the second column of results from Figure 3.
 ```console
 (elfcb) pmineiro@PMINEIRO-207% make learningshootoutgt10class
./do-learning-shootout.py --dirname gt10class
('EpsilonGreedy 0.05', Counter({'mle': 29, 'tie': 10, 'base': 1}))
('EpsilonGreedy 0.1', Counter({'mle': 28, 'tie': 10, 'base': 2}))
('EpsilonGreedy 0.25', Counter({'mle': 31, 'tie': 8, 'base': 1}))
('Bag 10', Counter({'mle': 22, 'tie': 16, 'base': 2}))
('Bag 32', Counter({'mle': 23, 'tie': 16, 'base': 1}))
('Cover 10', Counter({'tie': 21, 'mle': 16, 'base': 3}))
('Cover 32', Counter({'tie': 28, 'mle': 8, 'base': 4}))
 ```
 
 * ```make learningshootoutmlegt10class``` will (eventually) produce the first column of results from Table 5.
 ```console
 (elfcb) pmineiro@PMINEIRO-209% make -learningshootoutmlegt10class
./do-learning-shootout.py --dirname gt10class --challenger mle
('EpsilonGreedy 0.05', Counter({'tie': 25, 'mle': 9, 'base': 6}))
('EpsilonGreedy 0.1', Counter({'tie': 29, 'base': 7, 'mle': 4}))
('EpsilonGreedy 0.25', Counter({'tie': 26, 'base': 10, 'mle': 4}))
('Bag 10', Counter({'tie': 31, 'mle': 5, 'base': 4}))
('Bag 32', Counter({'tie': 30, 'base': 7, 'mle': 3}))
('Cover 10', Counter({'tie': 32, 'mle': 5, 'base': 3}))
('Cover 32', Counter({'tie': 33, 'mle': 4, 'base': 3}))
 ```
