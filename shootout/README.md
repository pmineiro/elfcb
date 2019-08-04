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
('EpsilonGreedy 0.05', Counter({'tie': 18, 'mle': 17, 'base': 5}))
('EpsilonGreedy 0.1', Counter({'tie': 18, 'mle': 16, 'base': 6}))
('EpsilonGreedy 0.25', Counter({'tie': 24, 'mle': 12, 'base': 4}))
('Bag 10', Counter({'tie': 30, 'mle': 6, 'base': 4}))
('Bag 32', Counter({'tie': 26, 'mle': 11, 'base': 3}))
('Cover 10', Counter({'tie': 26, 'mle': 8, 'base': 6}))
('Cover 32', Counter({'tie': 25, 'mle': 10, 'base': 5}))
```
 * ```make learningshootoutgt10class``` will (eventually) produce the second column of results from Figure 3.
 ```console
 (elfcb) pmineiro@PMINEIRO-207% make learningshootoutgt10class
./do-learning-shootout.py --dirname gt10class
('EpsilonGreedy 0.05', Counter({'mle': 30, 'tie': 9, 'base': 1}))
('EpsilonGreedy 0.1', Counter({'mle': 29, 'tie': 9, 'base': 2}))
('EpsilonGreedy 0.25', Counter({'mle': 31, 'tie': 8, 'base': 1}))
('Bag 10', Counter({'mle': 22, 'tie': 14, 'base': 4}))
('Bag 32', Counter({'mle': 22, 'tie': 16, 'base': 2}))
('Cover 10', Counter({'tie': 20, 'mle': 18, 'base': 2}))
('Cover 32', Counter({'tie': 26, 'mle': 10, 'base': 4}))
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
### Incremental Learning

This is an "online" (in the computationally incremental sense) dual update strategy combined with learning.  It is not in the original paper.

* ```make learningshootoutonlineorig40``` will (eventually) produce results analogous to the first column of Table 3.  Results with cover are particularly good compared to the batch strategy.
```console
(elfcb) pmineiro@PMINEIRO-132% make learningshootoutonlineorig40
./do-learning-shootout.py --dirname orig40 --online
('EpsilonGreedy 0.05', Counter({'mle': 17, 'base': 13, 'tie': 10}))
('EpsilonGreedy 0.1', Counter({'base': 16, 'mle': 14, 'tie': 10}))
('EpsilonGreedy 0.25', Counter({'tie': 17, 'mle': 14, 'base': 9}))
('Bag 10', Counter({'tie': 20, 'base': 13, 'mle': 7}))
('Bag 32', Counter({'tie': 20, 'base': 12, 'mle': 8}))
('Cover 10', Counter({'mle': 18, 'tie': 17, 'base': 5}))
('Cover 32', Counter({'tie': 18, 'mle': 17, 'base': 5}))
```
* ```make learningshootoutonlinegt10class`` will (eventually) produce results analogous to the first column of Table 5.  It equals or exceeds the batch strategy across the board. 
```console
(elfcb) pmineiro@PMINEIRO-26% make learningshootoutonlinegt10class
./do-learning-shootout.py --dirname gt10class --online
('EpsilonGreedy 0.05', Counter({'mle': 31, 'base': 6, 'tie': 3}))
('EpsilonGreedy 0.1', Counter({'mle': 32, 'tie': 4, 'base': 4}))
('EpsilonGreedy 0.25', Counter({'mle': 37, 'tie': 2, 'base': 1}))
('Bag 10', Counter({'mle': 28, 'tie': 7, 'base': 5}))
('Bag 32', Counter({'mle': 30, 'tie': 6, 'base': 4}))
('Cover 10', Counter({'mle': 33, 'tie': 4, 'base': 3}))
('Cover 32', Counter({'mle': 32, 'tie': 5, 'base': 3}))
```
