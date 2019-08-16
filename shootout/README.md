# Shootout

Results from the paper comparing empirical likelihood to baselines for estimation and learning.

## Estimation

* ```make estimationshootout``` will (eventually) produce the results in Tables 1 and 4.
```console
(elfcb) pmineiro@PMINEIRO-253% make estimationshootout
./do-estimation-shootout.py --dirname orig40
('EpsilonGreedy 0.05',
 {'ipsvsmle': Counter({'mle': 26, 'tie': 11, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 29, 'mle': 10, 'snips': 1})})
('EpsilonGreedy 0.1',
 {'ipsvsmle': Counter({'mle': 24, 'tie': 13, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 35, 'mle': 5})})
('EpsilonGreedy 0.25',
 {'ipsvsmle': Counter({'mle': 27, 'tie': 10, 'ips': 3}),
  'snipsvsmle': Counter({'tie': 38, 'mle': 2})})
('Bag 10',
 {'ipsvsmle': Counter({'tie': 20, 'mle': 11, 'ips': 9}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 5, 'snips': 2})})
('Bag 32',
 {'ipsvsmle': Counter({'mle': 18, 'tie': 15, 'ips': 7}),
  'snipsvsmle': Counter({'tie': 33, 'mle': 5, 'snips': 2})})
('Cover 10',
 {'ipsvsmle': Counter({'tie': 15, 'mle': 14, 'ips': 11}),
  'snipsvsmle': Counter({'tie': 32, 'mle': 8})})
('Cover 32',
 {'ipsvsmle': Counter({'ips': 16, 'tie': 13, 'mle': 11}),
  'snipsvsmle': Counter({'tie': 29, 'snips': 7, 'mle': 4})})
```

## Learning

 * ```make learningshootoutorig40``` will (eventually) produce the first column of results from Figure 3.
 ```console
(elfcb) pmineiro@PMINEIRO-52% make learningshootoutorig40
eval ./do-learning-shootout.py --dirname orig40
('EpsilonGreedy 0.05', Counter({'tie': 18, 'ci': 12, 'base': 10}))
('EpsilonGreedy 0.1', Counter({'tie': 18, 'ci': 17, 'base': 5}))
('EpsilonGreedy 0.25', Counter({'tie': 20, 'ci': 16, 'base': 4}))
('Bag 10', Counter({'ci': 20, 'tie': 18, 'base': 2}))
('Bag 32', Counter({'tie': 30, 'ci': 7, 'base': 3}))
('Cover 10', Counter({'tie': 22, 'ci': 16, 'base': 2}))
('Cover 32', Counter({'tie': 25, 'ci': 15}))
```

* ```make learningshootoutmleorig40``` will (eventually) produce the second column of results from Figure 3.
```console
(elfcb) pmineiro@PMINEIRO-4% make learningshootoutmleorig40
eval ./do-learning-shootout.py --dirname orig40 --challenger mle
('EpsilonGreedy 0.05', Counter({'tie': 26, 'mle': 8, 'base': 6}))
('EpsilonGreedy 0.1', Counter({'tie': 22, 'mle': 15, 'base': 3}))
('EpsilonGreedy 0.25', Counter({'tie': 37, 'base': 2, 'mle': 1}))
('Bag 10', Counter({'tie': 27, 'mle': 12, 'base': 1}))
('Bag 32', Counter({'tie': 32, 'mle': 7, 'base': 1}))
('Cover 10', Counter({'tie': 27, 'mle': 12, 'base': 1}))
('Cover 32', Counter({'tie': 32, 'mle': 8}))
```

# Other Stuff

Things not in the paper.

### Alternative Set of Data Sets

 * ```make learningshootoutgt10class``` 
 ```console
 (elfcb) pmineiro@PMINEIRO-207% make learningshootoutgt10class
eval ./do-learning-shootout.py --dirname gt10class
('EpsilonGreedy 0.05', Counter({'tie': 35, 'base': 3, 'mle': 2}))
('EpsilonGreedy 0.1', Counter({'tie': 35, 'base': 4, 'mle': 1}))
('EpsilonGreedy 0.25', Counter({'tie': 31, 'base': 8, 'mle': 1}))
('Bag 10', Counter({'tie': 36, 'mle': 4}))
('Bag 32', Counter({'tie': 39, 'mle': 1}))
('Cover 10', Counter({'tie': 39, 'mle': 1}))
('Cover 32', Counter({'tie': 36, 'base': 3, 'mle': 1}))
 ```
 
 * ```make learningshootoutmlegt10class``` 
 ```console
 (elfcb) pmineiro@PMINEIRO-209% make learningshootoutmlegt10class
eval ./do-learning-shootout.py --dirname gt10class --challenger mle
('EpsilonGreedy 0.05', Counter({'tie': 31, 'base': 5, 'mle': 4}))
('EpsilonGreedy 0.1', Counter({'tie': 34, 'base': 4, 'mle': 2}))
('EpsilonGreedy 0.25', Counter({'tie': 36, 'base': 3, 'mle': 1}))
('Bag 10', Counter({'tie': 37, 'mle': 2, 'base': 1}))
('Bag 32', Counter({'tie': 36, 'mle': 3, 'base': 1}))
('Cover 10', Counter({'tie': 37, 'mle': 2, 'base': 1}))
('Cover 32', Counter({'tie': 39, 'base': 1}))
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
* ```make learningshootoutonlinegt10class``` will (eventually) produce results analogous to the first column of Table 5.  It equals or exceeds the batch strategy across the board.
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
### Control Variates

This is new since the paper. The empirical likelihood model is augmented with additional constraints corresponding to random variables whose true expectation is known to be zero.  There are two variants.

#### Reward predictor control variate

Given a reward predictor $\hat{r}$, we can form a control variate $E_{(x, r) \sim D, a \sim h}\left[\frac{\pi(a|x)}{h(a|x)} \hat{r}(x, a)\right] = E_{(x, r) \sim D, a \sim \pi}\left[\hat{r}(x,a)\right]$.  Forcing the empirical likelihood latent distribution to obey this constraint is analogous to doubly robust estimation.  It provides improvement, but note that the comparison strategies do not employ a reward predictor, so this is arguably not &ldquo;apples-to-apples&rdquo;.

##### Estimation
Comparing to ```make estimationshootout``` above indicates improvement.
```console
(elfcb) pmineiro@PMINEIRO-31% make estimationshootoutdr
./do-estimation-shootout.py --dirname orig40 --challenger mledr
('EpsilonGreedy 0.05',
 {'ipsvsmledr': Counter({'mledr': 25, 'tie': 12, 'ips': 3}),
  'snipsvsmledr': Counter({'tie': 29, 'mledr': 11})})
('EpsilonGreedy 0.1',
 {'ipsvsmledr': Counter({'mledr': 25, 'tie': 12, 'ips': 3}),
  'snipsvsmledr': Counter({'tie': 30, 'mledr': 10})})
('EpsilonGreedy 0.25',
 {'ipsvsmledr': Counter({'mledr': 28, 'tie': 9, 'ips': 3}),
  'snipsvsmledr': Counter({'tie': 29, 'mledr': 11})})
('Bag 10',
 {'ipsvsmledr': Counter({'tie': 21, 'mledr': 10, 'ips': 9}),
  'snipsvsmledr': Counter({'tie': 32, 'mledr': 6, 'snips': 2})})
('Bag 32',
 {'ipsvsmledr': Counter({'mledr': 18, 'tie': 15, 'ips': 7}),
  'snipsvsmledr': Counter({'tie': 31, 'mledr': 7, 'snips': 2})})
('Cover 10',
 {'ipsvsmledr': Counter({'mledr': 15, 'tie': 14, 'ips': 11}),
  'snipsvsmledr': Counter({'tie': 32, 'mledr': 8})})
('Cover 32',
 {'ipsvsmledr': Counter({'ips': 16, 'tie': 13, 'mledr': 11}),
  'snipsvsmledr': Counter({'tie': 29, 'snips': 6, 'mledr': 5})})
```

##### Learning
Comparing to ```make learningshootoutmleorig40``` indicates improvement.
```console
(elfcb) pmineiro@PMINEIRO-162% make learningshootoutmledrorig40 
eval ./do-learning-shootout.py --dirname orig40 --challenger mledr
('EpsilonGreedy 0.05', Counter({'tie': 25, 'mledr': 10, 'base': 5}))
('EpsilonGreedy 0.1', Counter({'tie': 26, 'mledr': 12, 'base': 2}))
('EpsilonGreedy 0.25', Counter({'tie': 33, 'mledr': 6, 'base': 1}))
('Bag 10', Counter({'tie': 24, 'mledr': 13, 'base': 3}))
('Bag 32', Counter({'tie': 30, 'mledr': 9, 'base': 1}))
('Cover 10', Counter({'tie': 28, 'mledr': 8, 'base': 4}))
('Cover 32', Counter({'tie': 37, 'mledr': 2, 'base': 1}))
```
For completeness ...
```console
(elfcb) pmineiro@PMINEIRO-417% make learningshootoutmledrgt10class
eval ./do-learning-shootout.py --dirname gt10class --challenger mledr
('EpsilonGreedy 0.05', Counter({'tie': 37, 'mledr': 2, 'base': 1}))
('EpsilonGreedy 0.1', Counter({'tie': 38, 'base': 2}))
('EpsilonGreedy 0.25', Counter({'tie': 34, 'base': 3, 'mledr': 3}))
('Bag 10', Counter({'tie': 34, 'mledr': 6}))
('Bag 32', Counter({'tie': 35, 'mledr': 3, 'base': 2}))
('Cover 10', Counter({'tie': 29, 'mledr': 11}))
('Cover 32', Counter({'tie': 36, 'mledr': 4}))
```

#### Action control variates

For each action a, we have $E_{(x,r) \sim D, a' \sim h}\left[ \frac{\pi(a'|x)}{h(a'|x)} 1_{a'=a} \right] = E_{(x,r) \sim D}\left[ \pi(a|x) \right ]$ (in English: the expected value of the importance weight for each action a is equal to the probability that the evaluated policy \pi plays a).  These control variates do not use reward information, so this is an &ldquo;apples-to-apples&rdquo; comparison.  

##### Estimation
Comparing to ```make estimationshootout``` above indicates improvement.
```console
(elfcb) pmineiro@PMINEIRO-70% make estimationshootoutcv
./do-estimation-shootout.py --dirname orig40 --challenger mlecv
('EpsilonGreedy 0.05',
 {'ipsvsmlecv': Counter({'mlecv': 26, 'tie': 11, 'ips': 3}),
  'snipsvsmlecv': Counter({'tie': 32, 'mlecv': 8})})
('EpsilonGreedy 0.1',
 {'ipsvsmlecv': Counter({'mlecv': 24, 'tie': 13, 'ips': 3}),
  'snipsvsmlecv': Counter({'tie': 34, 'mlecv': 6})})
('EpsilonGreedy 0.25',
 {'ipsvsmlecv': Counter({'mlecv': 27, 'tie': 10, 'ips': 3}),
  'snipsvsmlecv': Counter({'tie': 37, 'mlecv': 3})})
('Bag 10',
 {'ipsvsmlecv': Counter({'tie': 20, 'mlecv': 11, 'ips': 9}),
  'snipsvsmlecv': Counter({'tie': 33, 'mlecv': 5, 'snips': 2})})
('Bag 32',
 {'ipsvsmlecv': Counter({'mlecv': 18, 'tie': 15, 'ips': 7}),
  'snipsvsmlecv': Counter({'tie': 32, 'mlecv': 6, 'snips': 2})})
('Cover 10',
 {'ipsvsmlecv': Counter({'mlecv': 15, 'tie': 13, 'ips': 12}),
  'snipsvsmlecv': Counter({'tie': 32, 'mlecv': 8})})
('Cover 32',
 {'ipsvsmlecv': Counter({'ips': 15, 'tie': 14, 'mlecv': 11}),
  'snipsvsmlecv': Counter({'tie': 30, 'snips': 6, 'mlecv': 4})})
```


