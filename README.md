# elfcb
Code to reproduce the results in the paper [Empirical Likelihood for Contextual Bandits](https://arxiv.org/abs/1906.03323)

## Manifest

* [estimate.ipynb](estimate.ipynb): MLE on synthetic data (Figure 2 of the paper)
* [ci.ipynb](ci.ipynb): CI on synthetic data (Figure 1 and Figure 3 of the paper)
* [shootout](shootout): Generate contents of Tables 1, 3, 4 and 5.

## Setting up the Python Environment

We used [miniconda](https://docs.conda.io/en/latest/miniconda.html).  

### Step 1: Building pyvw

Here's a recipe.
```console
(base) % sudo add-apt-repository ppa:mhier/libboost-latest
...
(base) % sudo aptitude update
...
(base) % sudo aptitude install libboost1.68-dev
...
(base) % conda create -n elfcb python=3.7 numpy scipy
...
(base) % conda activate elfcb
(elfcb) % conda install -c statiskit libboost_python
...
(elfcb) % sudo ln -sf $HOME/miniconda3/envs/elfcb/lib/libboost_python37.so /usr/lib/ # sad life
(elfcb) % sudo ln -sf $HOME/miniconda3/envs/elfcb/lib/libboost_python37.so.1.68.0 /usr/lib/ # sad life
(elfcb) % pip install cmake
...
(elfcb) % cd $VW # $VW is where you cloned https://github.com/VowpalWabbit/vowpal_wabbit
(elfcb) % make clean && make
...
(elfcb) % cd python && python setup.py install
...
(elfcb) % sudo rm /usr/lib/libboost_python37.so /usr/lib/libboost_python37.so.1.68.0 # sad life
```

### Step 2: Install Python packages

```console
(elfcb) % pip install cvxpy jupyter jupyter-contrib-nbextensions jupyter-nbextensions-configurator matplotlib quadprog tqdm
(elfcb) % conda install -c conda-forge cvxopt
...
