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
(base) pmineiro@PMINEIRO-1% sudo add-apt-repository ppa:mhier/libboost-latest
...
(base) pmineiro@PMINEIRO-2% sudo aptitude update
...
(base) pmineiro@PMINEIRO-3% sudo aptitude install libboost1.68-dev
...
(base) pmineiro@PMINEIRO-4% conda create -n elfcb python=3.7 numpy scipy
...
(base) pmineiro@PMINEIRO-5% conda activate elfcb
(elfcb) pmineiro@PMINEIRO-6% conda install -c statiskit libboost_python
...
(elfcb) pmineiro@PMINEIRO-7% sudo ln -sf $HOME/miniconda3/envs/elfcb/lib/libboost_python37.so /usr/lib/ # sad life
(elfcb) pmineiro@PMINEIRO-8% sudo ln -sf $HOME/miniconda3/envs/elfcb/lib/libboost_python37.so.1.68.0 /usr/lib/ # sad life
(elfcb) pmineiro@PMINEIRO-9% pip install cmake
...
(elfcb) pmineiro@PMINEIRO-10% cd $VW # $VW is where you cloned https://github.com/VowpalWabbit/vowpal_wabbit
(elfcb) pmineiro@PMINEIRO-11% make clean && make
...
(elfcb) pmineiro@PMINEIRO-12% cd python && python setup.py install
...
(elfcb) pmineiro@PMINEIRO-13% sudo rm /usr/lib/libboost_python37.so /usr/lib/libboost_python37.so.1.68.0 # sad life
```

### Step 2: Install Python packages

```console
(elfcb) pmineiro@PMINEIRO-14% pip install cvxpy jupyter jupyter-contrib-nbextensions jupyter-nbextensions-configurator matplotlib tqdm
(elfcb) pmineiro@PMINEIRO-14% conda install -c conda-forge cvxopt
...
