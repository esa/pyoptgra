Optgra
======

This repository provides *pyoptgra*, a python package wrapping (and including) OPTGRA.
OPTGRA is an optimization algorithm developed and implemented by Johannes Schoenmaekers, it is specifically designed for near-linear constrained problems, which commonly occur in trajectory optimization.

Installation
============

With Pip
--------

* ```pip install pyoptgra --extra-index-url https://gitlab.esa.int/api/v4/projects/4531/packages/pypi/simple```

Usage
=====

Pyoptgra is designed as a `pygmo <https://esa.github.io/pygmo2/>` user-defined algorithm: First create an instance of the *optgra* class with all relevant parameters, then pass a pygmo.population containing your problem to the instance's *evolve* method:

>>> import pygmo
>>> import pyoptgra
>>> prob = pygmo.problem(pygmo.schwefel(30)) # using the schwefel test problem from pygmo, with 30 dimensions
>>> pop = pygmo.population(prob, 1)
>>> algo = pygmo.algorithm(pyoptgra.optgra())
>>> pop = algo.evolve(pop) # the actual call to OPTGRA

License
=======

Pyoptgra/Optgra is available under two different licenses. The version to be found at Github and the Python Package Index (pypi) is available under version 3 of the GNU General Public License, while the version available at the Space Codev platform is available under the ESA Software Community Licence (ESCL). The versions are otherwise identical in content.

We explicitly reserve the right to release future versions of Pyoptgra and Optgra under different licenses.