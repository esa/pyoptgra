Optgra
======

|build-status|

.. |build-status| image:: https://github.com/esa/pyoptgra/actions/workflows/workflow.yaml/badge.svg
   :target: https://github.com/esa/pyoptgra/actions

This repository provides *pyoptgra*, a python package wrapping (and including) OPTGRA.
OPTGRA is an optimization algorithm developed and implemented by Johannes Schoenmaekers, it is specifically designed for near-linear constrained problems, which commonly occur in trajectory optimization.

The full documentation can be found here_.

.. _here: https://esa.github.io/pyoptgra/

Installation
============

With Pip
--------

Pyoptgra is available on PyPi and can be installed with pip:

* ``pip install pyoptgra``

Compile from Source
-------------------

First install a C++ compiler, a fortran compiler, cmake, python and python build, then clone the repository and build with ``python -m build``

Usage
=====

Pyoptgra is designed as a pygmo_ user-defined algorithm: First create an instance of the *optgra* class with all relevant parameters, then pass a pygmo.population containing your problem to the instance's *evolve* method:

.. _pygmo: https://esa.github.io/pygmo2/

>>> import pygmo
>>> import pyoptgra
>>> prob = pygmo.problem(pygmo.schwefel(30)) # using the schwefel test problem from pygmo, with 30 dimensions
>>> pop = pygmo.population(prob, 1)
>>> algo = pygmo.algorithm(pyoptgra.optgra())
>>> pop = algo.evolve(pop) # the actual call to OPTGRA

License
=======

Copyright 2008, 2021 European Space Agency

Pyoptgra/Optgra is available under two different licenses. You may choose to license and use it under version 3 of the GNU General Public License or under the ESA Software Community Licence (ESCL) 2.4 Weak Copyleft. We explicitly reserve the right to release future versions of Pyoptgra and Optgra under different licenses.

Copies of GPL3 and ESCL 2.4 can be found in the root directory of this package,
you can also obtain them at https://www.gnu.org/licenses/gpl-3.0.txt
and https://essr.esa.int/license/european-space-agency-community-license-v2-4-weak-copyleft