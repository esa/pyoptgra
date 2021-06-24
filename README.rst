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

Variable Scaling Factors
------------------------

The free variables in an optimization problem are often on different scales, and the optimization performs better if these are aligned.
For this, pyoptgra supports *variable scaling factors*, to be passed with the keyword argument of the same name:

>>> import pygmo
>>> import pyoptgra
>>>
>>> prob = pygmo.problem(pygmo.schwefel(4)) # schwefel test problem with 4 dimensions
>>> pop = pygmo.population(prob, 1)
>>>
>>> scaling_factors = [1,2,2,1]
>>> algo = pygmo.algorithm(pyoptgra.optgra(variable_scaling_factors=scaling_factors))
>>> pop = algo.evolve(pop)

Recommended scaling factors for trajectory optimization problems, depending on the type of variable:

* 5e+0 for longitude (deg)
* 5e-2 for V-infinity (km/s)
* 1e+3 for mass (kg)
* 1e+3 for delta-V (m/s)
* 1e-0 for right ascencion (deg)
* 5e-1 for declination (deg)

Constraint Tolerances
---------------------

The maximum allowed violation of each constraint can be set with the *c_tol* property of the passed problem.
Setting constraint tolerances to zero may lead to divergence.

>>> import pygmo
>>> import pyoptgra
>>>
>>> prob = pygmo.problem(pygmo.luksan_vlcek1(dim=4)) # the luksan_vlcek1 problem has dim-2 constraints
>>> prob.c_tol = [1e-10, 1e-10]
>>> pop = pygmo.population(prob, 1)
>>>
>>> algo = pygmo.algorithm(pyoptgra.optgra())
>>> pop = algo.evolve(pop)

Recommended constraint tolerances for different constraint types:

* 1e-3 for epoch (day)
* 1e-3 for ene (J)
* 1e-6 for eccentricity
* 1e-3 for distance (km)
* 1e-3 for longitude
* 1e-3 for V-infinity (km/s)
* 1e-3 for mass (kg)
* 1e-3 for Delta-V (m/s)
* 1e-3 for ratio of deflection w.r.t. maximum deflection
* 1e-3 for V-infinity and ocm direction (deg)
* 1e-4 for orbital plane (deg)
