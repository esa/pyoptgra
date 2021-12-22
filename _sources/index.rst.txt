.. Pyoptgra documentation master file, created by
   sphinx-quickstart on Wed May  5 19:25:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.rst

Further reading
---------------

 - You can set scaling factors for variables and constraints, see :ref:`sec:variable-scaling-factors`.
 - Constraint tolerances can be set with the c_tol attribute of the passed problem, see :ref:`sec:constraint-tolerances`.
 - Optgra offers several functions for :ref:`sec:sensitivity-analysis` of a problem with respect to constraints and parameters, as well as functions for local updates. This feature is still under development.
 - If you have previously used the fortran interface of Optgra, you might be interested in the internals of how pyoptgra calls the underlying fortran code: :ref:`sec:internals`
 - An example of how to optimise a Godot problem can be found here: :ref:`sec:example-notebook`.



Table of Content
----------------

.. toctree::
   :maxdepth: 1

   sensitivity
   scaling-and-tolerances
   changes-from-fortran
   example-notebook

API Reference
=============

.. toctree::
   :maxdepth: 1

   api