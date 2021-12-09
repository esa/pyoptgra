.. _api:

===================
The pyoptgra module
===================

.. autoclass:: pyoptgra.optgra(*args)

	.. automethod:: evolve(self, population pop)
	.. automethod:: prepare_sensitivity(self, problem, x)
	.. automethod:: sensitivity_matrices(self)
	.. automethod:: linear_update_new_callable(self, problem)
	.. automethod:: linear_update_delta(self, constraint_delta)

======================
The optgra C++ wrapper
======================

.. doxygenfunction:: optimize
.. doxygenfunction:: compute_sensitivity_matrices

.. doxygenfunction:: prepare_sensitivity_state
.. doxygenfunction:: get_sensitivity_matrices
.. doxygenfunction:: sensitivity_update_new_callable
.. doxygenfunction:: sensitivity_update_constraint_delta