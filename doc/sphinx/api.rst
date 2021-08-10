.. _api:

===================
The pyoptgra module
===================

.. autoclass:: pyoptgra.optgra(*args)

	.. automethod:: evolve(population pop)
	.. automethod:: prepare_sensitivity(problem, x)
	.. automethod:: sensitivity_matrices()
	.. automethod:: linear_update_new_callable()
	.. automethod:: linear_update_delta(constraint_delta)

======================
The optgra C++ wrapper
======================

.. doxygenfunction:: optimize
.. doxygenfunction:: compute_sensitivity_matrices

.. doxygenfunction:: prepare_sensitivity_state
.. doxygenfunction:: get_sensitivity_matrices
.. doxygenfunction:: sensitivity_update_new_callable
.. doxygenfunction:: sensitivity_update_constraint_delta