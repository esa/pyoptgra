.. _api:

===================
The pyoptgra module
===================

.. autoclass:: pyoptgra.optgra(*args)

	.. automethod:: evolve(population pop)

======================
The optgra C++ wrapper
======================

.. doxygenfunction:: optimize
.. doxygenfunction:: compute_sensitivity_matrices

.. doxygenfunction:: prepare_sensitivity_state
.. doxygenfunction:: get_sensitivity_matrices
.. doxygenfunction:: sensitivity_update_new_callable
.. doxygenfunction:: sensitivity_update_constraint_delta