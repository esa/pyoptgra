.. _sec:internals:

Internal Workings of the Python to Fortran Interface
====================================================

The python class pyoptgra handles all calls to the Fortran subroutines of Optgra in the background, including allocating and deallocating memory.
Since only one instance of Optgra can have allocated memory in each process, the allocations with oginit and deallocations with ogclos are wrapped in each function call, so that the pyoptgra class does not hold any reference to allocated memory. Thus, all calls are isolated from each other and a user does not have to worry about the internal state.

.. _sec:internals-init:

Pyoptgra Initialization
-----------------------
The __init__ method of pyoptgra does not call any Fortran functions, but stores the construction arguments on the python side.

.. _sec:internals-evolve:

evolve
------
The evolve method calls, by way of the C++ wrapper, the following Optgra functions to initialize Optgra:

 - *oginit* - initialize data structures
 - *ogctyp* - pass constraint types
 - *ogderi* - set method of derivatives
 - *ogdist* - set maximum distance per iteration
 - *ogvtyp* - set variable types
 - *ogiter* - set maximum number of iterations
 - *ogomet* - set optimization method
 - *ogwlog* - set logging

If scaling factors and constraint priorities are given, the following functions are called to pass them:
 - *ogvsca* - variable scaling factors
 - *ogcsca* - constraint scaling factors
 - *ogcpri* - constraint priorities

Finally, *ogexec* is called to start an optimization run and get the result, followed by *ogclos* to deallocate the Optgra memory.

.. _sec:internals-prepare:

prepare_sensitivity
-------------------

The method prepare_sensitivity calls the same initialization functions of Optgra as the evolve method:

 - *oginit* - initialize data structures
 - *ogctyp* - pass constraint types
 - *ogderi* - set method of derivatives
 - *ogdist* - set maximum distance per iteration
 - *ogvtyp* - set variable types
 - *ogiter* - set maximum number of iterations
 - *ogomet* - set optimization method
 - *ogwlog* - set logging

Following the initialization, it calls *ogsopt* (-1) and *ogexec* to initialize the sensitivity analysis in Optgra.
The values of the common block variables SENVAR, SENQUA, SENCON, SENACT, SENDER, ACTCON, CONACT and CONRED are then copied (using the new function *oggsst*) and stored in the python object, these will later be used to set a new instance of OPTGRA to the prepared state.
Finally, *ogclos* is called to de-allocate the Optgra memory. 

This approach of copying the internal state to python instead of keeping a reference to the allocated Fortran memory was chosen since multiple calls involving the same allocated Fortran common block interfere with each other. Thus all other calls to evolve from the same process would be blocked, even if used in other objects. This would conflict with the object-oriented approach chosen for pygmo, where different instances of the same object can be initialized and used independently.


sensitivity_matrices
--------------------

The method sensitivity_matrices calls the same initialization functions of Optgra as prepare_sensitivity, most of them with the default paramters.

The variables SENVAR, SENQUA, SENCON, SENACT, SENDER, ACTCON, CONACT and CONRED are set (using *ogssst*) to the values gotten in prepare_sensitivity, then *ogsens* is called and the output is returned.

Finally, *ogclos* is called to deallocate the Fortran memory.


linear_update_new_callable
--------------------------
The method linear_update_new_callable first initializes the Optgra memory by calling the same initialization functions as prepare_sensitivity, then sets the sensitivity state captured in the variables SENVAR, SENQUA, SENCON, SENACT, SENDER, ACTCON, CONACT and CONRED.

It then calls *ogsopt* (1) to initialize the correct sensitivity mode, followed by *ogexec* with the new callable.

Finally, *ogclos* is called to deallocate the Fortran memory.


linear_update_delta
-------------------

The method linear_update_delta first initializes the Optgra memory by calling the same initialization functions as prepare_sensitivity, then sets the sensitivity state captured in the variables SENVAR, SENQUA, SENCON, SENACT, SENDER, ACTCON, CONACT and CONRED.

It then calls *ogsopt* (2) to initialize the correct sensitivity mode, followed by *ogcdel* to set the constraint deltas and *ogexec*.

Finally, *ogclos* is called to deallocate the Fortran memory.