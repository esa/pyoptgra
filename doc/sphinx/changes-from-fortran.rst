For Users of the Fortran Version of Optgra
==========================================

The python class pyoptgra handles all calls to the Fortran subroutines of Optgra in the background, including allocating and deallocating memory.
Since only one instance of Optgra can have allocated memory in each process, the allocations with oginit and deallocations with ogclos are wrapped in each function call, so that the pyoptgra class does not hold any reference to allocated memory.


Pyoptgra Initialization
=======================
The __init__ method of pyoptgra does not call any Fortran functions, but stores the construction arguments on the python side.


Evolve
======
The evolve method calls, by way of the C++ wrapper, the following Optgra functions to initialize Optgra:

 - oginit - initialize data structures
 - ogctyp - pass constraint types
 - ogderi - set method of derivatives
 - ogdist - set maximum distance per iteration
 - ogvtyp - set variable types
 - ogiter - set maximum number of iterations
 - ogomet - set optimization method
 - ogwlog - set logging

If scaling factors and constraint priorities are given, the following functions are called to pass them:
 - ogvsca - variable scaling factors
 - ogcsca - constraint scaling factors
 - ogcpri - constraint priorities

Finally, ogexec is called to start an optimization run and get the result, followed by ogclos to deallocate the Optgra memory.
