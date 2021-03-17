cmake_minimum_required(VERSION 3.5)

project( fortranTest )

enable_language(Fortran)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Cmake build type is ${CMAKE_BUILD_TYPE}")

set (OPTGRAFILES ogclos.F ogcpri.F ogcsca.F ogcstr.F ogctyp.F ogderi.F ogdist.F ogeval.F ogexcl.F ogexec.F ogincl.F oginit.F ogiter.F ogleft.F ogrigt.F ogomet.F ogopti.F ogcorr.F ogvsca.F ogvstr.F ogwrit.F ogdata.inc)

# add_library(optgra ${OPTGRAFILES})

add_executable(optgra wrapper.cpp ${OPTGRAFILES})

enable_testing()