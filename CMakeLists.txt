cmake_minimum_required(VERSION 3.5)

project( fortranTest )

enable_language(Fortran)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Pagmo REQUIRED)

message("Cmake build type is ${CMAKE_BUILD_TYPE}")

set (OPTGRAFILES 
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogclos.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogcpri.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogcsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogcstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogctyp.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogderi.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogdist.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogeval.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogexcl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogexec.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogincl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/oginit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogiter.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogleft.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogrigt.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogomet.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogopti.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogcorr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogvsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogvstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogwrit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/ogdata.inc"
 )

add_library(optgra "${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.hpp" ${OPTGRAFILES})
target_link_libraries(optgra PUBLIC Pagmo::pagmo)


add_executable(optgra-test "${CMAKE_CURRENT_SOURCE_DIR}/src/wrappertest.cpp")
target_link_libraries(optgra-test optgra)


enable_testing()