cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project( fortranTest )

enable_language(Fortran)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

message("Cmake build type is ${CMAKE_BUILD_TYPE}")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install" CACHE PATH "Prefix prepended to install directories" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(SKBUILD_INSTALL_DIR _sk_install ) # set install dir got godot shared objects when using SKBUILD

set (OPTGRAFILES 
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogclos.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcpri.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogctyp.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogderi.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogdist.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogeval.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogexcl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogexec.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogincl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/oginit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogiter.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogleft.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogrigt.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogomet.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogopti.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcorr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogvsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogvstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogwlog.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogwrit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogdata.inc"
 )

add_library(optgra "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/wrapper.hpp" ${OPTGRAFILES})

add_executable(optgra-test "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/wrappertest.cpp")
target_link_libraries(optgra-test optgra)

pybind11_add_module(core "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/bindings.cpp")
target_link_libraries(core PRIVATE optgra)

install(TARGETS core optgra LIBRARY DESTINATION .)

enable_testing()