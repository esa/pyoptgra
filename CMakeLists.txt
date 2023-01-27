cmake_minimum_required(VERSION 3.5)

set(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."	FORCE)

project(optgra-wrapper LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# compiler setup

enable_language(Fortran)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Cmake build type is ${CMAKE_BUILD_TYPE}")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install" CACHE PATH "Prefix prepended to install directories" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if (NOT SKBUILD)
    set(PYBIND11_FINDPYTHON ON)
endif()

# build/configure use of ext libraries
include(cmake/modules/Update_submodules.cmake)
include(cmake/modules/Pybind11.cmake)

# add targets for unit testing framework on C level
add_subdirectory(ext/Catch2)

set(SKBUILD_INSTALL_DIR _sk_install ) # set install dir when using SKBUILD

set (OPTGRAFILES 
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcdel.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogclos.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcpri.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcorr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogcstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogctyp.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogderi.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogdist.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogeval.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogexcl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogexec.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/oggsst.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogincl.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/oginit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogiter.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogleft.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogomet.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogopti.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogrigt.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogsens.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogsopt.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogssst.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogvsca.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogvstr.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogvtyp.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogwlog.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogwrit.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogplog.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogpwri.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogpwri_start.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogpwri_end.F"
 "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/ogdata.inc"
 )

add_library(optgra "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/wrapper.hpp" ${OPTGRAFILES})
target_include_directories(optgra PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core")

add_executable(optgra-test "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/wrappertest.cpp")
target_link_libraries(optgra-test optgra)
target_link_libraries(optgra-test Catch2::Catch2WithMain)

add_executable(raii-test "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/raiitest.cpp")
target_link_libraries(raii-test optgra)
target_link_libraries(raii-test Catch2::Catch2WithMain)

add_executable(bare-test "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/bare-test.cpp")
target_link_libraries(bare-test optgra)
target_link_libraries(bare-test Catch2::Catch2WithMain)

pybind11_add_module(core "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/bindings.cpp")
target_link_libraries(core PRIVATE optgra)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/pyoptgra/core/__init__.py" DESTINATION .)
install(TARGETS core optgra LIBRARY DESTINATION . ARCHIVE DESTINATION lib)
