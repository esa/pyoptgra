#cmake_policy(SET CMP0077 NEW) # does not work because it is overriden by cmake_minimum_required() in pybind11 CMakeLists.txt
#set(PYBIND11_INSTALL ON)  # requires Policy CMP0077 to be set to NEW: option() honors normal variables.
add_subdirectory(ext/pybind11)
# manual install of pybind11 headers (workaround because pybind11 build ignores policy settings)
if (NOT SKBUILD)
    install( DIRECTORY ext/pybind11/include/pybind11 DESTINATION ${CMAKE_INSTALL_PREFIX}/include ) # do not add trailing `/` after "pybind11"
endif()
