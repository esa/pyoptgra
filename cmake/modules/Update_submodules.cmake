# Update git submodules as needed

# The Python source distribution isn't a git checkout
if (EXISTS ${PROJECT_SOURCE_DIR}/.git)
    message(STATUS "Performing submodule update")
    execute_process(COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()
