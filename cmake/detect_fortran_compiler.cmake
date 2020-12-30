cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
cmake_policy(VERSION 2.8.5)

# This cmake script (when saved as detect_fortran_compiler.cmake) is invoked by:
#
#     cmake -P detect_fortran_compiler.cmake
#
# It is written for clarity, not brevity.

# First make a new directory, so that we don't mess up the current one.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory fortran_detection_area
    WORKING_DIRECTORY .
)

# Here, we generate a key file that CMake needs.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E echo "enable_language(Fortran OPTIONAL)\n message(\"\${CMAKE_Fortran_COMPILER}\")"
    WORKING_DIRECTORY fortran_detection_area
    OUTPUT_FILE CMakeLists.txt
)


# Have CMake check the basic configuration.  The output is
# actually in the form that you posted in your question, but
# instead of displaying it onscreen, we save it to a variable
# so that we can select only parts of it to print later.
execute_process(
    COMMAND ${CMAKE_COMMAND} --check-system-vars
    OUTPUT_VARIABLE the_output
    ERROR_VARIABLE the_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY fortran_detection_area
)

# Eliminate the directory, including all of the files within it that
# CMake created.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E remove_directory fortran_detection_area
    WORKING_DIRECTORY .
)

# Here, you have the entire message captured as a variable.
# Uncomment this next line to convince yourself of this.
#message(STATUS "the_output = |${the_output}|.")
#message(STATUS "the_error  = |${the_error}|.")

# Here, we search the message to see if the C++ compiler was found or not,
# and print an arbitrary message accordingly.
string(FIND "${the_error}" "CMAKE_Fortran_COMPILER-NOTFOUND" scan_result)
#message(STATUS "scan_result = |${scan_result}|.")
set(Fortran_COMPILER_FOUND 1)
if(NOT(-1 EQUAL "${scan_result}"))
  set(Fortran_COMPILER_FOUND 0)
endif()

