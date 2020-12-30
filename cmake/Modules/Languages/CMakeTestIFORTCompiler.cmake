# This file is used by EnableLanguage in cmGlobalGenerator to determine that
# the IFORT builder GNAT_EXECUTABLE_BUILDER = gnatmake can actually compile
# and link the most basic of programs.  If not, a fatal error is set and
# cmake stops processing commands and will not generate any makefiles or
# projects.

function(PrintTestCompilerStatus LANG MSG)
  if(CMAKE_GENERATOR MATCHES Make)
    message(STATUS "Check for working ${LANG} compiler: ${CMAKE_${LANG}_COMPILER}${MSG}")
  else()
    message(STATUS "Check for working ${LANG} compiler using: ${CMAKE_GENERATOR}${MSG}")
  endif()
endfunction()

unset(CMAKE_IFORT_COMPILER_WORKS CACHE)
if (NOT CMAKE_IFORT_COMPILER_WORKS)
  PrintTestCompilerStatus("IFORT" "")
  set(test_program "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testIFORTCompiler.f90")
  set(test_cmake "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/CMakeLists.txt")
  file(WRITE ${test_program}
    "
    PROGRAM TESTFortran
    PRINT *, 'Hello'
    END
    "
    )
  FILE(WRITE ${test_cmake}
    "
  cmake_minimum_required(VERSION 3.0)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules/Languages)
  set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL \"\" FORCE)
  project(test IFORT)
  set_source_files_properties(${test_program} PROPERTIES LANGUAGE IFORT)
  add_executable(testIFORTCompiler ${test_program})
  set_target_properties(testIFORTCompiler PROPERTIES LINKER_LANGUAGE IFORT)
    "
    )

  try_compile(CMAKE_IFORT_COMPILER_WORKS
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp
    projectName
    OUTPUT_VARIABLE __CMAKE_IFORT_COMPILER_OUTPUT
    )


  #  try_compile(CMAKE_IFORT_COMPILER_WORKS ${CMAKE_BINARY_DIR}
  #  #    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testIFORTCompiler.cxx
  #  ${test_program}
  #  OUTPUT_VARIABLE __CMAKE_IFORT_COMPILER_OUTPUT)
  # message("stt= ${__CMAKE_IFORT_COMPILER_OUTPUT}")
  # Move result from cache to normal variable.
  set(CMAKE_IFORT_COMPILER_WORKS ${CMAKE_IFORT_COMPILER_WORKS})
  unset(CMAKE_IFORT_COMPILER_WORKS CACHE)
  set(IFORT_TEST_WAS_RUN 1)
endif (NOT CMAKE_IFORT_COMPILER_WORKS)

if(NOT CMAKE_IFORT_COMPILER_WORKS)
  PrintTestCompilerStatus("IFORT" " -- broken")
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining if the IFORT compiler works failed with "
    "the following output:\n${__CMAKE_IFORT_COMPILER_OUTPUT}\n\n")
  message(FATAL_ERROR "The Intel Fortran compiler \"${CMAKE_IFORT_COMPILER}\" "
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n ${__CMAKE_IFORT_COMPILER_OUTPUT}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(IFORT_TEST_WAS_RUN)
    PrintTestCompilerStatus("IFORT" " -- works")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the IFORT compiler works passed with "
      "the following output:\n${__CMAKE_IFORT_COMPILER_OUTPUT}\n\n")
  endif()

  # Try to identify the ABI and configure it into CMakeIFORTCompiler.cmake
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
  CMAKE_DETERMINE_COMPILER_ABI(IFORT ${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp)
  # Try to identify the compiler features
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompileFeatures.cmake)
  CMAKE_DETERMINE_COMPILE_FEATURES(IFORT)

  # Re-configure to save learned information.
  #  configure_file(
  #  ${CMAKE_ROOT}/Modules/CMakeIFORTCompiler.cmake.in
  #  ${CMAKE_PLATFORM_INFO_DIR}/CMakeIFORTCompiler.cmake
  #  @ONLY
  #  )
  #include(${CMAKE_PLATFORM_INFO_DIR}/CMakeIFORTCompiler.cmake)

  if(CMAKE_IFORT_SIZEOF_DATA_PTR)
    foreach(f ${CMAKE_IFORT_ABI_FILES})
      include(${f})
    endforeach()
    unset(CMAKE_IFORT_ABI_FILES)
  endif()
endif()

unset(__CMAKE_IFORT_COMPILER_OUTPUT)
