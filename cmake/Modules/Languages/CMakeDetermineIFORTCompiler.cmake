# Determine the compiler to use for IFORT programs
# NOTE, a generator may set CMAKE_IFORT_COMPILER before
# loading this file to force a compiler.
# use environment variable IFORT first if defined by user, next use 
# the cmake variable CMAKE_GENERATOR_IFORT which can be defined by a generator
# as a default compiler

IF(NOT CMAKE_IFORT_COMPILER)

  # prefer the environment variable IFORT
  IF($ENV{IFORT} MATCHES ".+")
    GET_FILENAME_COMPONENT(CMAKE_IFORT_COMPILER_INIT $ENV{IFORT} PROGRAM PROGRAM_ARGS CMAKE_IFORT_FLAGS_ENV_INIT)
    IF(CMAKE_IFORT_FLAGS_ENV_INIT)
      SET(CMAKE_IFORT_COMPILER_ARG1 "${CMAKE_IFORT_FLAGS_ENV_INIT}" CACHE STRING "First argument to IFORT compiler")
    ENDIF(CMAKE_IFORT_FLAGS_ENV_INIT)
    IF(EXISTS ${CMAKE_IFORT_COMPILER_INIT})
    ELSE(EXISTS ${CMAKE_IFORT_COMPILER_INIT})
      MESSAGE(FATAL_ERROR "Could not find compiler set in environment variable IFORT:\n$ENV{IFORT}.") 
    ENDIF(EXISTS ${CMAKE_IFORT_COMPILER_INIT})
  ENDIF($ENV{IFORT} MATCHES ".+")

  # next try prefer the compiler specified by the generator
  IF(CMAKE_GENERATOR_IFORT) 
    IF(NOT CMAKE_IFORT_COMPILER_INIT)
      SET(CMAKE_IFORT_COMPILER_INIT ${CMAKE_GENERATOR_IFORT})
    ENDIF(NOT CMAKE_IFORT_COMPILER_INIT)
  ENDIF(CMAKE_GENERATOR_IFORT)

  # finally list compilers to try
  IF(CMAKE_IFORT_COMPILER_INIT)
    SET(CMAKE_IFORT_COMPILER_LIST ${CMAKE_IFORT_COMPILER_INIT})
  ELSE(CMAKE_IFORT_COMPILER_INIT)
    SET(CMAKE_IFORT_COMPILER_LIST ifort )  
  ENDIF(CMAKE_IFORT_COMPILER_INIT)

  # Find the compiler.
  FIND_PROGRAM(CMAKE_IFORT_COMPILER NAMES ${CMAKE_IFORT_COMPILER_LIST} DOC "IFORT compiler")
  IF(CMAKE_IFORT_COMPILER_INIT AND NOT CMAKE_IFORT_COMPILER)
    SET(CMAKE_IFORT_COMPILER "${CMAKE_IFORT_COMPILER_INIT}" CACHE FILEPATH "IFORT compiler" FORCE)
  ENDIF(CMAKE_IFORT_COMPILER_INIT AND NOT CMAKE_IFORT_COMPILER)
ENDIF(NOT CMAKE_IFORT_COMPILER)
MARK_AS_ADVANCED(CMAKE_IFORT_COMPILER)

GET_FILENAME_COMPONENT(COMPILER_LOCATION "${CMAKE_IFORT_COMPILER}"  PATH)

# FIND_PROGRAM(GNAT_EXECUTABLE_BUILDER NAMES ifort PATHS ${COMPILER_LOCATION} )
FIND_PROGRAM(CMAKE_AR NAMES xiar PATHS ${COMPILER_LOCATION} )

FIND_PROGRAM(CMAKE_RANLIB NAMES ranlib)
IF(NOT CMAKE_RANLIB)
   SET(CMAKE_RANLIB : CACHE INTERNAL "noop for ranlib")
ENDIF(NOT CMAKE_RANLIB)
MARK_AS_ADVANCED(CMAKE_RANLIB)

# configure variables set in this file for fast reload later on
#CONFIGURE_FILE(${CMAKE_ROOT}/Modules/CMakeIFORTCompiler.cmake.in 
#message(STATUS "DEBUG: CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
#message(STATUS "DEBUG: CMAKE_FILES_DIRECTORY = ${CMAKE_FILES_DIRECTORY}")
#message(STATUS "DEBUG: CMAKE_PLATFORM_INFO_DIR = ${CMAKE_PLATFORM_INFO_DIR}")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/Modules/Languages/CMakeIFORTCompiler.cmake.in 
  "${CMAKE_PLATFORM_INFO_DIR}/CMakeIFORTCompiler.cmake" IMMEDIATE)
MARK_AS_ADVANCED(CMAKE_AR)

SET(CMAKE_IFORT_COMPILER_ENV_VAR "IFORT")