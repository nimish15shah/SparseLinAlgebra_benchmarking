# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
#
# ==================================================================================================
#
# Defines the following variables:
#   CLMAGMA_FOUND          Boolean holding whether or not the clMAGMA library was found
#   CLMAGMA_INCLUDE_DIRS   The clMAGMA include directory
#   CLMAGMA_LIBRARIES      The clMAGMA library
#
# In case clMAGMA is not installed in the default directory, set the CLMAGMA_ROOT variable to point to
# the root of clMAGMA, such that 'magma.h' can be found in $CLMAGMA_ROOT/include. This can either be
# done using an environmental variable (e.g. export CLMAGMA_ROOT=/path/to/clMAGMA) or using a CMake
# variable (e.g. cmake -DCLMAGMA_ROOT=/path/to/clMAGMA ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CLMAGMA_HINTS
  ${CLMAGMA_ROOT}
  $ENV{CLMAGMA_ROOT}
)
set(CLMAGMA_PATHS
  /usr
  /usr/local
)

# Finds the include directories
find_path(CLMAGMA_INCLUDE_DIRS
  NAMES magma.h
  HINTS ${CLMAGMA_HINTS}
  PATH_SUFFIXES src include inc include/x86_64 include/x64
  PATHS ${CLMAGMA_PATHS}
  DOC "clMAGMA include header magma.h"
)
mark_as_advanced(CLMAGMA_INCLUDE_DIRS)

# Finds the library
find_library(CLMAGMA_LIBRARIES
  NAMES clmagma
  HINTS ${CLMAGMA_HINTS}
  PATH_SUFFIXES lib build/library lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32
  PATHS ${CLMAGMA_PATHS}
  DOC "clMAGMA library"
)
mark_as_advanced(CLMAGMA_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CLMAGMA_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'magma.h', install clMAGMA or set CLMAGMA_ROOT")
endif()
if(NOT CLMAGMA_LIBRARIES)
    message(STATUS "Could NOT find clMAGMA library, install it or set CLMAGMA_ROOT")
endif()

# Determines whether or not clMAGMA was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(clMAGMA DEFAULT_MSG CLMAGMA_INCLUDE_DIRS CLMAGMA_LIBRARIES)

# ==================================================================================================
