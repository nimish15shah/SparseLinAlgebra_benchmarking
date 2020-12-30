
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(MKL)

set(COMMON_INCLUDE_DIR "../../common/")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

# ==================================================================================================

if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 2_sparse)
endif()
set(NAME ${DWARF_PREFIX}_dss)

# ==================================================================================================
# C++ compiler settings

find_package(Common)

select_compiler_flags(cxx_flags
  GNU "-march=native"   # I suggest remove "-O3" as this is controlled by the CMAKE_BUILD_TYPE
  CLANG "-march=native" # same here
  Intel "-axavx2,avx")
set(CXX_FLAGS ${cxx_flags})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wno-comment")
  if(APPLE)
    set(CXX_FLAGS "${CXX_FLAGS} -Wa,-q")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS} -g -L/esat/puck1/users/nshah/cpu_gpu_parallel/intel_full_compiler/intel/mkl/2021.1-beta10/lib/intel64 -Wl,--no-as-needed")

# ==================================================================================================

# LUD with the MKL library
if (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR} ${COMMON_INCLUDE_DIR}) 
	link_directories(${MKL_LIBRARY_DIR})
  #add_executable(${NAME} dss_sym_c.c)
  add_executable(${NAME} benchmark_dss.cpp)
	target_link_libraries(${NAME} mkl_intel_lp64 mkl_intel_thread iomp5 stdc++ mkl_core pthread m dl)
	install(TARGETS ${NAME} DESTINATION bin)
else ()
	message("## Skipping '${NAME}': no MKL support found")
  install(CODE "MESSAGE(\"${NAME} can only be built with MKL.\")")
endif()

unset(NAME)

# ==================================================================================================
