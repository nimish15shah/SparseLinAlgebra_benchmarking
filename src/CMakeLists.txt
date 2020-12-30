
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================

# CMake project
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)
project("src")
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

# ==================================================================================================

# Dwarf 2: Sparse Linear Algebra
message("--------------------")
message("Dwarf 2: Sparse Linear Algebra:")
message("--------------------")
set(DWARF_PREFIX 2_sparse) # The prefix of the name of the binaries produced

# Add the examples
#add_subdirectory(krylov)
#add_subdirectory(spgemm/mkl_shmem)
#add_subdirectory(spgemm/mkl_xphi)
#add_subdirectory(spmv/mkl_shmem)
add_subdirectory(cpu_mkl)

# ==================================================================================================




