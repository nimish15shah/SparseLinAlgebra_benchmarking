
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Cedric Nugteren <cedric.nugteren@surfsara.nl>
#   Evghenii Gaburov <evghenii.gaburov@surfsara.nl>
#
# ==================================================================================================

# Includes only once
if (NOT COMMON_CMAKE_SET)
  set(COMMON_CMAKE_SET True)

  # ================================================================================================

  # RPATH settings
  set(CMAKE_SKIP_BUILD_RPATH false) # Use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_BUILD_WITH_INSTALL_RPATH false) # When building, don't use the install RPATH already
  set(CMAKE_INSTALL_RPATH "") # The RPATH to be used when installing
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH false) # Don't add the automatically determined parts

  # ================================================================================================

  # Package scripts location
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_LIST_DIR}/Modules/"
    "${CMAKE_CURRENT_LIST_DIR}/")

  # ================================================================================================

endif()

# ==================================================================================================
