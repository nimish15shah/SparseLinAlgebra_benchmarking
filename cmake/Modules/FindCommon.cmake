
function(clone_source src dst)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${dst}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${CMAKE_CURRENT_BINARY_DIR}/${dst}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    COMMENT "Clone ${src} to ${dst}"
    )
endfunction()

function(dummy_install target_name lib_name)
  install(CODE "MESSAGE(\"${target_name} can only be built with ${lib_name}.\")")
endfunction()

macro(get_sources_and_options _sources _option_list _option_name)
  set( ${_sources} )
  set( ${_option_list} )
  set( _found_options False)
  foreach(arg ${ARGN})
    if ("x${arg}" STREQUAL "x${_option_name}")
      set (_found_options True)
    else()
      if (_found_options)
        list(APPEND ${_option_list} ${arg})
      else()
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

# MACRO picks correct compiler flags. For example
# select_compiler_flags(cxx_flags 
#    GNU "-std=c++11 -mnative -Wall -Werror"
#    Intel "-std=c++11 -axavx,core-avx2"
#    CLANG "-std=c++11 -Weverything"
#    PGI "-std=c++11" 
# )
macro(select_compiler_flags _flags)
  set( ${_flags} )

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(_compiler "GNU")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(_compiler "CLANG")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(_compiler "CLANG")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(_compiler "Intel")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(_compiler "MSCV")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(_compiler "PGI")
  endif()
 
  set (_found_compiler False)
  foreach(arg ${ARGN})
    if ("x${arg}" STREQUAL "x${_compiler}")
      set(_found_compiler True)
    else()
      if (_found_compiler)
        set(${_flags} ${arg})
        set(_found_compiler False)
      endif()
    endif()
  endforeach()
endmacro()

