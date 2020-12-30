find_package(Common)


SET(ISPC_VERSION_REQUIRED "1.8.2")

IF (NOT ISPC_EXECUTABLE)
  #  if ($ENV{ISPC})
  #  FIND_PROGRAM(ISPC_EXECUTABLE NAME ispc PATH $ENV{ISPC} DOC "Path to the ISPC executable.")
  # endif()
  FIND_PROGRAM(ISPC_EXECUTABLE ispc DOC "Path to the ISPC executable.")
  IF (NOT ISPC_EXECUTABLE)
    MESSAGE(STATUS  "Intel SPMD Compiler (ISPC) is not found.")
  ELSE()
    MESSAGE(STATUS "Found Intel SPMD Compiler (ISPC): ${ISPC_EXECUTABLE}")
    set(ISPC_FOUND True)
  ENDIF()
ENDIF()


if (NOT ISPC_EXECUTABLE)
  set(ISPC_FOUND False)
else(NOT ISPC_EXECUTABLE)
  set(ISPC_FOUND True)
  IF(NOT ISPC_VERSION)
    EXECUTE_PROCESS(COMMAND ${ISPC_EXECUTABLE} --version OUTPUT_VARIABLE ISPC_OUTPUT)
    STRING(REGEX MATCH " ([0-9]+[.][0-9]+[.][0-9]+)(dev|knl|ptx)? " DUMMY "${ISPC_OUTPUT}")
    SET(ISPC_VERSION ${CMAKE_MATCH_1})

    #IF (ISPC_VERSION VERSION_LESS ISPC_VERSION_REQUIRED)
      #      MESSAGE(FATAL_ERROR "Need at least version ${ISPC_VERSION_REQUIRED} of Intel SPMD Compiler (ISPC).")
      #    ENDIF()
    MESSAGE(STATUS "Found ISPC version ${ISPC_VERSION}")

    SET(ISPC_VERSION ${ISPC_VERSION} CACHE STRING "ISPC Version")
    MARK_AS_ADVANCED(ISPC_VERSION)
    MARK_AS_ADVANCED(ISPC_EXECUTABLE)
  ENDIF()

  function(ispc_compile _obj_list)
    get_sources_and_options(sources compile_flags "COMPILE_FLAGS" ${ARGN})

    set(__XEON__ True)
    
    IF (__XEON__)
      SET (ISPC_TARGET_EXT ${CMAKE_CXX_OUTPUT_EXTENSION})
    ELSE()
      SET (ISPC_TARGET_EXT .cpp)
      SET (ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --opt=force-aligned-memory)
    ENDIF()
    
    IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
      SET(ISPC_ARCHITECTURE "x86-64")
    ELSE()
      SET(ISPC_ARCHITECTURE "x86")
    ENDIF()
    
    SET(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${ISPC_TARGET_DIR})
    
    IF(ISPC_INCLUDE_DIR)
      STRING(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
      SET(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
    ENDIF()

    set(___XEON___ True)
    IF (__XEON__)
      STRING(REPLACE ";" "," ISPC_TARGET_ARGS "${ISPC_TARGETS}")
    ELSE()
      #    SET(ISPC_TARGET_ARGS generic-16)
      #SET(ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --emit-c++ -D__XEON_PHI__ --c++-include-file=${ISPC_DIR}/examples/intrinsics/knc.h)j
    ENDIF()
    
    SET(ISPC_OBJECTS ${${_obj_list}})

    foreach (src ${sources})
      get_filename_component(fname ${src} NAME_WE)
      get_filename_component(dir   ${src} PATH)
      
      IF("${dir}" STREQUAL "")
        SET(outdir ${ISPC_TARGET_DIR})
      ELSE("${dir}" STREQUAL "")
        SET(outdir ${ISPC_TARGET_DIR}/${dir})
      ENDIF("${dir}" STREQUAL "")
      SET(outdirh ${ISPC_TARGET_DIR})

      SET(deps "")
      IF (EXISTS ${outdir}/${fname}.dev.idep)
        FILE(READ ${outdir}/${fname}.dev.idep contents)
        STRING(REPLACE " " ";"     contents "${contents}")
        STRING(REPLACE ";" "\\\\;" contents "${contents}")
        STRING(REPLACE "\n" ";"    contents "${contents}")
        FOREACH(dep ${contents})
          IF (EXISTS ${dep})
            SET(deps ${deps} ${dep})
          ENDIF (EXISTS ${dep})
        ENDFOREACH(dep ${contents})
      ENDIF ()

      SET(results "${outdir}/${fname}.dev${ISPC_TARGET_EXT}")

      # if we have multiple targets add additional object files
      IF (__XEON__)
        LIST(LENGTH ISPC_TARGETS NUM_TARGETS)
        IF (NUM_TARGETS GREATER 1)
          FOREACH(target ${ISPC_TARGETS})
            SET(results ${results} "${outdir}/${fname}.dev_${target}${ISPC_TARGET_EXT}")
          ENDFOREACH()
        ENDIF()
      ENDIF()

      IF (WIN32)
        SET(ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --dllexport)
      ELSE()
        SET(ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --pic)
      ENDIF()

      separate_arguments(compile_flags)

      ADD_CUSTOM_COMMAND(
        OUTPUT ${results} ${outdirh}/${fname}_ispc.h
        COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
        COMMAND ${ISPC_EXECUTABLE}
        ${compile_flags}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${ISPC_INCLUDE_DIR_PARMS}
        --arch=${ISPC_ARCHITECTURE}
        #      --addressing=${EMBREE_ISPC_ADDRESSING}
        #      -O3
        #--target=${ISPC_TARGET_ARGS}
        #--woff
        # --opt=fast-math
        ${ISPC_ADDITIONAL_ARGS}
        -h ${outdirh}/${fname}_ispc.h
        -MMM  ${outdir}/${fname}.dev.idep
        -o ${outdir}/${fname}.dev${ISPC_TARGET_EXT}
        ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${deps}
        COMMENT "Building ISPC object ${outdir}/${fname}.dev${ISPC_TARGET_EXT}"
      )

      SET(ISPC_OBJECTS ${ISPC_OBJECTS} ${results})
    endforeach()

    set(${_obj_list} ${ISPC_OBJECTS} PARENT_SCOPE)
  endfunction()
endif(NOT ISPC_EXECUTABLE)
