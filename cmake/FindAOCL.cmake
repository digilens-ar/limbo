# find AOCL
if ( AOCL_FOUND )
  return()  # No need to find it again
endif()

if ( !WIN32 )
 message(FATAL_ERROR "Only Windows is supported for this find module currently.")
endif()

set ( __prefix "C:" )
set ( __aocl_dirs "" )
list ( APPEND __aocl_dirs "${__prefix}/Program Files/AMD/AOCL-Windows" )

find_path( AOCL_DIR NAMES amd-blis/include/LP64/blis.h HINTS ${__aocl_dirs} )

if(AOCL_FIND_ILP64 STREQUAL "ILP64")
set(__aocl_int_subdir "ILP64")
else()
set(__aocl_int_subdir "LP64")
endif()

if (AOCL_FIND_MT)
set(__aocl_mt_suffix "-Win-MT.lib")
else()
set(__aocl_mt_suffix "-Win.lib")
endif()

set(AOCL_LIB "AOCL_LIB-NOTFOUND" )
set(AOCL_DEBUG_LIB "AOCL_DEBUG_LIB-NOTFOUND" )

if (AOCL_DIR)
    set(AOCL_BLIS_DIR "${AOCL_DIR}/amd-blis")
    set(AOCL_LIBFLAME_DIR "${AOCL_DIR}/amd-libflame")
    set(AOCL_BLIS_INCLUDE_DIR "${AOCL_BLIS_DIR}/include/${__aocl_int_subdir}" )
    set(AOCL_LIBFLAME_INCLUDE_DIR "${AOCL_LIBFLAME_DIR}/include/${__aocl_int_subdir}" )
    set(AOCL_INCLUDE_DIRS "${AOCL_BLIS_INCLUDE_DIR}" "${AOCL_LIBFLAME_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Did not find AOCL include dirs")
endif()
 

find_library(AOCL_BLIS_LIB NAMES "AOCL-LibBlis${__aocl_mt_suffix}" HINTS ${AOCL_BLIS_DIR}/lib/${__aocl_int_subdir} )
find_library(AOCL_LIBFLAME_LIB NAMES "AOCL-LibFlame${__aocl_mt_suffix}" HINTS ${AOCL_LIBFLAME_DIR}/lib/${__aocl_int_subdir} )
if ( AOCL_BLIS_LIB AND AOCL_LIBFLAME_LIB )
    add_library( AOCL::BLIS STATIC IMPORTED)
    add_library( AOCL::LIBFLAME STATIC IMPORTED)
    set (AOCL_LIBRARIES AOCL::BLIS AOCL::LIBFLAME )

    set_target_properties( AOCL::BLIS PROPERTIES
	    IMPORTED_LOCATION ${AOCL_BLIS_LIB}  # Uhoh can't find debug
        INTERFACE_INCLUDE_DIRECTORIES "${AOCL_BLIS_INCLUDE_DIR}"
    )

    set_target_properties( AOCL::LIBFLAME PROPERTIES
	    IMPORTED_LOCATION ${AOCL_LIBFLAME_LIB}  # Uhoh can't find debug
        INTERFACE_INCLUDE_DIRECTORIES "${AOCL_LIBFLAME_INCLUDE_DIR}"
    )
    set(AOCL_FOUND TRUE )
else()
    message( FATAL_ERROR "AOCL not found." )
endif()