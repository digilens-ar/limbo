cmake_minimum_required (VERSION 3.20)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED VCPKG_OVERLAY_TRIPLETS)
    set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/custom-triplets/" CACHE STRING "")  # Use our locally stored custom triplets.
    set(VCPKG_TARGET_TRIPLET "x64-windows-limbo" CACHE STRING "")  
    set(VCPKG_HOST_TRIPLET "x64-windows-limbo" CACHE STRING "") 
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(GitVersionDetect)
set(LIMBO_VERSION_MAJOR ${GITVERSIONDETECT_VERSION_MAJOR})
set(LIMBO_VERSION_MINOR ${GITVERSIONDETECT_VERSION_MINOR})
set(LIMBO_VERSION_PATCH ${GITVERSIONDETECT_VERSION_PATCH})
set(LIMBO_VERSION ${GITVERSIONDETECT_VERSION_MAJOR}.${GITVERSIONDETECT_VERSION_MINOR}.${GITVERSIONDETECT_VERSION_PATCH})
set(LIMBO_VERSION_FULL ${GITVERSIONDETECT_VERSION})

project ("limbo" VERSION ${LIMBO_VERSION})
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

option(LIMBO_USE_NLOPT "Link to NLOPT and enable its algorithms for optimization" ON)
option(LIMBO_USE_CMAES "Link to libcmaes and enable its algorithms for optimization" OFF)

add_library(limbo INTERFACE)
file(GLOB_RECURSE LIMBO_SOURCE_FILES CONFIGURE_DEPENDS "src/*.hpp")
target_sources(limbo INTERFACE
	FILE_SET limbo_public_headers
	TYPE HEADERS
	BASE_DIRS src
	FILES ${LIMBO_SOURCE_FILES}
)
#target_include_directories(limbo INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(limbo INTERFACE "NOMINMAX")  # This prevents windows.h from defining unwanted macros
target_compile_definitions(limbo INTERFACE "_CRT_SECURE_NO_WARNINGS")  # This prevents windows from pointless deprecation warnings.
target_link_libraries(limbo INTERFACE Boost::boost spdlog::spdlog Eigen3::Eigen)

if(LIMBO_USE_NLOPT)
	find_package(NLopt CONFIG REQUIRED)
	target_compile_definitions(limbo INTERFACE "USE_NLOPT")
	target_link_libraries(limbo INTERFACE NLopt::nlopt)
endif()

if(LIMBO_USE_CMAES)
	find_package(libcmaes CONFIG REQUIRED)
	target_compile_definitions(limbo INTERFACE "USE_LIBCMAES")
	target_link_libraries(limbo INTERFACE libcmaes)
endif()

option(LIMBO_USE_TBB "Use Intel TBB for parallelization" ON)
if (LIMBO_USE_TBB)
	target_compile_definitions(limbo INTERFACE "LIMBO_USE_TBB")
	target_link_libraries(limbo INTERFACE TBB::tbb)
	install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb)
endif()

option(LIMBO_USE_INTEL_MKL "Build with Eigen using MKL as a backend" ON) #  This makes hyperparameter tuning much more efficient
if (LIMBO_USE_INTEL_MKL)
	if(NOT TARGET MKL::MKL) # MKL does not guard against issues if you `find` it twice
		set(MKL_LINK static)
		if (LIMBO_USE_TBB)
			set(MKL_THREADING tbb_thread)
		else()
			set(MKL_THREADING sequential)
		endif()
		set(MKL_INTERFACE lp64)  # Eigen documentation says this is required instead of default ilp64
		find_package(MKL CONFIG REQUIRED) # You must install MKL yourself from Intel's installer
	endif()
	target_link_libraries(limbo INTERFACE MKL::MKL)
	target_compile_definitions(limbo INTERFACE LIMBO_USE_INTEL_MKL)
endif()

option(LIMBO_BUILD_TESTS "Build tests for limbo" OFF)
if(LIMBO_BUILD_TESTS)
	include(CTest)  # This enables testing
	find_package(GTest CONFIG REQUIRED)
	add_subdirectory(tests)
	find_package(benchmark REQUIRED)
	add_subdirectory(benchmarks)
	add_subdirectory(util)
endif()

option(LIMBO_BUILD_TUTORIALS "Build tutorials for limbo" OFF)
if(LIMBO_BUILD_TUTORIALS)
	add_subdirectory(tutorials)
endif()

option(LIMBO_BUILD_EXAMPLES "Build examples for limbo" OFF)
if(LIMBO_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()


# Export Package
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(
    TARGETS limbo 
    EXPORT limbo-export
    FILE_SET limbo_public_headers
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

write_basic_package_version_file(
    limbo-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(
    EXPORT limbo-export
    NAMESPACE limbo::
    FILE limbo-targets.cmake
)

install(
    EXPORT limbo-export
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/limbo
    FILE limbo-targets.cmake
    NAMESPACE limbo::
)

install(
    FILES
        ${PROJECT_SOURCE_DIR}/cmake/limbo-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/limbo-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/limbo
)