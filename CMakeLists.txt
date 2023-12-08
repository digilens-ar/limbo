cmake_minimum_required (VERSION 3.20)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED VCPKG_OVERLAY_TRIPLETS)
    set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/custom-triplets/" CACHE STRING "")  # Use our locally stored custom triplets.
    set(VCPKG_TARGET_TRIPLET "x64-windows-limbo" CACHE STRING "")   # allows us to statically link protobuf without statically linking the runtime
    set(VCPKG_HOST_TRIPLET "x64-windows-limbo" CACHE STRING "")   # allows us to statically link protobuf without statically linking the runtime
endif()

project ("limbo")
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

option(LIMBO_USE_NLOPT "Link to NLOPT and enable its algorithms for optimization" ON)
option(LIMBO_USE_CMAES "Link to libcmaes and enable its algorithms for optimization" OFF)

add_library(limbo INTERFACE)
target_include_directories(limbo INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(limbo INTERFACE "NOMINMAX")  # This prevents windows.h from defining unwanted macros
target_compile_definitions(limbo INTERFACE "_CRT_SECURE_NO_WARNINGS")  # This prevents windows from pointless deprecation warnings.
#target_compile_definitions(limbo INTERFACE "USE_SFERES")
target_link_libraries(limbo INTERFACE Boost::boost spdlog::spdlog)

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

option(LIMBO_USE_INTEL_MKL "Build with Eigen using MKL as a backend" OFF) #  This makes hyperparameter tuning much more efficient
option(LIMBO_USE_OPENBLAS "Build with Eigen using OpenBlas as a backend" ON) #  This makes hyperparameter tuning much more efficient
if (LIMBO_USE_INTEL_MKL AND LIMBO_USE_OPENBLAS)
    message(FATAL_ERROR "Cannot enable LIMBO_USE_INTEL_MKL and LIMBO_USE_OPENBLAS simultaneously.")
endif()

if (LIMBO_USE_INTEL_MKL)
	if(NOT TARGET MKL::MKL) # MKL does not guard against issues if you `find` it twice
		set(MKL_LINK static)
		if (LIMBO_USE_TBB)
			set(MKL_THREADING tbb_thread)
		else()
			set(MKL_THREADING sequential)
		endif()
		set(MKL_INTERFACE lp64)  # Eigen documentation says this is required instead of default ilp64
		find_package(MKL CONFIG REQUIRED) # You mut install MKL yourself from Intel's installer
	endif()
	target_link_libraries(limbo INTERFACE MKL::MKL)
	target_compile_definitions(limbo INTERFACE LIMBO_USE_INTEL_MKL)
endif()
if (LIMBO_USE_OPENBLAS)
	find_package(OpenBLAS)
	target_link_libraries(limbo INTERFACE OpenBLAS::OpenBLAS)
	target_compile_definitions(limbo INTERFACE LIMBO_USE_OPENBLAS)
	target_compile_definitions(limbo INTERFACE LAPACK_COMPLEX_CUSTOM)  # See https://github.com/OpenMathLib/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio
	target_compile_definitions(limbo INTERFACE lapack_complex_float=_Fcomplex)
	target_compile_definitions(limbo INTERFACE lapack_complex_double=_Dcomplex)
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