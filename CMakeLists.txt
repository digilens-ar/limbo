cmake_minimum_required (VERSION 3.20)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED VCPKG_OVERLAY_TRIPLETS)
    set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/custom-triplets/" CACHE STRING "")  # Use our locally stored custom triplets.
    set(VCPKG_TARGET_TRIPLET "x64-windows-limbo" CACHE STRING "")   # allows us to statically link protobuf without statically linking the runtime
    set(VCPKG_HOST_TRIPLET "x64-windows-limbo" CACHE STRING "")   # allows us to statically link protobuf without statically linking the runtime
endif()

project ("limbo")
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED COMPONENTS filesystem program_options thread system)
find_package(TBB CONFIG REQUIRED)
find_package(NLopt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(limbo INTERFACE)
target_include_directories(limbo INTERFACE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(limbo INTERFACE "NOMINMAX")  # This prevents windows.h from defining unwanted macros
target_compile_definitions(limbo INTERFACE "USE_NLOPT")
#target_compile_definitions(limbo INTERFACE "USE_SFERES") # I think these are mutually exclusive with NLOPT
#target_compile_definitions(limbo INTERFACE "USE_LIBCMAES")
target_link_libraries(limbo INTERFACE NLopt::nlopt)
target_link_libraries(limbo INTERFACE NLopt::nlopt Boost::boost Boost::filesystem Boost::system Boost::program_options Boost::thread)

option(LIMBO_USE_TBB "Use Intel TBB for parallelization" OFF)
if (LIMBO_USE_TBB)
target_compile_definitions(limbo INTERFACE "USE_TBB") # Bugs need to be fixed before this works
target_link_libraries(limbo INTERFACE TBB::tbb)
install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb)
endif()

option(LIMBO_BUILD_TESTS "Build tests for limbo" OFF)
if(LIMBO_BUILD_TESTS)
	find_package(GTest CONFIG REQUIRED)
	add_subdirectory(tests)
endif()