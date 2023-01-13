
cmake_minimum_required (VERSION 3.20)

project ("limbo")
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED COMPONENTS filesystem program_options thread system)
find_package(TBB CONFIG REQUIRED)
find_package(NLopt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(limbo INTERFACE)
target_include_directories(limbo INTERFACE "${CMAKE_SOURCE_DIR}/src")
target_compile_definitions(limbo INTERFACE "USE_NLOPT")
#target_compile_definitions(limbo INTERFACE "USE_SFERES")
#target_compile_definitions(limbo INTERFACE "USE_LIBCMAES")
target_compile_definitions(limbo INTERFACE "USE_TBB")
target_link_libraries(limbo INTERFACE NLopt::nlopt)
target_link_libraries(limbo INTERFACE NLopt::nlopt Boost::boost Boost::filesystem Boost::system Boost::program_options Boost::thread)

option(LIMBO_BUILD_TESTS "Biuld tests for limbo" OFF)
if(LIMBO_BUILD_TESTS)
	find_package(GTest CONFIG REQUIRED)
	add_subdirectory(tests)
endif()