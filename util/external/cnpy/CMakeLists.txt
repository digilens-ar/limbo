CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

find_package(ZLIB REQUIRED)

add_library(cnpy "cnpy.cpp")
target_link_libraries(cnpy ZLIB::ZLIB)
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cnpy PUBLIC _CRT_SECURE_NO_WARNINGS) # Windows don't warn that fopen isn't safe

install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)


#install(FILES "cnpy.h" DESTINATION include)
#install(FILES "mat2npz" "npy2mat" "npz2mat" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

option(CNPY_BUILD_EXAMPLES "Build the example for CNPY" ON)
if (CNPY_BUILD_EXAMPLES)
	add_executable(example1 example1.cpp)
	target_link_libraries(example1 cnpy)
endif()
