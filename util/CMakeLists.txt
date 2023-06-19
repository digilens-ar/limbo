cmake_minimum_required (VERSION 3.20)

add_subdirectory("external/cnpy")

add_executable(limbo_export_model "export_model_utility.cpp")
target_compile_definitions(limbo_export_model PRIVATE LIMBO_TEST_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
target_link_libraries(limbo_export_model PUBLIC limbo cnpy)
install(TARGETS limbo_export_model)