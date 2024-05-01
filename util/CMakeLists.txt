cmake_minimum_required (VERSION 3.20)

add_subdirectory("external/cnpy")

add_executable(limbo_export_model "export_model_utility.cpp")
target_link_libraries(limbo_export_model PUBLIC limbo cnpy)
install(TARGETS limbo_export_model)