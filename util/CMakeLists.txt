cmake_minimum_required (VERSION 3.20)

add_subdirectory("external/cnpy")

add_executable(limbo_export_model 
"export_model_utility_main.cpp"
"CartesianGenerator.cpp"
 )

target_link_libraries(limbo_export_model PUBLIC limbo cnpy TBB::tbb TBB::tbbmalloc_proxy spdlog::spdlog)
install(TARGETS limbo_export_model)
install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbbmalloc_proxy TBB::tbb TBB::tbbmalloc)
