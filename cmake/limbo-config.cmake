include(CMakeFindDependencyMacro)
find_dependency(spdlog)
find_dependency(Boost)
find_dependency(TBB)
find_dependency(Eigen3)

include("${CMAKE_CURRENT_LIST_DIR}/limbo-targets.cmake")

check_required_components(limbo)