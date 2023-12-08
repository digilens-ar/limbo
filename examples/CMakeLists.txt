cmake_minimum_required (VERSION 3.20)

add_executable(limbo_example_mono_dim  "mono_dim.cpp")
add_executable(limbo_example_obs_multi  "obs_multi.cpp")
add_executable(limbo_example_obs_multi_auto_mean  "obs_multi_auto_mean.cpp")

set(example_targets limbo_example_mono_dim limbo_example_obs_multi limbo_example_obs_multi_auto_mean)

foreach(targ ${example_targets})
	target_link_libraries(${targ} PUBLIC limbo)
	if (LIMBO_USE_BLAS STREQUAL "MKL")
		target_compile_definitions(${targ} PRIVATE EIGEN_USE_MKL_ALL)
	elseif (LIMBO_USE_BLAS STREQUAL "AOCL")
		target_compile_definitions(${targ} PRIVATE EIGEN_USE_BLAS)
		target_compile_definitions(${targ} PRIVATE EIGEN_USE_LAPACKE)
	endif()
endforeach()
install(TARGETS ${example_targets})
