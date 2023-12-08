cmake_minimum_required (VERSION 3.20)

add_executable(limbo_tutorial_basic  "basic_example.cpp")
add_executable(limbo_tutorial_gp  "gp.cpp")
add_executable(limbo_tutorial_opt  "opt.cpp")
add_executable(limbo_tutorial_stat  "statistics.cpp")

set(tutorial_targets limbo_tutorial_basic limbo_tutorial_gp limbo_tutorial_opt limbo_tutorial_stat)

if(LIMBO_USE_CMAES)
	add_executable(limbo_tutorial_advanced  "advanced_example.cpp")
	list(APPEND tutorial_targets limbo_tutorial_advanced)
endif()

foreach(targ ${tutorial_targets})
	target_link_libraries(${targ} PUBLIC limbo)
	if (LIMBO_USE_INTEL_MKL)
		target_compile_definitions(${targ} PRIVATE EIGEN_USE_MKL_ALL)
	endif()
	if (LIMBO_USE_OPENBLAS)
		target_compile_definitions(${targ} PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
	endif()
endforeach()
install(TARGETS ${tutorial_targets})
