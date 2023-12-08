cmake_minimum_required (VERSION 3.20)

add_executable(
	limbo_benchmarks
	"limbo/hyperparametertuning.cpp"
)

if (LIMBO_USE_BLAS STREQUAL "MKL")
	target_compile_definitions(limbo_benchmarks PRIVATE EIGEN_USE_MKL_ALL)
elseif (LIMBO_USE_BLAS STREQUAL "AOCL")
	target_compile_definitions(limbo_benchmarks PRIVATE EIGEN_USE_BLAS)
	target_compile_definitions(limbo_benchmarks PRIVATE EIGEN_USE_LAPACKE)
endif()

target_link_libraries(limbo_benchmarks PUBLIC limbo benchmark::benchmark_main limbo)
install(TARGETS limbo_benchmarks)
