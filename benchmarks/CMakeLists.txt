cmake_minimum_required (VERSION 3.20)

add_executable(
	limbo_benchmarks
	"limbo/hyperparametertuning.cpp"
	"limbo/bench.cpp"
)

if (LIMBO_USE_INTEL_MKL)
	target_compile_definitions(limbo_benchmarks PRIVATE EIGEN_USE_MKL_ALL)
endif()

target_link_libraries(limbo_benchmarks PRIVATE limbo benchmark::benchmark_main)
install(TARGETS limbo_benchmarks)
