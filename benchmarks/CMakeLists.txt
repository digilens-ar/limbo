cmake_minimum_required (VERSION 3.20)

add_executable(
	limbo_benchmarks
	"limbo/hyperparametertuning.cpp"
)

target_link_libraries(limbo_benchmarks PUBLIC limbo benchmark::benchmark_main limbo)
install(TARGETS limbo_benchmarks)
