cmake_minimum_required (VERSION 3.20)

add_executable(
	limbo_tests 
	"test.cpp"
	"test_serialize.cpp"
	"test_optimizers.cpp"
	"test_nlopt.cpp"
	"test_mean.cpp"
	"test_macros.cpp"
	"test_kernel.cpp"
	"test_init_functions.cpp"
)
target_compile_definitions(limbo_tests PRIVATE LIMBO_TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/temp")
target_link_libraries(limbo_tests PUBLIC limbo GTest::gtest_main)
install(TARGETS limbo_tests)

include(GoogleTest) 
gtest_discover_tests(limbo_tests)