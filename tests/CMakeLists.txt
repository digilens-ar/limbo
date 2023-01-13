cmake_minimum_required (VERSION 3.20)

add_executable(
	limbo_tests 
	"test.cpp"
	"test_serialize.cpp"
	"test_optimizers.cpp"
	"test_nlopt.cpp"
)
target_link_libraries(limbo_tests PUBLIC limbo GTest::gtest_main)
install(TARGETS limbo_tests)

include(GoogleTest) 
gtest_discover_tests(limbo_tests)