cmake_minimum_required (VERSION 3.20)

add_executable(limbo_tests "test.cpp")
target_link_libaries(limbo_tests PUBLIC limbo GTest::gtest_main)
install(TARGETS limbo_tests)

include(GoogleTest) 
gtest_discover_tests(limbo_tests)