This is a fork of [Limbo](https://github.com/resibots)
Limbo (LIbrary for Model-Based Optimization) is an open-source C++11 library for Gaussian Processes and data-efficient optimization (e.g., Bayesian optimization) that is designed to be both highly flexible and very fast. It can be used as a state-of-the-art optimization library or to experiment with novel algorithms with "plugin" components.

Notable Changes
------------------------
 - Various issues that prevented Limbo from compiling on Windows have been fixed
 - An issue that prevented Limbo from compiling with recent versions of TBB has been fixed
 - The WAF build system is replaced with CMake / vcpkg.
 - The Boost tests are translated to Gtest tests.
 - All Boost dependencies except for Boost::fusion are removed.
 - All `experimental` code is removed
 - The way that templated `Params` are passed to components of the optimizer is simplified
 - New stop conditions and status functions are added
 - C++20 `concepts` are used to make the templated code easier to read and write.
 - Assumption that printing messages to std::cerr is ok is replaced with usage of exceptions.
 - many instances of `operator()` are replaced with named methods.
 - Support for non-linear constraints is enhanced.
 - Added `EvaluationStatus` to allow objective functions to request a gracely end of optimization or skipping of sample.
 - Replaced usages of std::cout with `spdlog` logging library.

