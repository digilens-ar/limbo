This is a fork of [Limbo](https://github.com/resibots)
Limbo (LIbrary for Model-Based Optimization) is an open-source C++11 library for Gaussian Processes and data-efficient optimization (e.g., Bayesian optimization) that is designed to be both highly flexible and very fast. It can be used as a state-of-the-art optimization library or to experiment with novel algorithms with "plugin" components.

Notable Changes
------------------------
 - Various issues that prevented Limbo from compiling on Windows have been fixed
 - An issue that prevented Limbo from compiling with recent versions of TBB has been fixed
 - The WAF build system is replaced with CMake / vcpkg.
 - The Boost tests are translated to Gtest tests.
 - All Boost dependencies except for Boost::fusion are removed.
