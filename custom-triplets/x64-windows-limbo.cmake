# Based off of x64-windows-static-md except specifying dynamic linkage for some libraries due to licensing constraints.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static) # Default to static linkage if not specified otherwise

# Dynamic link TBB since they don't support static (due to singleton usage.)
if(PORT MATCHES "tbb")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()