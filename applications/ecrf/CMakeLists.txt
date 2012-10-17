SET(MKL_LIBRARY_DIR   /opt/intel/composerxe/mkl/lib)
FIND_PACKAGE(IntelMKL)

ADD_EXECUTABLE(ecrf  ecrf.cpp)
TARGET_LINK_LIBRARIES(ecrf engine physics io fetl utilities  em ${MKL_STATIC_LIBRARIES} )