SET(MKL_LIBRARY_DIR   /opt/intel/composerxe/mkl/lib)
FIND_PACKAGE(IntelMKL)

include_directories(BEFORE ${MKL_C_INLCUDE_DIR} )

#add_library(pic  pic.cpp misc.cpp gyro_gauge.cpp delta_f.cpp)
# 	 	                      
#TARGET_LINK_LIBRARIES( pic  ${MKL_STATIC_LIBRARIES} )
 	

cxx_test(pic_test ${MKL_STATIC_LIBRARIES} )  
