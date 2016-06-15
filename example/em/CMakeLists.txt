

ADD_EXECUTABLE(demo_em demo_em.cpp)

target_compile_options(demo_em PUBLIC -DCARTESIAN_COORDINATE_SYTEM)

TARGET_LINK_LIBRARIES(demo_em simpla_lib)


IF (CUDA_FOUND)
	CUDA_ADD_EXECUTABLE(sp_boris_yee   BorisYeeMain.cu)
	CUDA_ADD_LIBRARY(BorisYee_cuda   BorisYee.cu    )
    TARGET_LINK_LIBRARIES(sp_boris_yee BorisYee_cuda sp_capi_cuda -lcuda  ${CUDA_LIBRARIES})
ELSE (CUDA_FOUND)
	ADD_LIBRARY(BorisYee   BorisYee.c    )
    TARGET_LINK_LIBRARIES(sp_boris_yee BorisYee)
ENDIF (CUDA_FOUND)

