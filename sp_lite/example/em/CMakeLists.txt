IF (CUDA_FOUND)
	CUDA_ADD_EXECUTABLE(sp_boris_yee  BorisYee.cu BorisYeeMain.cu ${SP_CAPI_CUDA_SRC})
 ELSE (CUDA_FOUND)
	ADD_LIBRARY(BorisYee   BorisYee.c    )
    TARGET_LINK_LIBRARIES(sp_boris_yee BorisYee)
ENDIF (CUDA_FOUND)

