IF (CUDA_FOUND)
	CUDA_ADD_EXECUTABLE(sp_boris_yee  BorisYee.cu BorisYeeMain.cu ${SP_API_CUDA_SRC})

ENDIF (CUDA_FOUND)

