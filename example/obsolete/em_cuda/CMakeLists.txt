IF (CUDA_FOUND)
	FILE(GLOB_RECURSE sp_boris_yee_SRC ./*.cu )
	CUDA_ADD_EXECUTABLE(sp_boris_yee  ${sp_boris_yee_SRC} )
	TARGET_LINK_LIBRARIES(sp_boris_yee    cuda  sp_api_cuda io data_model ${CUDA_LIBRARIES})	
ENDIF (CUDA_FOUND)

