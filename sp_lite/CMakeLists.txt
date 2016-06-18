FILE(GLOB_RECURSE sp_cpp_SRC ./*.cpp)
ADD_LIBRARY(sp_cpp ${sp_cpp_SRC} )


IF(CUDA_FOUND)
	FILE(GLOB_RECURSE SP_API_CUDA_SRC ./*.cu )

	CUDA_ADD_LIBRARY(sp_api_cuda ${SP_API_CUDA_SRC} )

	TARGET_LINK_LIBRARIES(sp_api_cuda sp_cpp  io data_model ${CUDA_LIBRARIES})
 
ENDIF(CUDA_FOUND)
 