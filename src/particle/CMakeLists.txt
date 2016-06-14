add_library(particle BucketContainer.c )
ADD_SUBDIRECTORY(test)

IF(CUDA_FOUND)
	CUDA_ADD_LIBRARY(particle_cuda BucketContainer.cu )
ENDIF(CUDA_FOUND)