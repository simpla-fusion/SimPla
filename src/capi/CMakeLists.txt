ADD_LIBRARY(sp_capi  spMesh.c spField.c spParticle.c spBucketFunction.c   )


IF (CUDA_FOUND)
	CUDA_ADD_LIBRARY(sp_capi_cuda  spMesh.cu spField.cu spParticle.cu spBucketFunction.cu   )
    TARGET_LINK_LIBRARIES(sp_capi_cuda sp_capi -lcuda  ${CUDA_LIBRARIES})
ENDIF (CUDA_FOUND)

