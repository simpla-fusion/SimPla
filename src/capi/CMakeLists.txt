ADD_LIBRARY(sp_capi  spMesh.c spField.c spParticle.c spBucketFunction.c   )


IF (CUDA_FOUND)
	SET(SP_CAPI_CUDA_SRC spBucketFunction.cu  spMesh.cu spField.cu spParticle.cu  )
	CUDA_ADD_LIBRARY(sp_capi_cuda ${SP_CAPI_CUDA_SRC} )
    TARGET_LINK_LIBRARIES(sp_capi_cuda sp_capi -lcuda  ${CUDA_LIBRARIES})
ENDIF (CUDA_FOUND)

